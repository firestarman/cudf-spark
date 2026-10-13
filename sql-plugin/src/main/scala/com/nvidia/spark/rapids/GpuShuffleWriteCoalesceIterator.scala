/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import scala.collection.AbstractIterator
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import com.nvidia.spark.rapids.Arm.closeOnExcept
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion

import org.apache.spark.TaskContext
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuShuffleWriteCoalesceIterator {
  val MIN_TARGET_SIZE: Long = 32L * 1024

  def shouldCoalesce(
      enabled: Boolean,
      useMultiThreadedShuffle: Boolean,
      useGpuShuffle: Boolean,
      useKudoGpuSlicing: Boolean): Boolean = {
    enabled && useMultiThreadedShuffle && !useGpuShuffle && !useKudoGpuSlicing
  }

  def perTaskBudget(totalBudget: Long, parallelism: Int, numPartitions: Int): Long = {
    require(totalBudget > 0, "Total shuffle write coalesce budget must be positive")
    require(parallelism > 0, "Shuffle write parallelism must be positive")
    require(numPartitions > 0, "Shuffle partition count must be positive")
    val configuredPerTask = math.max(totalBudget / parallelism, 1L)
    val minimumPerTask = saturatedMultiply(MIN_TARGET_SIZE, numPartitions)
    math.max(configuredPerTask, minimumPerTask)
  }

  def targetSize(
      configuredTarget: Long,
      taskBudget: Long,
      numPartitions: Int): Long = {
    require(configuredTarget > 0, "Shuffle write coalesce target must be positive")
    require(taskBudget > 0, "Shuffle write coalesce task budget must be positive")
    require(numPartitions > 0, "Shuffle partition count must be positive")
    math.max(math.min(configuredTarget, taskBudget / numPartitions), MIN_TARGET_SIZE)
  }

  private def saturatedMultiply(value: Long, multiplier: Int): Long = {
    if (value > Long.MaxValue / multiplier) {
      Long.MaxValue
    } else {
      value * multiplier
    }
  }
}

/**
 * Coalesces small CPU-sliced shuffle batches by reduce partition before serialization.
 *
 * The input iterator owns each batch until it advances to the next partitioned input batch.
 * This iterator retains every buffered batch and releases those references after concatenation.
 */
class GpuShuffleWriteCoalesceIterator(
    input: Iterator[Product2[Int, ColumnarBatch]],
    targetSize: Long,
    maxBufferedSize: Long,
    outputBatchesMetric: GpuMetric = NoopMetric,
    outputRowsMetric: GpuMetric = NoopMetric,
    concatTimeMetric: GpuMetric = NoopMetric)
  extends AbstractIterator[Product2[Int, ColumnarBatch]] with AutoCloseable {

  require(targetSize > 0, "Shuffle write coalesce target must be positive")
  require(maxBufferedSize >= targetSize,
    "Shuffle write coalesce buffer limit must be at least the target size")

  private case class PendingBatches(
      var size: Long,
      var rows: Long,
      batches: ArrayBuffer[ColumnarBatch])

  private val pending = new mutable.LinkedHashMap[Int, PendingBatches]()
  private val ready = new mutable.Queue[(Int, PendingBatches)]()
  private var bufferedSize = 0L
  private var current: Option[(Int, ColumnarBatch)] = None
  private var currentConsumed = true
  private var closed = false

  Option(TaskContext.get()).foreach { taskContext =>
    onTaskCompletion(taskContext)(close())
  }

  private def batchSize(batch: ColumnarBatch): Long = {
    if (batch.numCols() == 0) {
      math.max(batch.numRows().toLong, 1L)
    } else {
      batch.column(0) match {
        case _: SlicedGpuColumnVector =>
          SlicedGpuColumnVector.getTotalHostMemoryUsed(batch)
        case other =>
          throw new IllegalArgumentException(
            s"Unsupported shuffle column for write coalescing: ${other.getClass.getName}")
      }
    }
  }

  private def retain(batch: ColumnarBatch): ColumnarBatch = {
    if (batch.numCols() == 0) {
      batch
    } else {
      SlicedGpuColumnVector.incRefCount(batch)
    }
  }

  private def enqueueReady(partitionId: Int, batches: PendingBatches): Unit = {
    ready.enqueue((partitionId, batches))
  }

  private def flushOldestPending(): Unit = {
    pending.headOption.foreach { case (partitionId, batches) =>
      pending.remove(partitionId)
      enqueueReady(partitionId, batches)
    }
  }

  private def addInput(partitionId: Int, batch: ColumnarBatch): Unit = {
    if (batch.numRows() > 0) {
      val size = batchSize(batch)
      val existing = pending.remove(partitionId)
      existing.foreach { accumulated =>
        if (accumulated.rows + batch.numRows() > Int.MaxValue) {
          enqueueReady(partitionId, accumulated)
        }
      }
      val accumulated = existing.filter(_.rows + batch.numRows() <= Int.MaxValue)
        .getOrElse(PendingBatches(0L, 0L, new ArrayBuffer[ColumnarBatch]()))
      accumulated.batches += retain(batch)
      accumulated.size += size
      accumulated.rows += batch.numRows()
      bufferedSize += size
      if (accumulated.size >= targetSize) {
        enqueueReady(partitionId, accumulated)
      } else {
        pending.put(partitionId, accumulated)
      }
      if (ready.isEmpty && bufferedSize >= maxBufferedSize) {
        flushOldestPending()
      }
    }
  }

  private def fillReady(): Unit = {
    while (ready.isEmpty && input.hasNext) {
      val next = input.next()
      addInput(next._1, next._2)
    }
    if (ready.isEmpty) {
      flushOldestPending()
    }
  }

  private def nextCoalesced(): Option[(Int, ColumnarBatch)] = {
    fillReady()
    if (ready.nonEmpty) {
      val (partitionId, accumulated) = ready.dequeue()
      bufferedSize -= accumulated.size
      val batch = concatTimeMetric.ns {
        GpuBatchUtils.concatShuffleBatchesAndClose(accumulated.batches.toSeq)
      }
      closeOnExcept(batch) { _ =>
        outputBatchesMetric += 1
        outputRowsMetric += batch.numRows()
        Some((partitionId, batch))
      }
    } else {
      None
    }
  }

  private def closeCurrent(): Unit = {
    current.foreach(_._2.close())
    current = None
  }

  override def hasNext: Boolean = {
    if (!closed && currentConsumed) {
      closeCurrent()
      try {
        current = nextCoalesced()
        currentConsumed = false
      } catch {
        case t: Throwable =>
          close()
          throw t
      }
    }
    !closed && current.isDefined
  }

  override def next(): Product2[Int, ColumnarBatch] = {
    if (!hasNext) {
      throw new NoSuchElementException("No more coalesced shuffle write batches")
    }
    currentConsumed = true
    current.get
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      closeCurrent()
      pending.valuesIterator.flatMap(_.batches).toSeq.safeClose()
      ready.iterator.flatMap(_._2.batches).toSeq.safeClose()
      pending.clear()
      ready.clear()
      bufferedSize = 0L
    }
  }
}
