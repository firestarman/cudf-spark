/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

import java.util
import java.util.concurrent.{Callable, ConcurrentLinkedQueue, Future, TimeUnit}

import scala.util.control.NonFatal

import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import com.nvidia.spark.rapids.io.async.{ThrottlingExecutor, TrafficController}

import org.apache.spark.TaskContext
import org.apache.spark.sql.rapids.execution.TrampolineUtil
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch


/**
 * Similar as GpuShuffleCoalesceIterator, but pulling in host batches asynchronously, to
 * overlap the host batch reading and the downstream GPU operations.
 *
 * @note this iterator is not thread safe.
 */
class GpuShuffleAsyncCoalesceIterator(iter: Iterator[CoalescedHostResult],
    dataTypes: Array[DataType],
    targetBatchSize: Long,
    outputBatchesMetric: GpuMetric = NoopMetric,
    outputRowsMetric: GpuMetric = NoopMetric,
    asyncReadTimeMetric: GpuMetric = NoopMetric,
    opTimeMetric: GpuMetric = NoopMetric,
    readThrottlingMetric: GpuMetric = NoopMetric,
    readTimeMetric: GpuMetric = NoopMetric,
    semReadTimeMetric: GpuMetric = NoopMetric,
    bgJobLaunchTimeMetric: GpuMetric = NoopMetric,
    prefetchBatches: Int = 1,
    prefetchWaitTimeMetric: GpuMetric = NoopMetric,
    prefetchWaitCountMetric: GpuMetric = NoopMetric
) extends Iterator[ColumnarBatch] with AutoCloseable {

  require(prefetchBatches > 0, "Shuffle prefetch batch count must be positive")

  // Don't install the callback if in a unit test
  Option(TaskContext.get()).foreach { tc =>
    onTaskCompletion(tc)(close())
  }

  val executor =
    new ThrottlingExecutor(
      TrampolineUtil.newDaemonCachedThreadPool(
        "async shuffle read thread for " + Thread.currentThread().getName, 1, 1),
      TrafficController.getReadInstance,
      stat => {
        readThrottlingMetric.add(stat.accumulatedThrottleTimeNs)
      }
    )

  override def close(): Unit = {
    closed = true
    try {
      executor.shutdownNow(10, TimeUnit.SECONDS)
    } finally {
      readLock.synchronized {
        val results = new scala.collection.mutable.ArrayBuffer[AutoCloseable]()
        var result = prefetchedResults.poll()
        while (result != null) {
          results += result.batch
          result = prefetchedResults.poll()
        }
        readFutures.clear()
        results.safeClose()
      }
    }
  }

  // Identity-based ownership avoids confusing distinct but equal rows-only results during close.
  private class ReadResult(val batch: CoalescedHostResult, val launchTime: Long)

  private val readFutures = new util.ArrayDeque[Future[Option[ReadResult]]]
  private val prefetchedResults = new ConcurrentLinkedQueue[ReadResult]
  private val readLock = new Object
  @volatile private var closed = false
  @volatile private var inputExhausted = false
  private var prefetchStarted = false

  private class ReadCallable extends Callable[Option[ReadResult]] {
    private val submittedAt = System.nanoTime()

    // The actual async read, including the host batches read and concatenation in
    // "HostCoalesceIteratorBase.next()".
    override def call(): Option[ReadResult] = readLock.synchronized {
      val launchTime = System.nanoTime() - submittedAt
      if (closed || inputExhausted) {
        None
      } else try {
        NvtxRegistry.ASYNC_SHUFFLE_READ {
          if (iter.hasNext) {
            val batch = iter.next()
            closeOnExcept(batch) { _ =>
              val result = new ReadResult(batch, launchTime)
              prefetchedResults.add(result)
              Some(result)
            }
          } else {
            inputExhausted = true
            None
          }
        }
      } catch {
        case NonFatal(e) =>
          inputExhausted = true
          throw e
      }
    }
  }

  private def fillPrefetchQueue(): Unit = {
    prefetchStarted = true
    while (readFutures.size() < prefetchBatches && !inputExhausted && !closed) {
      // Reserve an estimate for each host coalesce job. The inner input buffering executor
      // separately accounts for its single input prefetch, as in the original two-stage pipeline.
      readFutures.addLast(executor.submitWithDeferredReservation(new ReadCallable, targetBatchSize))
    }
  }

  private def withReadMetrics[R](f: => R): R = opTimeMetric.ns {
    asyncReadTimeMetric.ns {
      GpuMetric.withSemaphoreTime(readTimeMetric, semReadTimeMetric, TaskContext.get()) {
        f
      }
    }
  }

  override def hasNext: Boolean = !closed && withReadMetrics {
    if (prefetchStarted) {
      while (!readFutures.isEmpty && ShufflePrefetchWaitMetrics.await(
          readFutures.peekFirst(), prefetchWaitTimeMetric, prefetchWaitCountMetric).isEmpty) {
        readFutures.removeFirst()
      }
      !readFutures.isEmpty
    } else {
      iter.hasNext
    }
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) {
      throw new NoSuchElementException("No more batches")
    }
    NvtxRegistry.ASYNC_SHUFFLE_READ {
      val hostConcatedRet = withReadMetrics {
        if (!readFutures.isEmpty) {
          val result = ShufflePrefetchWaitMetrics.await(
            readFutures.removeFirst(), prefetchWaitTimeMetric, prefetchWaitCountMetric).get
          prefetchedResults.remove(result)
          bgJobLaunchTimeMetric.add(result.launchTime)
          result.batch
        } else { // The first batch, just read it directly
          iter.next()
        }
      }
      val gpuCB = withResource(hostConcatedRet) { _ =>
        // We acquire the GPU regardless of whether the concatenated batch is an empty batch
        // or not, because the downstream tasks expect the `GpuShuffleCoalesceIterator`
        // to acquire the semaphore and may generate GPU data from batches that are empty.
        GpuSemaphore.acquireIfNecessary(TaskContext.get())
        GpuMetric.ns(opTimeMetric)(hostConcatedRet.toGpuBatch(dataTypes))
      }
      closeOnExcept(gpuCB) { _ =>
        GpuMetric.ns(opTimeMetric) {
          // Only the single background reader touches iter after prefetch starts.
          fillPrefetchQueue()
          outputBatchesMetric += 1
          outputRowsMetric += gpuCB.numRows()
          gpuCB
        }
      }
    }
  }

}
