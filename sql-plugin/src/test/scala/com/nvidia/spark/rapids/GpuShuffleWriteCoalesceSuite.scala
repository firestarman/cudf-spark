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

import java.lang.{Integer => JInt}
import java.util.{Arrays, List => JList}

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{DType, HostColumnVector}
import ai.rapids.cudf.HostColumnVector.{BasicType, ListType}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.types.{ArrayType, IntegerType, StringType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

class GpuShuffleWriteCoalesceSuite extends AnyFunSuite {

  private case class TestRow(
      intValue: Option[Int],
      stringValue: Option[String],
      listValue: Option[Seq[Option[Int]]])

  private def boxed(value: Int): JInt = JInt.valueOf(value)

  private def list(values: JInt*): JList[JInt] = Arrays.asList(values: _*)

  private def makeBatch(
      integers: Array[JInt],
      strings: Array[String],
      lists: Array[JList[JInt]]): ColumnarBatch = {
    require(integers.length == strings.length && strings.length == lists.length)
    val hostColumns = closeOnExcept(new Array[RapidsHostColumnVector](3)) { columns =>
      columns(0) = new RapidsHostColumnVector(
        IntegerType, HostColumnVector.fromBoxedInts(integers: _*))
      columns(1) = new RapidsHostColumnVector(
        StringType, HostColumnVector.fromStrings(strings: _*))
      val listType = new ListType(true, new BasicType(true, DType.INT32))
      columns(2) = new RapidsHostColumnVector(
        ArrayType(IntegerType, containsNull = true),
        HostColumnVector.fromLists(listType, lists: _*))
      columns
    }
    val slicedColumns = withResource(hostColumns) { columns =>
      closeOnExcept(new Array[ColumnVector](columns.length)) { slices =>
        columns.indices.foreach { index =>
          slices(index) = new SlicedGpuColumnVector(columns(index), 0, integers.length)
        }
        slices
      }
    }
    new ColumnarBatch(slicedColumns, integers.length)
  }

  private def extractRows(batch: ColumnarBatch): Seq[TestRow] = {
    (0 until batch.numRows()).map { row =>
      val intColumn = batch.column(0)
      val stringColumn = batch.column(1)
      val listColumn = batch.column(2)
      val intValue = if (intColumn.isNullAt(row)) None else Some(intColumn.getInt(row))
      val stringValue = if (stringColumn.isNullAt(row)) {
        None
      } else {
        Some(stringColumn.getUTF8String(row).toString)
      }
      val listValue = if (listColumn.isNullAt(row)) {
        None
      } else {
        val array = listColumn.getArray(row)
        Some((0 until array.numElements()).map { index =>
          if (array.isNullAt(index)) None else Some(array.getInt(index))
        })
      }
      TestRow(intValue, stringValue, listValue)
    }
  }

  private def retain(batch: ColumnarBatch): ColumnarBatch = {
    if (batch.numCols() == 0) {
      batch
    } else {
      SlicedGpuColumnVector.incRefCount(batch)
    }
  }

  test("coalesce host shuffle batches by partition and update final output metrics") {
    val first = makeBatch(
      Array(boxed(1), null),
      Array("one", null),
      Array(list(boxed(1), boxed(2)), null))
    val otherPartition = makeBatch(
      Array(boxed(9)),
      Array("nine"),
      Array(list(boxed(9))))
    val second = makeBatch(
      Array(boxed(3), boxed(4)),
      Array("three", "four"),
      Array(list(boxed(3), null), list()))
    val inputs = Seq(first, otherPartition, second)
    val outputBatches = new LocalGpuMetric
    val outputRows = new LocalGpuMetric
    val concatTime = new LocalGpuMetric

    withResource(inputs) { _ =>
      val iterator = new GpuShuffleWriteCoalesceIterator(
        Iterator((0, first), (1, otherPartition), (0, second)),
        targetSize = 1024 * 1024,
        maxBufferedSize = 2 * 1024 * 1024,
        outputBatchesMetric = outputBatches,
        outputRowsMetric = outputRows,
        concatTimeMetric = concatTime)
      val outputs = new ArrayBuffer[(Int, ColumnarBatch)]()
      try {
        while (iterator.hasNext) {
          val next = iterator.next()
          outputs += ((next._1, retain(next._2)))
        }
        withResource(outputs.map(_._2)) { _ =>
          val rowsByPartition = outputs.map { case (partitionId, batch) =>
            partitionId -> extractRows(batch)
          }.toMap
          assert(rowsByPartition(0) == Seq(
            TestRow(Some(1), Some("one"), Some(Seq(Some(1), Some(2)))),
            TestRow(None, None, None),
            TestRow(Some(3), Some("three"), Some(Seq(Some(3), None))),
            TestRow(Some(4), Some("four"), Some(Seq.empty))))
          assert(rowsByPartition(1) == Seq(
            TestRow(Some(9), Some("nine"), Some(Seq(Some(9))))))
          assert(outputBatches.value == 2)
          assert(outputRows.value == 5)
          assert(concatTime.value >= 0)
        }
      } finally {
        iterator.close()
      }
    }
  }

  test("global buffer limit flushes the oldest partition") {
    val first = makeBatch(Array(boxed(1)), Array("one"), Array(list(boxed(1))))
    val second = makeBatch(Array(boxed(2)), Array("two"), Array(list(boxed(2))))
    withResource(Seq(first, second)) { _ =>
      val firstSize = SlicedGpuColumnVector.getTotalHostMemoryUsed(first)
      val secondSize = SlicedGpuColumnVector.getTotalHostMemoryUsed(second)
      val target = firstSize + secondSize
      val iterator = new GpuShuffleWriteCoalesceIterator(
        Iterator((0, first), (1, second)), target, target)
      try {
        assert(iterator.hasNext)
        val output = iterator.next()
        assert(output._1 == 0)
        assert(extractRows(output._2).map(_.intValue) == Seq(Some(1)))
      } finally {
        iterator.close()
      }
    }
  }

  test("coalesce rows-only batches") {
    val first = new ColumnarBatch(Array.empty[ColumnVector], 3)
    val second = new ColumnarBatch(Array.empty[ColumnVector], 4)
    val outputBatches = new LocalGpuMetric
    val outputRows = new LocalGpuMetric
    val iterator = new GpuShuffleWriteCoalesceIterator(
      Iterator((2, first), (2, second)),
      targetSize = 10,
      maxBufferedSize = 20,
      outputBatchesMetric = outputBatches,
      outputRowsMetric = outputRows)
    try {
      assert(iterator.hasNext)
      val output = iterator.next()
      assert(output._1 == 2)
      assert(output._2.numCols() == 0)
      assert(output._2.numRows() == 7)
      assert(outputBatches.value == 1)
      assert(outputRows.value == 7)
    } finally {
      iterator.close()
      first.close()
      second.close()
    }
  }

  test("routing and target sizing for 6000 partitions") {
    assert(GpuShuffleWriteCoalesceIterator.shouldCoalesce(
      enabled = true,
      useMultiThreadedShuffle = true,
      useGpuShuffle = false,
      useKudoGpuSlicing = false))
    assert(!GpuShuffleWriteCoalesceIterator.shouldCoalesce(
      enabled = false,
      useMultiThreadedShuffle = true,
      useGpuShuffle = false,
      useKudoGpuSlicing = false))
    assert(!GpuShuffleWriteCoalesceIterator.shouldCoalesce(
      enabled = true,
      useMultiThreadedShuffle = false,
      useGpuShuffle = false,
      useKudoGpuSlicing = false))
    assert(!GpuShuffleWriteCoalesceIterator.shouldCoalesce(
      enabled = true,
      useMultiThreadedShuffle = true,
      useGpuShuffle = true,
      useKudoGpuSlicing = false))
    assert(!GpuShuffleWriteCoalesceIterator.shouldCoalesce(
      enabled = true,
      useMultiThreadedShuffle = true,
      useGpuShuffle = false,
      useKudoGpuSlicing = true))

    val taskBudget = GpuShuffleWriteCoalesceIterator.perTaskBudget(
      totalBudget = 10L * 1024 * 1024 * 1024,
      parallelism = 64,
      numPartitions = 6000)
    assert(taskBudget == GpuShuffleWriteCoalesceIterator.MIN_TARGET_SIZE * 6000)
    assert(GpuShuffleWriteCoalesceIterator.targetSize(
      configuredTarget = 5L * 1024 * 1024,
      taskBudget = taskBudget,
      numPartitions = 6000) == GpuShuffleWriteCoalesceIterator.MIN_TARGET_SIZE)
  }

  test("write coalesce defaults are enabled and bounded") {
    val conf = new RapidsConf(Map.empty[String, String])
    assert(conf.isShuffleWriteCoalesceEnabled)
    assert(conf.shuffleWriteCoalesceMinPartitionSize == 5L * 1024 * 1024)
    assert(conf.shuffleWriteCoalesceTotalPartitionsSize == 10L * 1024 * 1024 * 1024)
  }

  test("close is idempotent and terminates iteration") {
    val input = makeBatch(Array(boxed(1)), Array("one"), Array(list(boxed(1))))
    withResource(input) { _ =>
      val iterator = new GpuShuffleWriteCoalesceIterator(
        Iterator((0, input)), targetSize = 1024, maxBufferedSize = 1024)
      assert(iterator.hasNext)
      iterator.close()
      iterator.close()
      assert(!iterator.hasNext)
      intercept[NoSuchElementException](iterator.next())
    }
  }
}
