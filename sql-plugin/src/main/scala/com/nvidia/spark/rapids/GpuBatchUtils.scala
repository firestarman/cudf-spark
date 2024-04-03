/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

import scala.collection.JavaConverters.seqAsJavaListConverter
import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{DType, HostColumnVector, HostColumnVectorCore, HostMemoryBuffer, Table}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits.{AutoCloseableProducingSeq, AutoCloseableSeq}

import org.apache.spark.sql.types.{ArrayType, DataType, DataTypes, MapType, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Utility class with methods for calculating various metrics about GPU memory usage
 * prior to allocation, along with some operations with batches.
 */
object GpuBatchUtils {

  /** Validity buffers are 64 byte aligned */
  val VALIDITY_BUFFER_BOUNDARY_BYTES = 64

  /** Validity buffers are 64 byte aligned and each byte represents 8 rows */
  val VALIDITY_BUFFER_BOUNDARY_ROWS = VALIDITY_BUFFER_BOUNDARY_BYTES * 8

  /** Number of bytes per offset (32 bit) */
  val OFFSET_BYTES = 4

  /** Estimate the number of rows required to meet a batch size limit */
  def estimateRowCount(
      desiredBatchSizeBytes: Long,
      currentBatchSize: Long,
      currentBatchRowCount: Long): Int = {
    assert(currentBatchRowCount > 0, "batch must contain at least one row")
    val targetRowCount: Long = if (currentBatchSize > desiredBatchSizeBytes) {
      currentBatchRowCount
    } else if (currentBatchSize == 0) {
      //  batch size can be 0 when doing a count() operation and the actual data isn't needed
      currentBatchRowCount
    } else {
      ((desiredBatchSizeBytes / currentBatchSize.floatValue()) * currentBatchRowCount).toLong
    }
    targetRowCount.min(Integer.MAX_VALUE).toInt
  }

  /** Estimate the amount of GPU memory a batch of rows will occupy once converted */
  def estimateGpuMemory(schema: StructType, rowCount: Long): Long = {
    schema.fields.indices.map(estimateGpuMemory(schema, _, rowCount)).sum
  }

  /** Estimate the amount of GPU memory a batch of rows will occupy once converted */
  def estimateGpuMemory(schema: StructType, columnIndex: Int, rowCount: Long): Long = {
    val field = schema.fields(columnIndex)
    estimateGpuMemory(field.dataType, field.nullable, rowCount)
  }

  /**
   * Get the minimum size a column could be that matches these conditions.
   */
  def minGpuMemory(dataType:DataType, nullable: Boolean, rowCount: Long): Long = {
    val validityBufferSize = if (nullable) {
      calculateValidityBufferSize(rowCount)
    } else {
      0
    }

    val dataSize = dataType match {
      case DataTypes.BinaryType | DataTypes.StringType | _: MapType | _: ArrayType=>
        // For nested types (like list or string) the smallest possible size is when
        // each row is empty (length 0). In that case there is no data, just offsets
        // and all of the offsets are 0.
        calculateOffsetBufferSize(rowCount)
      case dt: StructType =>
        dt.fields.map { f =>
          minGpuMemory(f.dataType, f.nullable, rowCount)
        }.sum
      case dt =>
        dt.defaultSize * rowCount
    }
    dataSize + validityBufferSize
  }

  def estimateGpuMemory(dataType: DataType, nullable: Boolean, rowCount: Long): Long = {
    val validityBufferSize = if (nullable) {
      calculateValidityBufferSize(rowCount)
    } else {
      0
    }
    val dataSize = dataType match {
      case dt@DataTypes.BinaryType =>
        val offsetBufferSize = calculateOffsetBufferSize(rowCount)
        val dataSize = dt.defaultSize * rowCount
        dataSize + offsetBufferSize
      case dt@DataTypes.StringType =>
        val offsetBufferSize = calculateOffsetBufferSize(rowCount)
        val dataSize = dt.defaultSize * rowCount
        dataSize + offsetBufferSize
      case dt: MapType =>
        // The Spark default map size assumes one entry for good or bad
        calculateOffsetBufferSize(rowCount) +
            estimateGpuMemory(dt.keyType, false, rowCount) +
            estimateGpuMemory(dt.valueType, dt.valueContainsNull, rowCount)
      case dt: ArrayType =>
        // The Spark default array size assumes one entry for good or bad
        calculateOffsetBufferSize(rowCount) +
            estimateGpuMemory(dt.elementType, dt.containsNull, rowCount)
      case dt: StructType =>
        dt.fields.map { f =>
          estimateGpuMemory(f.dataType, f.nullable, rowCount)
        }.sum
      case dt =>
        dt.defaultSize * rowCount
    }
    dataSize + validityBufferSize
  }

  def calculateValidityBufferSize(rows: Long): Long = {
    roundToBoundary((rows + 7)/8, 64)
  }

  def calculateOffsetBufferSize(rows: Long): Long = {
    (rows+1) * 4 // 32 bit offsets
  }

  /**
   * Generate indices which evenly splitting input batch
   *
   * @param rows      number of rows of input batch
   * @param numSplits desired number of splits
   * @return splitting indices
   */
  def generateSplitIndices(rows: Long, numSplits: Int): Array[Int] = {
    require(rows > 0, s"invalid input rows $rows")
    require(numSplits > 0, s"invalid numSplits $numSplits")
    val baseIncrement = (rows / numSplits).toInt
    var extraIncrements = (rows % numSplits).toInt
    val indicesBuf = ArrayBuffer[Int]()
    (1 until numSplits).foldLeft(0) { case (last, _) =>
      val current = if (extraIncrements > 0) {
        extraIncrements -= 1
        last + baseIncrement + 1
      } else {
        last + baseIncrement
      }
      indicesBuf += current
      current
    }
    indicesBuf.toArray
  }

  def isVariableWidth(dt: DataType): Boolean = !isFixedWidth(dt)

  def isFixedWidth(dt: DataType): Boolean = dt match {
    case DataTypes.StringType | DataTypes.BinaryType => false
    case _: ArrayType  => false
    case _: StructType  => false
    case _: MapType  => false
    case _ => true
  }

  private def roundToBoundary(bytes: Long, boundary: Int): Long = {
    val remainder = bytes % boundary
    if (remainder > 0) {
      bytes + boundary - remainder
    } else {
      bytes
    }
  }

  /**
   * Concatenate the input batches into a single one.
   * The caller is responsible for closing the returned batch.
   *
   * @param spillBatches the batches to be concatenated, will be closed after the call
   *                     returns.
   * @return the concatenated SpillableColumnarBatch or None if the input is empty.
   */
  def concatSpillBatchesAndClose(
      spillBatches: Seq[SpillableColumnarBatch]): Option[SpillableColumnarBatch] = {
    val retBatch = if (spillBatches.length >= 2) {
      // two or more batches, concatenate them
      val (concatTable, types) = RmmRapidsRetryIterator.withRetryNoSplit(spillBatches) { _ =>
        withResource(spillBatches.safeMap(_.getColumnarBatch())) { batches =>
          val batchTypes = GpuColumnVector.extractTypes(batches.head)
          withResource(batches.safeMap(GpuColumnVector.from)) { tables =>
            (Table.concatenate(tables: _*), batchTypes)
          }
        }
      }
      // Make the concatenated table spillable.
      withResource(concatTable) { _ =>
        SpillableColumnarBatch(GpuColumnVector.from(concatTable, types),
          SpillPriorities.ACTIVE_BATCHING_PRIORITY)
      }
    } else if (spillBatches.length == 1) {
      // only one batch
      spillBatches.head
    } else null

    Option(retBatch)
  }

  /**
   * Only support batches sliced on CPU for shuffle, meaning the internal
   * columns are instances of SlicedGpuColumnVector.
   */
  def concatShuffleBatchesAndClose(batches: Seq[ColumnarBatch],
      totalSize: Option[Long] = None): ColumnarBatch = {
    val (nonEmptyCBs, emptyCbs) = batches.partition(_.numRows() > 0)
    if (nonEmptyCBs.nonEmpty) {
      emptyCbs.safeClose()
      if (nonEmptyCBs.length == 1) {
        nonEmptyCBs.head
      } else { // more than one batch
        withResource(nonEmptyCBs) { _ =>
          concatShuffleBatches(nonEmptyCBs, totalSize)
        }
      }
    } else {
      assert(emptyCbs.nonEmpty)
      emptyCbs.tail.safeClose()
      emptyCbs.head
    }
  }

  private def concatShuffleBatches(batches: Seq[ColumnarBatch],
      totalSize: Option[Long]): ColumnarBatch = {
    val numCols = batches.head.numCols()
    // all batches should have the same columns number
    batches.tail.foreach(b => assert(numCols == b.numCols()))
    val sizeSum = totalSize.getOrElse(
      batches.map(SlicedGpuColumnVector.getTotalHostMemoryUsed).sum
    ) + (numCols << 6) // For the validity padding, numCols * 64 for the worst case
    val concatNumRows = batches.map(_.numRows()).sum
    // Allocate a single buffer for the merged batch.
    val concatHostCols = withResource(HostMemoryBuffer.allocate(sizeSum)) { allBuf =>
      var outOff = 0L
      (0 until numCols).safeMap { idx =>
        val cols = batches.map(_.column(idx).asInstanceOf[SlicedGpuColumnVector])
        // Concatenate the input sliced columns
        val (concatCol, concatLen) = concatSlicedColumns(cols, concatNumRows, allBuf, outOff)
        withResource(concatCol) { _ =>
          outOff += concatLen
          // The downstream shuffle writer expects SlicedGpuColumnVectors
          new SlicedGpuColumnVector(concatCol, 0, concatNumRows)
        }
      }
    }
    new ColumnarBatch(concatHostCols.toArray, concatNumRows)
  }

  private def concatSlicedColumns(cols: Seq[SlicedGpuColumnVector], totalRowsNum: Int,
      outBuf: HostMemoryBuffer, outOffset: Long): (RapidsHostColumnVector, Long) = {
    // All should have the same type
    val colSparkType = cols.head.dataType()
    assert(cols.tail.forall(_.dataType() == colSparkType),
      s"All the column types should be $colSparkType, but got (" +
        s"${cols.map(_.dataType()).mkString("; ")})")
    val (cudfHostColumn, colLen) = concatSlicedColumns(
      cols.map(c => (c.getBase, c.getStart, c.getEnd)), outBuf, outOffset, Some(totalRowsNum))
    (new RapidsHostColumnVector(colSparkType, cudfHostColumn), colLen)
  }

  /** (TODO Move concatenating HostColumnVectors to Rapids JNI) */
  private def concatSlicedColumns(cols: Seq[(HostColumnVectorCore, Int, Int)],
      outBuf: HostMemoryBuffer, outOffset: Long,
      totalRowsNum: Option[Int] = None): (HostColumnVector, Long) = {
    val colCudfType = cols.head._1.getType
    val concatRowsNum = totalRowsNum.getOrElse(cols.map(c => c._3 - c._2).sum)
    var curGlobalPos = outOffset
    // 1) Validity buffer. It is required if any has a validity buffer.
    val (concatValidityBuf, nullCount) = if (cols.exists(_._1.hasValidityVector)) {
      val concatValidityLen = RapidsHostColumnVector.getValidityBufferSize(concatRowsNum)
      closeOnExcept(outBuf.slice(curGlobalPos, concatValidityLen)) { destBuf =>
        curGlobalPos += concatValidityLen
        // Set all the bits to "1" by default.
        destBuf.setMemory(0, concatValidityLen, 0xff.toByte)
        var accNullCnt = 0L
        var destRowsNum = 0
        cols.foreach { case (c, sStart, sEnd) =>
          val validityBuf = c.getValidity
          if (validityBuf != null) {
            // Has nulls, set it one by one
            var rowId = sStart
            while (rowId < sEnd) {
              if (isNullAt(validityBuf, rowId)) {
                setNullAt(destBuf, destRowsNum)
                accNullCnt += 1
              }
              rowId += 1
              destRowsNum += 1
            }
          } else { // no nulls, just update the dest rows number
            destRowsNum += (sEnd - sStart)
          }
        }
        assert(destRowsNum == concatRowsNum)
        (destBuf, accNullCnt)
      }
    } else {
      (null, 0L)
    }

    // 2) Offset buffer. All should has the same type, so only need to check the first one
    val concatOffsetBuf = closeOnExcept(concatValidityBuf) { _ =>
      if (colCudfType.hasOffsets) {
        val concatOffsetLen = RapidsHostColumnVector.getOffsetBufferSize(concatRowsNum)
        closeOnExcept(outBuf.slice(curGlobalPos, concatOffsetLen)) { destBuf =>
          curGlobalPos += concatOffsetLen
          val offBufStep = RapidsHostColumnVector.OFFSET_STEP
          var destPos = 0L
          var accOffsetValue = 0
          // Compute offsets. Suppose all should have offset buffers.
          // The first one is always 0
          destBuf.setInt(destPos, accOffsetValue)
          destPos += offBufStep
          cols.foreach { case (c, sStart, sEnd) =>
            val offBuf = c.getOffsets
            val offBufEnd = sEnd << RapidsHostColumnVector.OFFSET_SHIFT_STEP
            var curOffBufPos = sStart << RapidsHostColumnVector.OFFSET_SHIFT_STEP
            val offsetDiff = accOffsetValue - offBuf.getInt(curOffBufPos)
            curOffBufPos += offBufStep
            while (curOffBufPos <= offBufEnd) {
              destBuf.setInt(destPos, offBuf.getInt(curOffBufPos) + offsetDiff)
              destPos += offBufStep
              curOffBufPos += offBufStep
            }
            // The last entry is offset value for the next buffer
            accOffsetValue = destBuf.getInt(destPos - offBufStep)
          }
          assert(destPos == concatOffsetLen)
          destBuf
        }
      } else {
        null
      }
    }

    // 3) data buffer
    val nonEmptyDataCols = cols.filter(_._1.getData != null)
    val concatDataBuf = closeOnExcept(Seq(concatValidityBuf, concatOffsetBuf)) { _ =>
      if (nonEmptyDataCols.nonEmpty) {
        // String or primitive type
        type DataBufFunc = ((HostColumnVectorCore, Int, Int)) => (HostMemoryBuffer, Long, Long)
        val getSlicedDataBuf: DataBufFunc = if (DType.STRING.equals(colCudfType)) {
          // String type has both data and offset
          c => { // c is (column, start, end)
            val start = c._1.getStartListOffset(c._2)
            (c._1.getData, start, c._1.getEndListOffset(c._3 - 1) - start)
          }
        } else { // non-nested type
          c => { // c is (column, start, end)
            val typeSize = colCudfType.getSizeInBytes.toLong
            assert(typeSize > 0, s"Non-nested type is expected, but got $colCudfType")
            (c._1.getData, c._2 * typeSize, (c._3 - c._2) * typeSize)
          }
        }
        val nonEmptyDataBufs = nonEmptyDataCols.map(getSlicedDataBuf)
        val concatDataLen = nonEmptyDataBufs.map(_._3).sum
        closeOnExcept(outBuf.slice(curGlobalPos, concatDataLen)) { destBuf =>
          curGlobalPos += concatDataLen
          var destPos = 0L
          // Just append the data buffer one by one
          nonEmptyDataBufs.foreach { case (srcBuf, srcStart, srcLen) =>
            destBuf.copyFromHostBuffer(destPos, srcBuf, srcStart, srcLen)
            destPos += srcLen
          }
          destBuf
        }
      } else {
        null
      }
    }

    // 4) children
    val concatNestedHcv = closeOnExcept(Seq(concatValidityBuf, concatOffsetBuf, concatDataBuf)) {
      _ =>
        if (colCudfType.isNestedType) {
          System.err.println(s"==> Got nested type ($colCudfType) columns to be concatenated.")
          // All should have the same children number
          val childrenNum = cols.head._1.getNumChildren
          assert(childrenNum > 0, "Non empty children is expected")
          (0 until childrenNum).safeMap { idx =>
            val sChildren = cols.map { case (c, start, end) =>
              val childView = c.getChildColumnView(idx)
              if (childView.getType.hasOffsets) {
                (childView, c.getStartListOffset(start).toInt, c.getEndListOffset(end -1).toInt)
              } else {
                (childView, start, end)
              }
            }
            val (childCol, colLen) = concatSlicedColumns(sChildren, outBuf, curGlobalPos)
            curGlobalPos += colLen
            childCol
          }.asInstanceOf[Seq[HostColumnVectorCore]].asJava
        } else {
          new java.util.ArrayList[HostColumnVectorCore]()
        }
    }

    val cudfHostColumn = new HostColumnVector(
      colCudfType, concatRowsNum, java.util.Optional.of(nullCount),
      concatDataBuf, concatValidityBuf, concatOffsetBuf, concatNestedHcv)
    (cudfHostColumn, curGlobalPos - outOffset)
  }

  private def setNullAt(validBuf: HostMemoryBuffer, rowId: Int): Unit = {
    val bucket = rowId >> 3 // = (rowId / 8)
    val curByte = validBuf.getByte(bucket)
    val bitmask = (~(1 << (rowId & 0x7).toByte))
    validBuf.setByte(bucket, (curByte & bitmask).toByte)
  }

  private def isNullAt(validBuf: HostMemoryBuffer, rowId: Int): Boolean = {
    val b = validBuf.getByte(rowId >> 3) // = (rowI / 8)
    val ret = b & (1 << (rowId & 0x7).toByte)
    ret == 0
  }
}
