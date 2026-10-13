/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.
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

import java.util.Optional

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{DType, HostColumnVector, HostColumnVectorCore, HostMemoryBuffer, Table}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._

import org.apache.spark.sql.types.{ArrayType, DataType, DataTypes, MapType, StructType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

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
  def minGpuMemory(dataType:DataType, nullable: Boolean, rowCount: Long,
      includeOffset: Boolean = true): Long = {
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
        if (includeOffset) calculateOffsetBufferSize(rowCount) else 0L
      case dt: StructType =>
        dt.fields.map { f =>
          minGpuMemory(f.dataType, f.nullable, rowCount, includeOffset)
        }.sum
      case dt if GpuColumnVector.isVariantType(dt) =>
        if (includeOffset) {
          calculateOffsetBufferSize(rowCount) * 2
        } else {
          0L
        }
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
      case dt if GpuColumnVector.isVariantType(dt) =>
        // Reuse Spark's 2-KiB Variant defaultSize estimate and account separately for both
        // physical cuDF offset buffers.
        calculateOffsetBufferSize(rowCount) * 2 + dt.defaultSize * rowCount
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
    case variantType if GpuColumnVector.isVariantType(variantType) => false
    case DataTypes.StringType | DataTypes.BinaryType => false
    case _: ArrayType  => false
    case _: StructType  => false
    case _: MapType  => false
    case _ => true
  }

  /**
   * Concatenate CPU-sliced shuffle batches and close the inputs.
   *
   * The output remains a batch of [[SlicedGpuColumnVector]] so it can be consumed by the
   * existing CPU and Kudo CPU shuffle serializers without another host-to-device copy.
   */
  def concatShuffleBatchesAndClose(batches: Seq[ColumnarBatch]): ColumnarBatch = {
    require(batches.nonEmpty, "At least one shuffle batch is required")
    val nonEmptyBatches = batches.filter(_.numRows() > 0)
    val emptyBatches = batches.filter(_.numRows() == 0)
    if (nonEmptyBatches.isEmpty) {
      closeOnExcept(emptyBatches.head) { batch =>
        emptyBatches.tail.safeClose()
        batch
      }
    } else if (nonEmptyBatches.length == 1) {
      closeOnExcept(nonEmptyBatches.head) { batch =>
        emptyBatches.safeClose()
        batch
      }
    } else {
      withResource(nonEmptyBatches) { _ =>
        emptyBatches.safeClose()
        val numCols = nonEmptyBatches.head.numCols()
        require(nonEmptyBatches.forall(_.numCols() == numCols),
          "All shuffle batches must have the same number of columns")
        if (numCols == 0) {
          val numRows = nonEmptyBatches.iterator.map(_.numRows().toLong).sum
          require(numRows <= Int.MaxValue,
            s"Cannot concatenate $numRows rows into one rows-only shuffle batch")
          new ColumnarBatch(Array.empty[ColumnVector], numRows.toInt)
        } else {
          concatShuffleBatches(nonEmptyBatches)
        }
      }
    }
  }

  private case class HostColumnSlice(column: HostColumnVectorCore, start: Int, end: Int) {
    require(start >= 0 && end >= start,
      s"Invalid host column slice [$start, $end)")
  }

  private def concatShuffleBatches(batches: Seq[ColumnarBatch]): ColumnarBatch = {
    val numRows = batches.iterator.map(_.numRows().toLong).sum
    require(numRows <= Int.MaxValue,
      s"Cannot concatenate $numRows rows into one shuffle batch")
    val rowCount = numRows.toInt
    val numCols = batches.head.numCols()
    val columnsByIndex = (0 until numCols).map { columnIndex =>
      batches.map { batch =>
        batch.column(columnIndex) match {
          case column: SlicedGpuColumnVector =>
            HostColumnSlice(column.getBase, column.getStart, column.getEnd)
          case other =>
            throw new IllegalArgumentException(
              s"Unsupported shuffle column for host coalescing: ${other.getClass.getName}")
        }
      }
    }
    val outputSize = columnsByIndex.iterator.map(estimateConcatSize).sum
    val outputColumns = withResource(HostMemoryBuffer.allocate(math.max(outputSize, 1L))) {
      outputBuffer =>
        var outputOffset = 0L
        (0 until numCols).safeMap { columnIndex =>
          val sparkType = batches.head.column(columnIndex).dataType()
          val (hostColumnCore, columnSize) =
            concatHostColumns(columnsByIndex(columnIndex), outputBuffer, outputOffset,
              topLevel = true)
          outputOffset += columnSize
          val hostColumn = hostColumnCore.asInstanceOf[HostColumnVector]
          closeOnExcept(hostColumn) { _ =>
            withResource(new RapidsHostColumnVector(sparkType, hostColumn)) { rapidsColumn =>
              new SlicedGpuColumnVector(rapidsColumn, 0, rowCount)
            }
          }
        }
    }
    closeOnExcept(outputColumns) { columns =>
      new ColumnarBatch(columns.toArray, rowCount)
    }
  }

  private def estimateConcatSize(columns: Seq[HostColumnSlice]): Long = {
    require(columns.nonEmpty, "At least one host column slice is required")
    val columnType = columns.head.column.getType
    require(columns.forall(_.column.getType == columnType),
      s"All host columns must have type $columnType")
    val rowCount = columns.iterator.map(c => c.end.toLong - c.start).sum
    require(rowCount <= Int.MaxValue, s"Host column row count $rowCount exceeds Int.MaxValue")
    var size = 0L
    if (columns.exists(_.column.hasValidityVector)) {
      size += calculateValidityBufferSize(rowCount)
    }
    if (columnType.hasOffsets) {
      size += calculateOffsetBufferSize(rowCount)
    }
    if (columnType == DType.STRING) {
      size += columns.iterator.map(stringDataLength).sum
    } else if (!columnType.isNestedType) {
      size += columns.iterator.map { slice =>
        if (slice.column.getData == null) {
          0L
        } else {
          (slice.end.toLong - slice.start) * columnType.getSizeInBytes
        }
      }.sum
    }
    if (columnType.isNestedType) {
      val numChildren = columns.head.column.getNumChildren
      require(columns.forall(_.column.getNumChildren == numChildren),
        "All nested host columns must have the same number of children")
      (0 until numChildren).foreach { childIndex =>
        size += estimateConcatSize(childSlices(columns, childIndex))
      }
    }
    size
  }

  private def stringDataLength(slice: HostColumnSlice): Long = {
    if (slice.column.getData == null || slice.start == slice.end) {
      0L
    } else {
      slice.column.getEndListOffset(slice.end - 1) -
        slice.column.getStartListOffset(slice.start)
    }
  }

  private def childSlices(
      columns: Seq[HostColumnSlice],
      childIndex: Int): Seq[HostColumnSlice] = {
    columns.map { slice =>
      val child = slice.column.getChildColumnView(childIndex)
      if (slice.column.getType == DType.LIST) {
        val childStart = Math.toIntExact(slice.column.getStartListOffset(slice.start))
        val childEnd = if (slice.start == slice.end) {
          childStart
        } else {
          Math.toIntExact(slice.column.getEndListOffset(slice.end - 1))
        }
        HostColumnSlice(child, childStart, childEnd)
      } else {
        HostColumnSlice(child, slice.start, slice.end)
      }
    }
  }

  private def concatHostColumns(
      columns: Seq[HostColumnSlice],
      outputBuffer: HostMemoryBuffer,
      outputOffset: Long,
      topLevel: Boolean): (HostColumnVectorCore, Long) = {
    val columnType = columns.head.column.getType
    val rowCount = columns.iterator.map(c => c.end.toLong - c.start).sum
    require(rowCount <= Int.MaxValue, s"Host column row count $rowCount exceeds Int.MaxValue")
    val numRows = rowCount.toInt
    var currentOffset = outputOffset

    val (validityBuffer, nullCount) = if (columns.exists(_.column.hasValidityVector)) {
      val validitySize = calculateValidityBufferSize(rowCount)
      closeOnExcept(outputBuffer.slice(currentOffset, validitySize)) { destination =>
        currentOffset += validitySize
        destination.setMemory(0, validitySize, 0xff.toByte)
        var destinationRow = 0
        var nulls = 0L
        columns.foreach { slice =>
          var sourceRow = slice.start
          while (sourceRow < slice.end) {
            val sourceValidity = slice.column.getValidity
            if (sourceValidity != null && isNullAt(sourceValidity, sourceRow)) {
              setNullAt(destination, destinationRow)
              nulls += 1
            }
            sourceRow += 1
            destinationRow += 1
          }
        }
        (destination, nulls)
      }
    } else {
      (null, 0L)
    }

    val offsetsBuffer = closeOnExcept(validityBuffer) { _ =>
      if (columnType.hasOffsets) {
        val offsetsSize = calculateOffsetBufferSize(rowCount)
        closeOnExcept(outputBuffer.slice(currentOffset, offsetsSize)) { destination =>
          currentOffset += offsetsSize
          var destinationPosition = 0L
          var accumulatedOffset = 0L
          destination.setInt(destinationPosition, 0)
          destinationPosition += OFFSET_BYTES
          columns.foreach { slice =>
            val sourceOffsets = slice.column.getOffsets
            val firstSourceOffset = sourceOffsets.getInt(slice.start.toLong * OFFSET_BYTES)
            var sourceRow = slice.start + 1
            while (sourceRow <= slice.end) {
              val sourceOffset = sourceOffsets.getInt(sourceRow.toLong * OFFSET_BYTES)
              val outputValue = accumulatedOffset + sourceOffset - firstSourceOffset
              require(outputValue <= Int.MaxValue,
                s"Concatenated host column offset $outputValue exceeds Int.MaxValue")
              destination.setInt(destinationPosition, outputValue.toInt)
              destinationPosition += OFFSET_BYTES
              sourceRow += 1
            }
            accumulatedOffset =
              destination.getInt(destinationPosition - OFFSET_BYTES).toLong
          }
          destination
        }
      } else {
        null
      }
    }

    val dataBuffer = closeOnExcept(Seq(validityBuffer, offsetsBuffer)) { _ =>
      val sourceBuffers = if (columnType == DType.STRING) {
        columns.flatMap { slice =>
          val length = stringDataLength(slice)
          if (length > 0) {
            Some((slice.column.getData,
              slice.column.getStartListOffset(slice.start), length))
          } else {
            None
          }
        }
      } else if (!columnType.isNestedType) {
        columns.flatMap { slice =>
          Option(slice.column.getData).map { source =>
            val elementSize = columnType.getSizeInBytes.toLong
            (source, slice.start * elementSize,
              (slice.end.toLong - slice.start) * elementSize)
          }
        }
      } else {
        Seq.empty
      }
      val dataSize = sourceBuffers.iterator.map(_._3).sum
      if (dataSize > 0) {
        closeOnExcept(outputBuffer.slice(currentOffset, dataSize)) { destination =>
          currentOffset += dataSize
          var destinationOffset = 0L
          sourceBuffers.foreach { case (source, sourceOffset, length) =>
            destination.copyFromHostBuffer(destinationOffset, source, sourceOffset, length)
            destinationOffset += length
          }
          destination
        }
      } else {
        null
      }
    }

    val childColumns = closeOnExcept(Seq(validityBuffer, offsetsBuffer, dataBuffer)) { _ =>
      if (columnType.isNestedType) {
        val numChildren = columns.head.column.getNumChildren
        (0 until numChildren).safeMap { childIndex =>
          val (child, childSize) =
            concatHostColumns(childSlices(columns, childIndex), outputBuffer, currentOffset,
              topLevel = false)
          currentOffset += childSize
          child
        }
      } else {
        Seq.empty[HostColumnVectorCore]
      }
    }

    val outputColumn = closeOnExcept(childColumns) { _ =>
      val optionalNullCount = Optional.of(java.lang.Long.valueOf(nullCount))
      if (topLevel) {
        new HostColumnVector(columnType, numRows, optionalNullCount,
          dataBuffer, validityBuffer, offsetsBuffer, childColumns.asJava)
      } else {
        new HostColumnVectorCore(columnType, numRows, optionalNullCount,
          dataBuffer, validityBuffer, offsetsBuffer, childColumns.asJava)
      }
    }
    (outputColumn, currentOffset - outputOffset)
  }

  private def setNullAt(validityBuffer: HostMemoryBuffer, rowId: Int): Unit = {
    val byteOffset = rowId >> 3
    val current = validityBuffer.getByte(byteOffset)
    val bitMask = ~(1 << (rowId & 0x7))
    validityBuffer.setByte(byteOffset, (current & bitMask).toByte)
  }

  private def isNullAt(validityBuffer: HostMemoryBuffer, rowId: Int): Boolean = {
    val current = validityBuffer.getByte(rowId >> 3)
    (current & (1 << (rowId & 0x7))) == 0
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
    val numBatches = spillBatches.length
    val retBatch = if (numBatches >= 2 && spillBatches.forall(isRowsOnly)) {
      // cuDF cannot build a Table without columns, and rows-only batches concatenate by row count.
      withResource(spillBatches) { _ =>
        val numRows = spillBatches.iterator.map(_.numRows().toLong).sum
        require(numRows <= Int.MaxValue, s"Cannot concatenate $numRows rows into one batch")
        new JustRowsColumnarBatch(numRows.toInt)
      }
    } else if (numBatches >= 2) {
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
    } else if (numBatches == 1) {
      // only one batch
      spillBatches.head
    } else null

    Option(retBatch)
  }

  // A compressed batch reports null types but always has a column.
  private def isRowsOnly(batch: SpillableColumnarBatch): Boolean = {
    val types = batch.dataTypes
    types != null && types.isEmpty
  }
}
