/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

import java.io._
import java.nio.ByteBuffer

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

import ai.rapids.cudf.{ContiguousTable, DeviceMemoryBuffer, HostColumnVector, HostMemoryBuffer, JCudfSerialization, NvtxColor, NvtxRange}
import ai.rapids.cudf.JCudfSerialization.SerializedTableHeader
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import com.nvidia.spark.rapids.format.TableMeta

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.serializer.{DeserializationStream, SerializationStream, Serializer, SerializerInstance}
import org.apache.spark.sql.types.{DataType, NullType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector => SparkColumnVector}

class SerializedBatchIterator(dIn: DataInputStream) extends Iterator[(Int, ColumnarBatch)] {
  private[this] var nextHeader: Option[SerializedTableHeader] = None
  private[this] var toBeReturned: Option[ColumnarBatch] = None
  private[this] var streamClosed: Boolean = false

  // Don't install the callback if in a unit test
  Option(TaskContext.get()).foreach { tc =>
    onTaskCompletion(tc) {
      toBeReturned.foreach(_.close())
      toBeReturned = None
      dIn.close()
    }
  }

  def tryReadNextHeader(): Option[Long] = {
    if (streamClosed){
      None
    } else {
      if (nextHeader.isEmpty) {
        withResource(new NvtxRange("Read Header", NvtxColor.YELLOW)) { _ =>
          val header = new SerializedTableHeader(dIn)
          if (header.wasInitialized) {
            nextHeader = Some(header)
          } else {
            dIn.close()
            streamClosed = true
            nextHeader = None
          }
        }
      }
      nextHeader.map(_.getDataLen)
    }
  }

  def tryReadNext(): Option[ColumnarBatch] = {
    if (nextHeader.isEmpty) {
      None
    } else {
      withResource(new NvtxRange("Read Batch", NvtxColor.YELLOW)) { _ =>
        val header = nextHeader.get
        if (header.getNumColumns > 0) {
          // This buffer will later be concatenated into another host buffer before being
          // sent to the GPU, so no need to use pinned memory for these buffers.
          closeOnExcept(
            HostMemoryBuffer.allocate(header.getDataLen, false)) { hostBuffer =>
            JCudfSerialization.readTableIntoBuffer(dIn, header, hostBuffer)
            Some(SerializedTableColumn.from(header, hostBuffer))
          }
        } else {
          Some(SerializedTableColumn.from(header))
        }
      }
    }
  }

  override def hasNext: Boolean = {
    tryReadNextHeader()
    nextHeader.isDefined
  }

  override def next(): (Int, ColumnarBatch) = {
    if (toBeReturned.isEmpty) {
      tryReadNextHeader()
      toBeReturned = tryReadNext()
      if (nextHeader.isEmpty || toBeReturned.isEmpty) {
        throw new NoSuchElementException("Walked off of the end...")
      }
    }
    val ret = toBeReturned.get
    toBeReturned = None
    nextHeader = None
    (0, ret)
  }
}

/**
 * Serializer for serializing `ColumnarBatch`s for use during normal shuffle.
 *
 * The serialization write path takes the cudf `Table` that is described by the `ColumnarBatch`
 * and uses cudf APIs to serialize the data into a sequence of bytes on the host. The data is
 * returned to the Spark shuffle code where it is compressed by the CPU and written to disk.
 *
 * The serialization read path is notably different. The sequence of serialized bytes IS NOT
 * deserialized into a cudf `Table` but rather tracked in host memory by a `ColumnarBatch`
 * that contains a [[SerializedTableColumn]]. During query planning, each GPU columnar shuffle
 * exchange is followed by a [[GpuShuffleCoalesceExec]] that expects to receive only these
 * custom batches of [[SerializedTableColumn]]. [[GpuShuffleCoalesceExec]] coalesces the smaller
 * shuffle partitions into larger tables before placing them on the GPU for further processing.
 *
 * @note The RAPIDS shuffle does not use this code.
 */
class GpuColumnarBatchSerializer(dataSize: GpuMetric, serTime: GpuMetric = NoopMetric,
    deserTime: GpuMetric = NoopMetric, isSerializedTable: Boolean = false,
    sparkTypes: Array[DataType] = Array.empty) extends Serializer with Serializable {
  override def newInstance(): SerializerInstance =
    new GpuColumnarBatchSerializerInstance(dataSize, serTime, deserTime,
      isSerializedTable, sparkTypes)
  override def supportsRelocationOfSerializedObjects: Boolean = true
}

private class GpuColumnarBatchSerializerInstance(dataSize: GpuMetric, serTime: GpuMetric,
    deserTime: GpuMetric, isSerializedTable: Boolean, sparkTypes: Array[DataType]
) extends SerializerInstance {
  private lazy val tableSerializer = new SimpleTableSerializer(sparkTypes, deserTime)

  override def serializeStream(out: OutputStream): SerializationStream = new SerializationStream
      with Logging {
    private[this] val dOut = new DataOutputStream(new BufferedOutputStream(out))

    private def serializeCpuBatch(batch: ColumnarBatch): Unit = {
      val numRows = batch.numRows()
      val numCols = batch.numCols()
      if (numCols > 0) {
        withResource(new ArrayBuffer[AutoCloseable]()) { toClose =>
          var startRow = 0
          val toHostCol: SparkColumnVector => HostColumnVector = batch.column(0) match {
            case sliced: SlicedGpuColumnVector =>
              // We don't have control over ColumnarBatch to put in the slice, so we have
              // to do it for each column.  In this case we are using the first column.
              startRow = sliced.getStart
              col => col.asInstanceOf[SlicedGpuColumnVector].getBase
            case _: GpuColumnVector =>
              col => {
                val hCol = col.asInstanceOf[GpuColumnVector].copyToHost()
                toClose += hCol
                hCol.getBase
              }
            case _: RapidsHostColumnVector =>
              col => col.asInstanceOf[RapidsHostColumnVector].getBase
          }
          val cols = (0 until numCols).map(i => toHostCol(batch.column(i))).toArray
          dataSize += JCudfSerialization.getSerializedSizeInBytes(cols, startRow, numRows)
          withResource(new NvtxRange("Serialize Batch", NvtxColor.YELLOW)) { _ =>
            JCudfSerialization.writeToStream(cols, dOut, startRow, numRows)
          }
        }
      } else { // Rows only batch
        withResource(new NvtxRange("Serialize Row Only Batch", NvtxColor.YELLOW)) { _ =>
          JCudfSerialization.writeRowsToStream(dOut, numRows)
        }
      }
    }

    private def serializeGpuBatch(batch: ColumnarBatch): Unit = {
      if (batch.numCols() > 0) {
        batch.column(0) match {
          case packTable: GpuPackedTableColumn =>
            withResource(new NvtxRange("Serialize Table", NvtxColor.RED)) { _ =>
              dataSize += tableSerializer.writeToStream(packTable.getContiguousTable, dOut)
            }
          case o => throw new IllegalArgumentException(
            s"Table with '${o.getClass.getSimpleName}' columns is not supported")
        }
      } else {
        withResource(new NvtxRange("Serialize Rows Only Table", NvtxColor.RED)) { _ =>
          dataSize += tableSerializer.writeRowsOnlyToStream(batch.numRows(), dOut)
        }
      }
    }

    private lazy val serializeBatch: ColumnarBatch => Unit = if (isSerializedTable) {
      serializeGpuBatch
    } else {
      serializeCpuBatch
    }

    override def writeValue[T: ClassTag](value: T): SerializationStream = {
      serTime.ns(serializeBatch(value.asInstanceOf[ColumnarBatch]))
      this
    }

    override def writeKey[T: ClassTag](key: T): SerializationStream = {
      // The key is only needed on the map side when computing partition ids. It does not need to
      // be shuffled.
      assert(null == key || key.isInstanceOf[Int])
      this
    }

    override def writeAll[T: ClassTag](iter: Iterator[T]): SerializationStream = {
      // This method is never called by shuffle code.
      throw new UnsupportedOperationException
    }

    override def writeObject[T: ClassTag](t: T): SerializationStream = {
      // This method is never called by shuffle code.
      throw new UnsupportedOperationException
    }

    override def flush(): Unit = {
      dOut.flush()
    }

    override def close(): Unit = {
      dOut.close()
    }
  }


  override def deserializeStream(in: InputStream): DeserializationStream = {
    new DeserializationStream {
      private[this] val dIn: DataInputStream = new DataInputStream(new BufferedInputStream(in))

      override def asKeyValueIterator: Iterator[(Int, ColumnarBatch)] = {
        if (isSerializedTable) {
          new SerializedTableIterator(dIn, tableSerializer)
        } else {
          new SerializedBatchIterator(dIn)
        }
      }

      override def asIterator: Iterator[Any] = {
        // This method is never called by shuffle code.
        throw new UnsupportedOperationException
      }

      override def readKey[T]()(implicit classType: ClassTag[T]): T = {
        // We skipped serialization of the key in writeKey(), so just return a dummy value since
        // this is going to be discarded anyways.
        null.asInstanceOf[T]
      }

      override def readValue[T]()(implicit classType: ClassTag[T]): T = {
        // This method should never be called by shuffle code.
        throw new UnsupportedOperationException
      }

      override def readObject[T]()(implicit classType: ClassTag[T]): T = {
        // This method is never called by shuffle code.
        throw new UnsupportedOperationException
      }

      override def close(): Unit = {
        dIn.close()
      }
    }
  }

  // These methods are never called by shuffle code.
  override def serialize[T: ClassTag](t: T): ByteBuffer = throw new UnsupportedOperationException
  override def deserialize[T: ClassTag](bytes: ByteBuffer): T =
    throw new UnsupportedOperationException
  override def deserialize[T: ClassTag](bytes: ByteBuffer, loader: ClassLoader): T =
    throw new UnsupportedOperationException
}

private[rapids] class SimpleTableSerializer(sparkTypes: Array[DataType], deserTime: GpuMetric) {

  private val P_MAGIC_NUM: Int = 0x43554447
  private val P_VERSION: Int = 0
  private val headerLen = 8 // the size in bytes of two Ints for a header
  private val tmpBuf = new Array[Byte](1024 * 64) // 64k

  private def writeByteBufferToStream(bBuf: ByteBuffer, dOut: DataOutputStream): Unit = {
    // Write the buffer size first
    val bufLen = bBuf.capacity()
    dOut.writeLong(bufLen.toLong)
    if (bBuf.hasArray) {
      dOut.write(bBuf.array())
    } else { // Probably a direct buffer
      var leftLen = bufLen
      while (leftLen > 0) {
        val copyLen = Math.min(tmpBuf.length, leftLen)
        bBuf.get(tmpBuf, 0, copyLen)
        dOut.write(tmpBuf, 0, copyLen)
        leftLen -= copyLen
      }
    }
  }

  private def writeHostBufferToStream(hBuf: HostMemoryBuffer, dOut: DataOutputStream): Unit = {
    // Write the buffer size first
    val bufLen = hBuf.getLength
    dOut.writeLong(bufLen)
    var leftLen = bufLen
    var hOffset = 0L
    while (leftLen > 0L) {
      val copyLen = Math.min(tmpBuf.length, leftLen)
      hBuf.getBytes(tmpBuf, 0, hOffset, copyLen)
      dOut.write(tmpBuf, 0, copyLen.toInt)
      leftLen -= copyLen
      hOffset += copyLen
    }
  }

  private def writeProtocolHeader(dOut: DataOutputStream): Unit = {
    dOut.writeInt(P_MAGIC_NUM)
    dOut.writeInt(P_VERSION)
  }

  def writeRowsOnlyToStream(numRows: Int, dOut: DataOutputStream): Long = {
    // 1) header
    writeProtocolHeader(dOut)
    // 2) metadata fo an empty batch
    val degenBatch = new ColumnarBatch(Array.empty, numRows)
    val tableMetaBuf = MetaUtils.buildDegenerateTableMeta(degenBatch).getByteBuffer
    writeByteBufferToStream(tableMetaBuf, dOut)
    headerLen + tableMetaBuf.capacity()
  }

  def writeToStream(table: ContiguousTable, dOut: DataOutputStream): Long = {
    // 1) header
    writeProtocolHeader(dOut)
    // 2) table metadata,
    val tableMetaBuf = MetaUtils.buildTableMeta(0, table).getByteBuffer
    writeByteBufferToStream(tableMetaBuf, dOut)
    // 3) table data, it is already serializable by the upstream process.
    val dataDevBuf = table.getBuffer
    withResource(HostMemoryBuffer.allocate(dataDevBuf.getLength)) { hostBuf =>
      hostBuf.copyFromDeviceBuffer(dataDevBuf)
      writeHostBufferToStream(hostBuf, dOut)
    }
    headerLen + tableMetaBuf.capacity() + dataDevBuf.getLength
  }

  private def readProtocolHeader(dIn: DataInputStream): Unit = {
    val magicNum = dIn.readInt()
    if (magicNum != P_MAGIC_NUM) {
      throw new IllegalStateException(s"Expected magic number $P_MAGIC_NUM for " +
        s"table serializer, but got $magicNum")
    }
    val version = dIn.readInt()
    if (version != P_VERSION) {
      throw new IllegalStateException(s"Version mismatch: expected $P_VERSION for " +
        s"table serializer, but got $version")
    }
  }

  private def readByteBufferFromStream(dIn: DataInputStream): ByteBuffer = {
    val bufLen = dIn.readLong().toInt
    val bufArray = new Array[Byte](bufLen)
    var readLen = 0
    // A single call to read(bufArray) can not always read the expected length. So
    // we do it here ourselves.
    do {
      val ret = dIn.read(bufArray, readLen, bufLen - readLen)
      if (ret < 0) {
        throw new EOFException()
      }
      readLen += ret
    } while (readLen < bufLen)
    ByteBuffer.wrap(bufArray)
  }

  private def readHostBufferFromStream(dIn: DataInputStream): HostMemoryBuffer = {
    val bufLen = dIn.readLong()
    closeOnExcept(HostMemoryBuffer.allocate(bufLen)) { hostBuf =>
      var leftLen = bufLen
      var hOffset = 0L
      while (leftLen > 0) {
        val copyLen = Math.min(tmpBuf.length, leftLen)
        val readLen = dIn.read(tmpBuf, 0, copyLen.toInt)
        if (readLen < 0) {
          throw new EOFException()
        }
        hostBuf.setBytes(hOffset, tmpBuf, 0, readLen)
        hOffset += readLen
        leftLen -= readLen
      }
      hostBuf
    }
  }

  def readFromStream(dIn: DataInputStream): ColumnarBatch = {
    // 1) read and check header
    readProtocolHeader(dIn)
    // 2) read table metadata
    val tableMeta = TableMeta.getRootAsTableMeta(readByteBufferFromStream(dIn))
    // Acquiring the GPU regardless of whether the coming batch is empty or not,
    // because the downstream tasks expect the GPU batch producer to acquire the
    // semaphore and may generate GPU data from batches that are empty.
    GpuSemaphore.acquireIfNecessary(TaskContext.get())
    deserTime.ns {
      if (tableMeta.packedMetaAsByteBuffer() == null) {
        // no packed metadata, must be a table with zero columns
        new ColumnarBatch(Array.empty, tableMeta.rowCount().toInt)
      } else {
        // 3) read table data
        val data = withResource(new NvtxRange("Shuffle Buffering", NvtxColor.RED)) { _ =>
          withResource(readHostBufferFromStream(dIn)) { dataHostBuf =>
            closeOnExcept(DeviceMemoryBuffer.allocate(dataHostBuf.getLength)) { dataDevBuf =>
              dataDevBuf.copyFromHostBuffer(dataHostBuf)
              dataDevBuf
            }
          }
        }
        withResource(new NvtxRange("Shuffle Deserialization", NvtxColor.YELLOW)) { _ =>
          withResource(data) { _ =>
            val bufferMeta = tableMeta.bufferMeta()
            if (bufferMeta == null || bufferMeta.codecBufferDescrsLength == 0) {
              MetaUtils.getBatchFromMeta(data, tableMeta, sparkTypes)
            } else {
              // Compressed table is not supported by the write side, but ok to
              // put it here for the read side. Since compression will be supported later.
              GpuCompressedColumnVector.from(data, tableMeta)
            }
          }
        }
      }
    }
  }
}

private[rapids] class SerializedTableIterator(
    dIn: DataInputStream,
    tableSerializer: SimpleTableSerializer) extends Iterator[(Int, ColumnarBatch)] {

  private var closed = false
  private var onDeck: Option[SpillableColumnarBatch] = None
  Option(TaskContext.get()).foreach { tc =>
    onTaskCompletion(tc) {
      onDeck.foreach(_.close())
      onDeck = None
      if (!closed) {
        dIn.close()
      }
    }
  }

  override def hasNext: Boolean = {
    if (onDeck.isEmpty) {
      tryReadNextBatch()
    }
    onDeck.isDefined
  }

  override def next(): (Int, ColumnarBatch) = {
    if (!hasNext) {
      throw new NoSuchElementException()
    }
    val ret = withResource(onDeck) { _ =>
      onDeck.get.getColumnarBatch()
    }
    onDeck = None
    (0, ret)
  }

  private def tryReadNextBatch(): Unit = {
    if (closed) {
      return
    }
    try {
      onDeck = Some(SpillableColumnarBatch(tableSerializer.readFromStream(dIn),
        SpillPriorities.ACTIVE_ON_DECK_PRIORITY))
    } catch {
      case _: EOFException => // we reach the end
        dIn.close()
        closed = true
        onDeck.foreach(_.close())
        onDeck = None
    }
  }
}

/**
 * A special `ColumnVector` that describes a serialized table read from shuffle.
 * This appears in a `ColumnarBatch` to pass serialized tables to [[GpuShuffleCoalesceExec]]
 * which should always appear in the query plan immediately after a shuffle.
 */
class SerializedTableColumn(
    val header: SerializedTableHeader,
    val hostBuffer: HostMemoryBuffer) extends GpuColumnVectorBase(NullType) {
  override def close(): Unit = {
    if (hostBuffer != null) {
      hostBuffer.close()
    }
  }

  override def hasNull: Boolean = throw new IllegalStateException("should not be called")

  override def numNulls(): Int = throw new IllegalStateException("should not be called")
}

object SerializedTableColumn {
  /**
   * Build a `ColumnarBatch` consisting of a single [[SerializedTableColumn]] describing
   * the specified serialized table.
   *
   * @param header header for the serialized table
   * @param hostBuffer host buffer containing the table data
   * @return columnar batch to be passed to [[GpuShuffleCoalesceExec]]
   */
  def from(
      header: SerializedTableHeader,
      hostBuffer: HostMemoryBuffer = null): ColumnarBatch = {
    val column = new SerializedTableColumn(header, hostBuffer)
    new ColumnarBatch(Array(column), header.getNumRows)
  }

  def getMemoryUsed(batch: ColumnarBatch): Long = {
    var sum: Long = 0
    if (batch.numCols == 1) {
      val cv = batch.column(0)
      cv match {
        case serializedTableColumn: SerializedTableColumn
            if serializedTableColumn.hostBuffer != null =>
          sum += serializedTableColumn.hostBuffer.getLength
        case _ =>
      }
    }
    sum
  }
}
