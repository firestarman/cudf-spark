/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

package org.apache.spark.sql.hive.rapids

import java.nio.charset.Charset
import java.util.Locale

import ai.rapids.cudf.{CompressionType, CSVWriterOptions, DType, ParquetWriterOptions, QuoteStyle, Scalar, Table, TableWriter => CudfTableWriter}
import com.google.common.base.Charsets
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.jni.CastStrings
import org.apache.hadoop.mapreduce.{Job, TaskAttemptContext}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.parquet.ParquetOptions
import org.apache.spark.sql.hive.rapids.GpuHiveTextFileUtils._
import org.apache.spark.sql.hive.rapids.shims.GpuInsertIntoHiveTableMeta
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{DataType, StringType, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuHiveFileFormat extends Logging {

  def tagGpuSupport(meta: GpuInsertIntoHiveTableMeta): Option[ColumnarFileFormat] = {
    val insertCmd = meta.wrapped
    // Bucketing write
    if (insertCmd.table.bucketSpec.isDefined) {
      meta.willNotWorkOnGpu("bucketed tables are not supported yet")
    }

    // Infer the file format from the serde string, similar as what Spark does in
    // RelationConversions for Hive.
    val serde = insertCmd.table.storage.serde.getOrElse("").toLowerCase(Locale.ROOT)
    val tempFileFormat = if (serde.contains("parquet")) {
      // Parquet specific tagging
      tagGpuSupportForParquet(meta)
    } else {
      // Default to text file format
      tagGpuSupportForText(meta)
    }

    if (meta.canThisBeReplaced) {
      Some(tempFileFormat)
    } else {
      None
    }
  }

  private def tagGpuSupportForParquet(meta: GpuInsertIntoHiveTableMeta): ColumnarFileFormat = {
    val insertCmd = meta.wrapped
    // Configs check for Parquet write enabling/disabling
    // FIXME Need to check serde and output format classes ?

    // FIXME Need a new format type for Hive Parquet write ?
    FileFormatChecks.tag(meta, insertCmd.table.schema, ParquetFormatType, WriteFileOp)

    // Compression
    var compType = CompressionType.NONE
    if (isCompressionEnabled(insertCmd.conf)) {
      val parquetOptions = new ParquetOptions(insertCmd.table.properties, insertCmd.conf)
      val compressionType =
        GpuParquetFileFormat.parseCompressionType(parquetOptions.compressionCodecClassName)
      if (compressionType.nonEmpty) {
        compType = compressionType.get
      } else {
        meta.willNotWorkOnGpu(
          s"compression codec ${parquetOptions.compressionCodecClassName} is not supported")
      }
    }
    new GpuHiveParquetFileFormat(compType)
  }

  private def tagGpuSupportForText(meta: GpuInsertIntoHiveTableMeta): ColumnarFileFormat = {
    if (!meta.conf.isHiveDelimitedTextEnabled) {
      meta.willNotWorkOnGpu("Hive text I/O has been disabled. To enable this, " +
        s"set ${RapidsConf.ENABLE_HIVE_TEXT} to true")
    }
    if (!meta.conf.isHiveDelimitedTextWriteEnabled) {
      meta.willNotWorkOnGpu("writing Hive delimited text tables has been disabled, " +
        s"to enable this, set ${RapidsConf.ENABLE_HIVE_TEXT_WRITE} to true")
    }

    val insertCommand = meta.wrapped
    val storage  = insertCommand.table.storage
    if (storage.outputFormat.getOrElse("") != textOutputFormat) {
      meta.willNotWorkOnGpu(s"unsupported output-format found: ${storage.outputFormat}, " +
        s"only $textOutputFormat is currently supported for text")
    }
    if (storage.serde.getOrElse("") != lazySimpleSerDe) {
      meta.willNotWorkOnGpu(s"unsupported serde found: ${storage.serde}, " +
        s"only $lazySimpleSerDe is currently supported for text")
    }

    val serializationFormat = storage.properties.getOrElse(serializationKey, "1")
    if (serializationFormat != ctrlASeparatedFormat) {
      meta.willNotWorkOnGpu(s"unsupported serialization format found: " +
        s"$serializationFormat, " +
        s"only \'^A\' separated text output (i.e. serialization.format=1) " +
        s"is currently supported")
    }

    val lineTerminator = storage.properties.getOrElse(lineDelimiterKey, newLine)
    if (lineTerminator != newLine) {
      meta.willNotWorkOnGpu(s"unsupported line terminator found: " +
        s"$lineTerminator, " +
        s"only newline (\'\\n\') separated text output is currently supported")
    }

    if (!storage.properties.getOrElse(escapeDelimiterKey, "").equals("")) {
      meta.willNotWorkOnGpu("escapes are not currently supported")
      // "serialization.escape.crlf" matters only if escapeDelimiterKey is set
    }

    val charset = Charset.forName(
      storage.properties.getOrElse("serialization.encoding", "UTF-8"))
    if (!charset.equals(Charsets.UTF_8)) {
      meta.willNotWorkOnGpu("only UTF-8 is supported as the charset")
    }

    if (isCompressionEnabled(insertCommand.conf)) {
      meta.willNotWorkOnGpu("compressed output is not supported, " +
        "set hive.exec.compress.output to false to enable writing Hive text via GPU")
    }

    FileFormatChecks.tag(meta, insertCommand.table.schema, HiveDelimitedTextFormatType,
      WriteFileOp)

    new GpuHiveTextFileFormat()
  }

  private def isCompressionEnabled(conf: SQLConf): Boolean = {
    conf.getConfString("hive.exec.compress.output", "false").toBoolean
  }
}

class GpuHiveParquetFileFormat(compType: CompressionType) extends ColumnarFileFormat {

  override def prepareWrite(sparkSession: SparkSession, job: Job,
      options: Map[String, String], dataSchema: StructType): ColumnarOutputWriterFactory = {
    new ColumnarOutputWriterFactory {
      override def getFileExtension(context: TaskAttemptContext): String = ".parquet"

      override def newInstance(path: String,
          dataSchema: StructType,
          context: TaskAttemptContext): ColumnarOutputWriter = {
        new GpuHiveParquetWriter(path, dataSchema, context, compType)
      }
    }
  }
}

class GpuHiveParquetWriter(override val path: String, dataSchema: StructType,
    context: TaskAttemptContext, compType: CompressionType)
  extends ColumnarOutputWriter(context, dataSchema, "HiveParquet", false) {

  override protected val tableWriter: CudfTableWriter = {
    // TODO How to set INT96 and FieldIDEnabled ?
    val writeOptions = SchemaUtils
      .writerOptionsFromSchema(ParquetWriterOptions.builder(), dataSchema)
      .withCompressionType(compType)
      .build()
    Table.writeParquetChunked(writeOptions, this)
  }
}

class GpuHiveTextFileFormat extends ColumnarFileFormat with Logging {

  override def supportDataType(dataType: DataType): Boolean = isSupportedType(dataType)

  override def prepareWrite(sparkSession: SparkSession,
                            job: Job,
                            options: Map[String, String],
                            dataSchema: StructType): ColumnarOutputWriterFactory = {
    new ColumnarOutputWriterFactory {
      override def getFileExtension(context: TaskAttemptContext): String = ".txt"

      override def newInstance(path: String,
                               dataSchema: StructType,
                               context: TaskAttemptContext): ColumnarOutputWriter = {
        new GpuHiveTextWriter(path, dataSchema, context)
      }
    }
  }
}

class GpuHiveTextWriter(override val path: String,
                        dataSchema: StructType,
                        context: TaskAttemptContext)
  extends ColumnarOutputWriter(context, dataSchema, "HiveText", false) {

  /**
   * This reformats columns, to iron out inconsistencies between
   * CUDF serialization results and the values expected by Apache Spark
   * (and Apache Hive's) `LazySimpleSerDe`.
   *
   * This writer currently reformats timestamp and floating point
   * columns.
   */
  override def transformAndClose(cb: ColumnarBatch): ColumnarBatch = {
    withResource(cb) { _ =>
      withResource(GpuColumnVector.from(cb)) { table =>
        val columns = for (i <- 0 until table.getNumberOfColumns) yield {
          table.getColumn(i) match {
            case c if c.getType.hasTimeResolution =>
              // By default, the CUDF CSV writer writes timestamps in the following format:
              //   "2020-09-16T22:32:01.123456Z"
              // Hive's LazySimpleSerDe format expects timestamps to be formatted thus:
              //   "uuuu-MM-dd HH:mm:ss[.SSS...]"
              // (Specifically, no `T` between `dd` and `HH`, and no `Z` at the end.)
              val col = withResource(c.asStrings("%Y-%m-%d %H:%M:%S.%f")) { asStrings =>
                withResource(Scalar.fromString("\\N")) { nullString =>
                  asStrings.replaceNulls(nullString)
                }
              }
              GpuColumnVector.from(col, StringType)
            case c if c.getType == DType.FLOAT32 || c.getType == DType.FLOAT64 =>
              val col = CastStrings.fromFloat(c)
              GpuColumnVector.from(col, StringType)
            case c =>
              GpuColumnVector.from(c.incRefCount(), cb.column(i).dataType())
          }
        }
        new ColumnarBatch(columns.toArray, cb.numRows())
      }
    }
  }

  override val tableWriter: CudfTableWriter = {
    val writeOptions = CSVWriterOptions.builder()
      .withFieldDelimiter('\u0001')
      .withRowDelimiter("\n")
      .withIncludeHeader(false)
      .withTrueValue("true")
      .withFalseValue("false")
      .withNullValue("\\N")
      .withQuoteStyle(QuoteStyle.NONE)
      .build

    Table.getCSVBufferWriter(writeOptions, this)
  }
}

