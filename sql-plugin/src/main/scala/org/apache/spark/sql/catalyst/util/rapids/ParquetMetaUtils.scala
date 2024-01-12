/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

package org.apache.spark.sql.catalyst.util.rapids

import java.io.PrintWriter

import scala.collection.JavaConverters._
import scala.collection.mutable

import org.apache.parquet.hadoop.metadata.{BlockMetaData, ColumnChunkMetaData}

object ParquetMetaUtils {

  def showDetails(out: PrintWriter, meta: BlockMetaData): Unit = {
    showDetails(out, meta, None)
  }

  private def showDetails(out: PrintWriter, meta: BlockMetaData, num: Option[Long]): Unit = {
    val rows = meta.getRowCount
    val tbs = meta.getTotalByteSize
    val offset = meta.getStartingPos
    out.println(s"row group${num.fold("")(" " + _)}: RC:$rows TS:$tbs OFFSET:$offset")
    out.println("-----------------------------")
    showDetails(out, meta.getColumns.asScala)
  }

  private case class PathNode(value: Either[ColumnChunkMetaData,
    mutable.LinkedHashMap[String, PathNode]])

  def showDetails(out: PrintWriter, colChunkMeta: Seq[ColumnChunkMetaData]): Unit = {
    val chunks = PathNode(Right(mutable.LinkedHashMap[String, PathNode]()))

    colChunkMeta.foreach { meta =>
      val paths = meta.getPath.toArray
      var cursor = chunks
      (0 until paths.length).foreach { i =>
        cursor.value match {
          case Right(children) if i == paths.length - 1 =>
            children(paths(i)) = PathNode(Left(meta))
          case Right(children) =>
            cursor = children.getOrElseUpdate(paths(i),
              PathNode(Right(mutable.LinkedHashMap[String, PathNode]())))
          case Left(_) =>
            throw new Exception("found leaf value in a non-leaf path")
        }
      }
    }

    showColumnChunkDetails(out, chunks, 0)
  }

  private def showColumnChunkDetails(out: PrintWriter, current: PathNode, depth: Int): Unit = {
    current.value match {
      case Right(children) =>
        children.foreach { case (key, child) =>
          val name = "." * depth + key
          child.value match {
            case Right(_) =>
              out.println(s"$name: ")
              showColumnChunkDetails(out, child, depth + 1)
            case Left(meta) =>
              out.print(s"$name: ")
              showDetails(out, meta, includeName = false)
          }
        }
      case Left(_) =>
        throw new Exception("found leaf value in a non-leaf path")
    }
  }

  private def showDetails(out: PrintWriter,
                          meta: ColumnChunkMetaData,
                          includeName: Boolean): Unit = {
    val dictPageOff = meta.getDictionaryPageOffset
    val firstPageOff = meta.getFirstDataPageOffset
    val compSize = meta.getTotalSize
    val unCompSize = meta.getTotalUncompressedSize
    val count = meta.getValueCount
    val ratio = unCompSize.toDouble / compSize
    val encodings = meta.getEncodings.asScala.filterNot(_ == null).mkString(",")

    if (includeName) {
      val path = meta.getPath.asScala.filterNot(_ == null).mkString(".")
      out.print(s"$path: ")
    }
    out.println(s" ${meta.getPrimitiveType} ${meta.getCodec} DO:$dictPageOff " +
      s"FPO:$firstPageOff SZ:$compSize/$unCompSize/$ratio VC:$count ENC:$encodings")
    val stats = Option(meta.getStatistics).map(_.toString).getOrElse("none")
    out.println(s" ST:[$stats]")
  }

}
