/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import org.apache.spark.sql.connector.read.Scan

trait GpuBatchScanExecMetrics extends GpuExec {
  import GpuMetric._

  def scan: Scan

  override def supportsColumnar = true

  override val outputRowsLevel: MetricsLevel = ESSENTIAL_LEVEL
  override val outputBatchesLevel: MetricsLevel = MODERATE_LEVEL
  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    GPU_DECODE_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_GPU_DECODE_TIME),
    BUFFER_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_BUFFER_TIME),
    FILTER_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_FILTER_TIME)
  ) ++ fileCacheMetrics

  lazy val fileCacheMetrics: Map[String, GpuMetric] = {
    // File cache only supported on Parquet files for now.
    scan match {
      case _: GpuParquetScan | _: GpuOrcScan =>
        createFileCacheMetrics() ++
          // For debugging ByteDance workloads
          Map(
            "compPageSize" -> createSizeMetric(DEBUG_LEVEL, "compressed page size"),
            "unCompPageSize" -> createSizeMetric(DEBUG_LEVEL, "uncompressed page size"),
            "nullCount" -> createSizeMetric(DEBUG_LEVEL, "null record count"),
            "maxCPR" -> createAverageMetric(DEBUG_LEVEL, "max compression ratio"),
            "minCPR" -> createAverageMetric(DEBUG_LEVEL, "min compression ratio"),
            "maxFieldSize" -> createAverageMetric(DEBUG_LEVEL, "max size of single field"),
            BUFFER_DATA_TIME -> createNanoTimingMetric(DEBUG_LEVEL, BUFFER_DATA_TIME),
            BUFFER_META_TIME -> createNanoTimingMetric(DEBUG_LEVEL, BUFFER_META_TIME),
            BUFFER_RESIZE_TIME -> createNanoTimingMetric(DEBUG_LEVEL, BUFFER_RESIZE_TIME))
      case _ => Map.empty
    }
  }
}
