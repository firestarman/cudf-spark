/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

import scala.collection.mutable
import scala.ref.WeakReference

import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.exchange.Exchange

/**A trait mixed with a GPU version of Exchange to support to mark if it is optimized by GPU*/
trait SupportPlanningMark extends Logging { this: Exchange =>

  // Some GPU Shuffle exchanges are used internally without a CPU instance, it is an Option.
  val cpuCanonicalExec: Option[Exchange]

  private var _isGpuPlanningComplete = false

  /**
   * Returns true if this node and children are finished being optimized by the RAPIDS Accelerator.
   */
  final def isGpuPlanningComplete: Boolean = _isGpuPlanningComplete

  /**
   * Method to call after all RAPIDS Accelerator optimizations have been applied
   * to indicate this node and its children are done being planned by the RAPIDS Accelerator.
   * Some optimizations, such as AQE exchange reuse fixup, need to know when a node will no longer
   * be updated so it can be tracked for reuse.
   */
  final def markGpuPlanningComplete(): Unit = {
    if (!_isGpuPlanningComplete) {
      _isGpuPlanningComplete = true
      if (cpuCanonicalExec.isDefined) {
        ExchangeMappingCache.trackExchangeMapping(cpuCanonicalExec.get, this)
      } else {
        logInfo("Trying to mark an exchange as GPU planning done without a CPU instance")
      }
    }
  }
}

/** Caches the mappings from canonical CPU exchanges to the GPU exchanges that replaced them */
object ExchangeMappingCache extends Logging {
  // Cache is a mapping from CPU broadcast plan to GPU broadcast plan. The cache should not
  // artificially hold onto unused plans, so we make both the keys and values weak. The values
  // point to their corresponding keys, so the keys will not be collected unless the value
  // can be collected. The values will be held during normal Catalyst planning until those
  // plans are no longer referenced, allowing both the key and value to be reaped at that point.
  private val cache = new mutable.WeakHashMap[Exchange, WeakReference[Exchange]]

  /** Try to find a recent GPU exchange that has replaced the specified CPU canonical plan. */
  def findGpuExchangeReplacement(cpuCanonical: Exchange): Option[Exchange] = {
    cache.get(cpuCanonical).flatMap(_.get)
  }

  /** Add a GPU exchange to the exchange cache */
  def trackExchangeMapping(cpuCanonical: Exchange, gpuExchange: Exchange): Unit = {
    val old = findGpuExchangeReplacement(cpuCanonical)
    if (!old.exists(_.asInstanceOf[SupportPlanningMark].isGpuPlanningComplete)) {
      cache.put(cpuCanonical, WeakReference(gpuExchange))
    }
  }
}
