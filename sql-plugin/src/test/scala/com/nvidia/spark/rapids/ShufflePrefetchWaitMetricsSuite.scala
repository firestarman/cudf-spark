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

import java.util.concurrent.{CompletableFuture, ExecutionException}

import org.mockito.ArgumentMatchers.anyLong
import org.mockito.Mockito.{doAnswer, mock, when}
import org.scalatest.funsuite.AnyFunSuite

/** Signals the producer after the consumer observes an incomplete future, without timing sleeps. */
private[rapids] object PrefetchWaitCountTestMetric {
  def apply(onWait: () => Unit): GpuMetric = {
    val metric = mock(classOf[GpuMetric])
    val counter = new LocalGpuMetric
    doAnswer(invocation => {
      counter += invocation.getArgument[Long](0)
      onWait()
      null
    }).when(metric).+=(anyLong())
    when(metric.value).thenAnswer(_ => counter.value)
    metric
  }
}

class ShufflePrefetchWaitMetricsSuite extends AnyFunSuite {
  test("completed prefetches have zero wait count and time") {
    val time = new LocalGpuMetric
    val count = new LocalGpuMetric
    assert(ShufflePrefetchWaitMetrics.await(CompletableFuture.completedFuture(7), time, count) == 7)
    assert(count.value == 0)
    assert(time.value == 0)
  }

  test("a prefetch wait is recorded once across repeated accesses") {
    val future = new CompletableFuture[Int]
    val time = new LocalGpuMetric
    val count = PrefetchWaitCountTestMetric(() => { future.complete(7); () })
    assert(ShufflePrefetchWaitMetrics.await(future, time, count) == 7)
    val recordedTime = time.value
    assert(count.value == 1)
    assert(recordedTime > 0)
    assert(ShufflePrefetchWaitMetrics.await(future, time, count) == 7)
    assert(count.value == 1)
    assert(time.value == recordedTime)
  }

  test("waits discovering a producer failure are timed and propagate the failure") {
    val future = new CompletableFuture[Int]
    val failure = new IllegalStateException("prefetch failed")
    val time = new LocalGpuMetric
    val count = PrefetchWaitCountTestMetric(() => { future.completeExceptionally(failure); () })
    val caught = intercept[ExecutionException] {
      ShufflePrefetchWaitMetrics.await(future, time, count)
    }
    assert(caught.getCause eq failure)
    assert(count.value == 1)
    assert(time.value > 0)
  }

  test("waits to discover EOF are included") {
    val future = new CompletableFuture[Option[Int]]
    val time = new LocalGpuMetric
    val count = PrefetchWaitCountTestMetric(() => { future.complete(None); () })
    assert(ShufflePrefetchWaitMetrics.await(future, time, count).isEmpty)
    assert(count.value == 1)
    assert(time.value > 0)
  }
}
