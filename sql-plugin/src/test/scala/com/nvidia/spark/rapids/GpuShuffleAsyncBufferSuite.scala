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

import java.util.concurrent.{Callable, CountDownLatch, ExecutionException, FutureTask, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.mutable.ArrayBuffer

import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.io.async.TrafficController
import org.mockito.Mockito.mock
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.vectorized.ColumnarBatch

class GpuShuffleAsyncBufferSuite extends AnyFunSuite {
  private def onPrefetchThread[T](f: => T): T = {
    val task = new FutureTask[T](new Callable[T] {
      override def call(): T = f
    })
    val thread = new Thread(task)
    thread.setDaemon(true)
    thread.start()
    try {
      task.get(10, TimeUnit.SECONDS)
    } finally {
      task.cancel(true)
      thread.join(TimeUnit.SECONDS.toMillis(10))
    }
  }

  test("prefetched input batches count background reads before coalescing") {
    val metric = new LocalGpuMetric
    val batches = Seq.fill(4)(new ColumnarBatch(Array.empty, 1))
    withResource(batches) { _ =>
      val iter = new ShufflePrefetchMetricIterator(batches.iterator, metric)
      assert(iter.hasNext)
      assert(iter.next() eq batches.head)
      assert(metric.value == 0)

      onPrefetchThread {
        assert(iter.hasNext)
        assert(metric.value == 0)
        assert(iter.next() eq batches(1))
        assert(iter.next() eq batches(2))
      }
      assert(metric.value == 2)

      assert(iter.next() eq batches(3))
      assert(!iter.hasNext)
      assert(metric.value == 2)
    }
  }

  test("prefetched input batches do not count failed reads") {
    val metric = new LocalGpuMetric
    val iter = new ShufflePrefetchMetricIterator(Iterator.empty, metric)
    onPrefetchThread {
      intercept[NoSuchElementException](iter.next())
    }
    assert(metric.value == 0)
  }

  private class TestTable(closedCount: AtomicInteger) extends AutoCloseable {
    override def close(): Unit = {
      closedCount.incrementAndGet()
    }
  }

  /** Exercise producer ownership without allocating device memory or running GPU kernels. */
  private class TestIterator(
      input: Iterator[ColumnarBatch],
      useAsync: Boolean,
      closedCount: AtomicInteger,
      launchMetric: GpuMetric,
      onBuffer: () => Unit = () => (),
      onConcat: () => Unit = () => ())
    extends GpuCoalesceIteratorBase[TestTable](
      input, 1L,
      NoopMetric, NoopMetric, NoopMetric, NoopMetric,
      useAsync = useAsync, bufferJobLaunchMetric = launchMetric) {

    override protected val tableOperator =
      mock(classOf[SerializedTableOperator[TestTable, ColumnarBatch]])

    override protected def bufferNextBatch(): Unit = {
      if (input.hasNext) {
        input.next()
        serializedTables.addLast(new TestTable(closedCount))
        numTablesInBatch = 1
        numRowsInBatch = 1
        batchByteSize = 1L
        onBuffer()
      }
    }

    // The concurrency tests isolate buffering; GPU retry is covered by GpuShuffleCoalesceSuite.
    override protected def concatenateTables(tables: ArrayBuffer[TestTable]): ColumnarBatch = {
      onConcat()
      withResource(tables) { _ => new ColumnarBatch(Array.empty, tables.length) }
    }
  }

  private def initializeThrottler(): Unit = {
    TrafficController.initialize(new RapidsConf(Map.empty[String, String]))
  }

  test("GPU shuffle input prefetch overlaps merge on the consumer thread") {
    initializeThrottler()
    val consumer = Thread.currentThread()
    val buffered = new CountDownLatch(1)
    val closedCount = new AtomicInteger
    val launchMetric = new LocalGpuMetric
    val prefetchedInputBatches = new LocalGpuMetric
    val input = new ShufflePrefetchMetricIterator(
      Iterator.fill[ColumnarBatch](3)(null), prefetchedInputBatches)
    withResource(new TestIterator(input,
      true, closedCount, launchMetric,
      onBuffer = () => {
        if (Thread.currentThread() != consumer) buffered.countDown()
      },
      onConcat = () => {
        assert(Thread.currentThread() == consumer)
        assert(buffered.await(10, TimeUnit.SECONDS), "Prefetch did not overlap merge")
      })) { iter =>
      (0 until 3).foreach { _ => withResource(iter.next()) { batch =>
        assert(batch.numRows() == 1)
      }}
      assert(!iter.hasNext)
      assert(launchMetric.value > 0)
      assert(prefetchedInputBatches.value == 2)
    }
    assert(closedCount.get() == 3)
  }

  test("closing GPU shuffle stops a pending producer before freeing its input") {
    initializeThrottler()
    val consumer = Thread.currentThread()
    val buffered = new CountDownLatch(1)
    val blocked = new CountDownLatch(1)
    val closedCount = new AtomicInteger
    withResource(new TestIterator(Iterator.fill[ColumnarBatch](2)(null),
      true, closedCount, new LocalGpuMetric,
      onBuffer = () => {
        if (Thread.currentThread() != consumer) {
          buffered.countDown()
          blocked.await()
        }
      })) { iter =>
      withResource(iter.next()) { _ =>
        assert(buffered.await(10, TimeUnit.SECONDS))
      }
      assert(closedCount.get() == 1)
      // ARM close interrupts the pending read and releases its buffered table.
    }
    assert(closedCount.get() == 2)
  }

  test("GPU shuffle propagates background read failures and releases buffered input") {
    initializeThrottler()
    val consumer = Thread.currentThread()
    val failure = new IllegalStateException("injected shuffle read failure")
    val closedCount = new AtomicInteger
    withResource(new TestIterator(Iterator.fill[ColumnarBatch](2)(null),
      true, closedCount, new LocalGpuMetric,
      onBuffer = () => {
        if (Thread.currentThread() != consumer) throw failure
      })) { iter =>
      withResource(iter.next()) { _ => () }
      val caught = intercept[ExecutionException](iter.next())
      assert(caught.getCause eq failure)
    }
    assert(closedCount.get() == 2)
  }

  test("GPU shuffle async read can be disabled") {
    val consumer = Thread.currentThread()
    val closedCount = new AtomicInteger
    val launchMetric = new LocalGpuMetric
    val prefetchedInputBatches = new LocalGpuMetric
    val input = new ShufflePrefetchMetricIterator(
      Iterator.fill[ColumnarBatch](1)(null), prefetchedInputBatches)
    withResource(new TestIterator(input,
      false, closedCount, launchMetric,
      onBuffer = () => assert(Thread.currentThread() == consumer))) { iter =>
      withResource(iter.next()) { _ => () }
      assert(launchMetric.value == 0)
      assert(prefetchedInputBatches.value == 0)
    }
    assert(closedCount.get() == 1)
  }
}
