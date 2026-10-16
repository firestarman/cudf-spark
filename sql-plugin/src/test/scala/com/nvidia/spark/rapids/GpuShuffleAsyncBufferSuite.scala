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

  private class TestTable(closedCount: AtomicInteger, val index: Int) extends AutoCloseable {
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
      onConcat: () => Unit = () => (),
      prefetchBatches: Int = 1,
      onBatch: Int => Unit = _ => (),
      waitTime: GpuMetric = NoopMetric,
      waitCount: GpuMetric = NoopMetric)
    extends GpuCoalesceIteratorBase[TestTable](
      input, 1L,
      NoopMetric, NoopMetric, NoopMetric, NoopMetric,
      useAsync = useAsync, bufferJobLaunchMetric = launchMetric,
      prefetchBatches = prefetchBatches, prefetchWaitTimeMetric = waitTime,
      prefetchWaitCountMetric = waitCount) {

    override protected val tableOperator =
      mock(classOf[SerializedTableOperator[TestTable, ColumnarBatch]])

    private var nextIndex = 0

    override protected def bufferNextBatch(): Unit = {
      if (input.hasNext) {
        input.next()
        nextIndex += 1
        serializedTables.addLast(new TestTable(closedCount, nextIndex))
        numTablesInBatch = 1
        numRowsInBatch = 1
        batchByteSize = 1L
        onBuffer()
      }
    }

    // The concurrency tests isolate buffering; GPU retry is covered by GpuShuffleCoalesceSuite.
    override protected def concatenateTables(tables: ArrayBuffer[TestTable]): ColumnarBatch = {
      onConcat()
      onBatch(tables.head.index)
      withResource(tables) { _ => new ColumnarBatch(Array.empty, tables.length) }
    }
  }

  private def initializeThrottler(): Unit = {
    TrafficController.initialize(new RapidsConf(Map.empty[String, String]))
  }

  test("GPU prefetch waits in hasNext are counted once and exclude the synchronous first group") {
    initializeThrottler()
    val consumer = Thread.currentThread()
    val producerStarted = new CountDownLatch(1)
    val releaseProducer = new CountDownLatch(1)
    val time = new LocalGpuMetric
    val count = PrefetchWaitCountTestMetric(() => releaseProducer.countDown())
    withResource(new TestIterator(Iterator.fill[ColumnarBatch](2)(null),
      true, new AtomicInteger, new LocalGpuMetric,
      onBuffer = () => {
        if (Thread.currentThread() != consumer) {
          producerStarted.countDown()
          assert(releaseProducer.await(10, TimeUnit.SECONDS))
        }
      }, waitTime = time, waitCount = count)) { iter =>
      withResource(iter.next()) { _ => () }
      assert(count.value == 0)
      assert(time.value == 0)
      assert(producerStarted.await(10, TimeUnit.SECONDS))
      assert(iter.hasNext)
      assert(count.value == 1)
      val recordedTime = time.value
      assert(recordedTime > 0)
      assert(iter.hasNext)
      withResource(iter.next()) { _ => () }
      assert(!iter.hasNext)
      assert(count.value == 1)
      assert(time.value == recordedTime)
    }
  }

  Seq(1, 2, 4).foreach { depth =>
    test(s"GPU shuffle prefetch depth $depth bounds and refills the input queue") {
      initializeThrottler()
      val consumer = Thread.currentThread()
      val closedCount = new AtomicInteger
      val backgroundReads = new AtomicInteger
      val initialQueueFilled = new CountDownLatch(depth)
      val refilled = new CountDownLatch(1)
      val merges = new AtomicInteger
      val consumed = new ArrayBuffer[Int]
      withResource(new TestIterator(Iterator.fill[ColumnarBatch](depth + 3)(null),
        true, closedCount, new LocalGpuMetric,
        onBuffer = () => {
          if (Thread.currentThread() != consumer) {
            val count = backgroundReads.incrementAndGet()
            initialQueueFilled.countDown()
            if (count > depth) refilled.countDown()
          }
        },
        onConcat = () => {
          assert(Thread.currentThread() == consumer)
          if (merges.incrementAndGet() == 1) {
            assert(initialQueueFilled.await(10, TimeUnit.SECONDS))
          }
        }, prefetchBatches = depth, onBatch = consumed += _)) { iter =>
        withResource(iter.next()) { _ => () }
        assert(backgroundReads.get() == depth)
        withResource(iter.next()) { _ => () }
        assert(refilled.await(10, TimeUnit.SECONDS))
        assert(backgroundReads.get() == depth + 1)
        (0 until depth + 1).foreach { _ => withResource(iter.next()) { _ => () } }
        assert(!iter.hasNext)
        assert(consumed.toSeq == (1 to depth + 3))
      }
      assert(closedCount.get() == depth + 3)
    }
  }

  test("closing GPU shuffle releases all completed prefetched groups") {
    initializeThrottler()
    val consumer = Thread.currentThread()
    val filled = new CountDownLatch(3)
    val closedCount = new AtomicInteger
    withResource(new TestIterator(Iterator.fill[ColumnarBatch](10)(null),
      true, closedCount, new LocalGpuMetric,
      onBuffer = () => {
        if (Thread.currentThread() != consumer) filled.countDown()
      }, prefetchBatches = 3)) { iter =>
      withResource(iter.next()) { _ => () }
      assert(filled.await(10, TimeUnit.SECONDS))
    }
    assert(closedCount.get() == 4)
  }

  test("GPU shuffle consumes the first prefetched group while a later group is blocked") {
    initializeThrottler()
    val consumer = Thread.currentThread()
    val reads = new AtomicInteger
    val laterReadStarted = new CountDownLatch(1)
    val releaseLaterRead = new CountDownLatch(1)
    val closedCount = new AtomicInteger
    withResource(new TestIterator(Iterator.fill[ColumnarBatch](4)(null),
      true, closedCount, new LocalGpuMetric,
      onBuffer = () => {
        if (Thread.currentThread() != consumer && reads.incrementAndGet() == 2) {
          laterReadStarted.countDown()
          releaseLaterRead.await()
        }
      }, prefetchBatches = 2)) { iter =>
      withResource(iter.next()) { _ => () }
      assert(laterReadStarted.await(10, TimeUnit.SECONDS))
      try {
        withResource(iter.next()) { _ => () }
      } finally {
        releaseLaterRead.countDown()
      }
    }
    assert(closedCount.get() >= 3)
  }

  test("GPU shuffle reports a later prefetch failure after earlier groups are consumed") {
    initializeThrottler()
    val consumer = Thread.currentThread()
    val created = new AtomicInteger
    val closedCount = new AtomicInteger
    val failure = new IllegalStateException("later prefetch failed")
    withResource(new TestIterator(Iterator.fill[ColumnarBatch](4)(null),
      true, closedCount, new LocalGpuMetric,
      onBuffer = () => {
        if (created.incrementAndGet() == 3) {
          assert(Thread.currentThread() != consumer)
          throw failure
        }
      }, prefetchBatches = 3)) { iter =>
      withResource(iter.next()) { _ => () }
      withResource(iter.next()) { _ => () }
      val caught = intercept[ExecutionException](iter.next())
      assert(caught.getCause eq failure)
    }
    assert(closedCount.get() == created.get())
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
      onBuffer = () => assert(Thread.currentThread() == consumer), prefetchBatches = 3)) { iter =>
      withResource(iter.next()) { _ => () }
      assert(launchMetric.value == 0)
      assert(prefetchedInputBatches.value == 0)
    }
    assert(closedCount.get() == 1)
  }
}
