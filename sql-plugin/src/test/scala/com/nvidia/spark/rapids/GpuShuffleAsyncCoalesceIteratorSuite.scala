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

import java.util.concurrent.{Callable, CountDownLatch, ExecutionException, Executors, TimeUnit}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger}

import scala.collection.mutable.ArrayBuffer

import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.io.async.{HostMemoryThrottle, ThrottlingExecutor, TrafficController}
import org.mockito.Mockito.mock
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

class GpuShuffleAsyncCoalesceIteratorSuite extends AnyFunSuite {
  private class HostInput(
      total: Int,
      consumer: Thread,
      created: AtomicInteger,
      closedCount: AtomicInteger,
      onRead: Int => Unit = _ => ()) extends Iterator[CoalescedHostResult] {
    private var nextIndex = 1

    override def hasNext: Boolean = nextIndex <= total

    override def next(): CoalescedHostResult = {
      if (!hasNext) throw new NoSuchElementException
      val index = nextIndex
      onRead(index)
      nextIndex += 1
      created.incrementAndGet()
      new CoalescedHostResult {
        private val closed = new AtomicBoolean

        override def getDataSize: Long = 1L

        override def toGpuBatch(types: Array[DataType]): ColumnarBatch = {
          assert(Thread.currentThread() == consumer)
          new ColumnarBatch(Array.empty, index)
        }

        override def close(): Unit = {
          assert(closed.compareAndSet(false, true), "Host result closed twice")
          closedCount.incrementAndGet()
        }
      }
    }
  }

  private def reader(input: Iterator[CoalescedHostResult], depth: Int)
      : GpuShuffleAsyncCoalesceIterator = {
    TrafficController.initialize(new RapidsConf(Map.empty[String, String]))
    new GpuShuffleAsyncCoalesceIterator(input, Array.empty, 1L, prefetchBatches = depth)
  }

  test("host prefetch waits in hasNext are counted once and exclude the synchronous first group") {
    TrafficController.initialize(new RapidsConf(Map.empty[String, String]))
    val producerStarted = new CountDownLatch(1)
    val releaseProducer = new CountDownLatch(1)
    val created = new AtomicInteger
    val closedCount = new AtomicInteger
    val time = new LocalGpuMetric
    val count = PrefetchWaitCountTestMetric(() => releaseProducer.countDown())
    val input = new HostInput(2, Thread.currentThread(), created, closedCount, index => {
      if (index == 2) {
        producerStarted.countDown()
        assert(releaseProducer.await(10, TimeUnit.SECONDS))
      }
    })
    withResource(new GpuShuffleAsyncCoalesceIterator(input, Array.empty, 1L,
      prefetchWaitTimeMetric = time, prefetchWaitCountMetric = count)) { iter =>
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
      assert(count.value == 1)
      assert(time.value == recordedTime)
    }
    assert(closedCount.get() == created.get())
  }

  Seq(1, 2, 4).foreach { depth =>
    test(s"host shuffle prefetch depth $depth bounds the queue and preserves result order") {
      val consumer = Thread.currentThread()
      val created = new AtomicInteger
      val closedCount = new AtomicInteger
      val filled = new CountDownLatch(depth)
      val refilled = new CountDownLatch(1)
      val input = new HostInput(depth + 3, consumer, created, closedCount, index => {
        if (index == 1) {
          assert(Thread.currentThread() == consumer)
        } else {
          assert(Thread.currentThread() != consumer)
          if (index <= depth + 1) filled.countDown() else refilled.countDown()
        }
      })
      withResource(reader(input, depth)) { iter =>
        withResource(iter.next()) { batch => assert(batch.numRows() == 1) }
        assert(filled.await(10, TimeUnit.SECONDS))
        // The final onRead callback runs just before its result is created. hasNext joins
        // only the first group, so use the read index rather than created as the queue bound.
        assert(created.get() <= depth + 1)
        withResource(iter.next()) { batch => assert(batch.numRows() == 2) }
        assert(refilled.await(10, TimeUnit.SECONDS))
        (3 to depth + 3).foreach { index =>
          withResource(iter.next()) { batch => assert(batch.numRows() == index) }
        }
        assert(!iter.hasNext)
        intercept[NoSuchElementException](iter.next())
      }
      assert(created.get() == depth + 3)
      assert(closedCount.get() == created.get())
    }
  }

  test("host shuffle consumes an earlier result while a later coalesce is blocked") {
    val consumer = Thread.currentThread()
    val created = new AtomicInteger
    val closedCount = new AtomicInteger
    val laterReadStarted = new CountDownLatch(1)
    val releaseLaterRead = new CountDownLatch(1)
    val input = new HostInput(4, consumer, created, closedCount, index => {
      if (index == 3) {
        laterReadStarted.countDown()
        releaseLaterRead.await()
      }
    })
    withResource(reader(input, 2)) { iter =>
      withResource(iter.next()) { _ => () }
      assert(laterReadStarted.await(10, TimeUnit.SECONDS))
      try {
        withResource(iter.next()) { batch => assert(batch.numRows() == 2) }
      } finally {
        releaseLaterRead.countDown()
      }
    }
    assert(closedCount.get() == created.get())
  }

  test("host shuffle propagates later coalesce failure after earlier results") {
    val created = new AtomicInteger
    val closedCount = new AtomicInteger
    val failure = new IllegalStateException("injected host coalesce failure")
    val input = new HostInput(5, Thread.currentThread(), created, closedCount, index => {
      if (index == 3) throw failure
    })
    withResource(reader(input, 3)) { iter =>
      withResource(iter.next()) { batch => assert(batch.numRows() == 1) }
      withResource(iter.next()) { batch => assert(batch.numRows() == 2) }
      val caught = intercept[ExecutionException](iter.next())
      assert(caught.getCause eq failure)
    }
    assert(created.get() == 2)
    assert(closedCount.get() == 2)
  }

  test("closing host shuffle interrupts a pending read and releases completed results") {
    val created = new AtomicInteger
    val closedCount = new AtomicInteger
    val blocked = new CountDownLatch(1)
    val input = new HostInput(10, Thread.currentThread(), created, closedCount, index => {
      if (index == 4) {
        blocked.countDown()
        new CountDownLatch(1).await()
      }
    })
    val iter = reader(input, 4)
    withResource(iter) { _ =>
      withResource(iter.next()) { _ => () }
      assert(blocked.await(10, TimeUnit.SECONDS))
    }
    iter.close()
    assert(!iter.hasNext)
    assert(created.get() == 3)
    assert(closedCount.get() == 3)
  }

  test("host shuffle releases distinct equal results on early close") {
    val created = new AtomicInteger
    val closedCount = new AtomicInteger
    val filled = new CountDownLatch(2)
    val input = new HostInput(3, Thread.currentThread(), created, closedCount, index => {
      if (index > 1) filled.countDown()
    }).map { result =>
      // Equality must not cause consumption to remove a different result's close ownership.
      new CoalescedHostResult {
        override def getDataSize: Long = result.getDataSize
        override def toGpuBatch(types: Array[DataType]): ColumnarBatch = result.toGpuBatch(types)
        override def close(): Unit = result.close()
        override def equals(other: Any): Boolean = other.isInstanceOf[CoalescedHostResult]
        override def hashCode(): Int = 1
      }
    }
    withResource(reader(input, 2)) { iter =>
      withResource(iter.next()) { _ => () }
      assert(filled.await(10, TimeUnit.SECONDS))
      withResource(iter.next()) { _ => () }
    }
    assert(closedCount.get() == created.get())
  }

  test("host shuffle handles empty input and rejects invalid depth") {
    withResource(reader(Iterator.empty, 2)) { iter =>
      assert(!iter.hasNext)
      intercept[NoSuchElementException](iter.next())
    }
    intercept[IllegalArgumentException](reader(Iterator.empty, 0))
  }

  test("closing both host prefetch stages stops the inner producer before freeing its input") {
    TrafficController.initialize(new RapidsConf(Map.empty[String, String]))
    val consumer = Thread.currentThread()
    val created = new AtomicInteger
    val closedCount = new AtomicInteger
    val blocked = new CountDownLatch(1)
    val input = Iterator.fill[ColumnarBatch](10)(null)
    class InputTable(val index: Int) extends AutoCloseable {
      override def close(): Unit = {
        closedCount.incrementAndGet()
      }
    }
    val host = new HostCoalesceIteratorBase[InputTable](input, 1L,
      NoopMetric, NoopMetric, NoopMetric, NoopMetric, useAsync = true) {
      override protected val tableOperator =
        mock(classOf[SerializedTableOperator[InputTable, CoalescedHostResult]])

      override protected def bufferNextBatch(): Unit = {
        if (input.hasNext) {
          input.next()
          val index = created.incrementAndGet()
          serializedTables.addLast(new InputTable(index))
          numTablesInBatch = 1
          numRowsInBatch = 1
          batchByteSize = 1L
          if (index == 4) {
            blocked.countDown()
            new CountDownLatch(1).await()
          }
        }
      }

      override protected def concatenateTables(tables: ArrayBuffer[InputTable])
          : CoalescedHostResult = {
        withResource(tables) { _ => RowCountOnlyMergeResult(tables.head.index) }
      }
    }
    withResource(host) { _ =>
      withResource(reader(host, 2)) { iter =>
        withResource(iter.next()) { batch =>
          assert(Thread.currentThread() == consumer)
          assert(batch.numRows() == 1)
        }
        assert(blocked.await(10, TimeUnit.SECONDS))
      }
      assert(closedCount.get() == 3)
    }
    assert(created.get() == 4)
    assert(closedCount.get() == 4)
  }

  test("queued host coalesces leave memory budget available for nested input prefetch") {
    val throttle = new HostMemoryThrottle(2L)
    val controller = new TrafficController(throttle)
    val coalesces = new ThrottlingExecutor(Executors.newSingleThreadExecutor(), controller, _ => ())
    val inputs = new ThrottlingExecutor(Executors.newSingleThreadExecutor(), controller, _ => ())
    val started = new CountDownLatch(1)
    val queued = new CountDownLatch(1)
    try {
      val first = coalesces.submitWithDeferredReservation(new Callable[Int] {
        override def call(): Int = {
          started.countDown()
          queued.await()
          val input = inputs.submit(new Callable[Int] {
            override def call(): Int = 42
          }, 1L)
          input.get(10, TimeUnit.SECONDS)
        }
      }, 1L)
      assert(started.await(10, TimeUnit.SECONDS))
      val later = coalesces.submitWithDeferredReservation(new Callable[Int] {
        override def call(): Int = 43
      }, 1L)
      val cancelled = coalesces.submitWithDeferredReservation(new Callable[Int] {
        override def call(): Int = fail("Cancelled coalesce should not run")
      }, 1L)
      cancelled.cancel(false)
      assert(controller.numScheduledTasks == 1)
      assert(throttle.getTotalHostMemoryBytes == 1L)
      queued.countDown()
      assert(first.get(10, TimeUnit.SECONDS) == 42)
      assert(later.get(10, TimeUnit.SECONDS) == 43)
      assert(controller.numScheduledTasks == 0)
      assert(throttle.getTotalHostMemoryBytes == 0L)
    } finally {
      queued.countDown()
      coalesces.shutdownNow(10, TimeUnit.SECONDS)
      inputs.shutdownNow(10, TimeUnit.SECONDS)
    }
  }
}
