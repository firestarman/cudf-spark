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

import java.util.{Comparator, LinkedList, PriorityQueue}

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{ColumnVector, ContiguousTable, NvtxColor, NvtxRange, Table}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits.{AutoCloseableProducingSeq, AutoCloseableSeq}
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.{splitSpillableInHalfByRows, splitTargetSizeInHalfGpu, withRetry, withRetryNoSplit}
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import com.nvidia.spark.rapids.shims.ShimUnaryExecNode

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder, UnsafeProjection, UnsafeRow}
import org.apache.spark.sql.catalyst.expressions.codegen.LazilyGeneratedOrdering
import org.apache.spark.sql.catalyst.plans.physical.{Distribution, OrderedDistribution, Partitioning, UnspecifiedDistribution}
import org.apache.spark.sql.execution.{SortExec, SparkPlan}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.execution.TrampolineUtil
import org.apache.spark.sql.vectorized.ColumnarBatch

sealed trait SortExecType extends Serializable

object OutOfCoreSort extends SortExecType
object FullSortSingleBatch extends SortExecType
object SortEachBatch extends SortExecType

class GpuSortMeta(
    sort: SortExec,
    conf: RapidsConf,
    parent: Option[RapidsMeta[_, _, _]],
    rule: DataFromReplacementRule)
  extends SparkPlanMeta[SortExec](sort, conf, parent, rule) {

  // Uses output attributes of child plan because SortExec will not change the attributes,
  // and we need to propagate possible type conversions on the output attributes of
  // GpuSortAggregateExec.
  override protected val useOutputAttributesOfChild: Boolean = true

  // For transparent plan like ShuffleExchange, the accessibility of runtime data transition is
  // depended on the next non-transparent plan. So, we need to trace back.
  override val availableRuntimeDataTransition: Boolean =
    childPlans.head.availableRuntimeDataTransition

  override def convertToGpu(): GpuExec = {
    GpuSortExec(childExprs.map(_.convertToGpu()).asInstanceOf[Seq[SortOrder]],
      sort.global,
      childPlans.head.convertIfNeeded(),
      if (conf.stableSort) FullSortSingleBatch else OutOfCoreSort
    )(sort.sortOrder)
  }
}

object GpuSortExec {
  def targetSize(sqlConf: SQLConf): Long = {
    val batchSize = RapidsConf.GPU_BATCH_SIZE_BYTES.get(sqlConf)
    targetSize(batchSize)
  }

  def targetSize(batchSize: Long): Long = {
    // To avoid divide by zero errors, underflow and overflow issues in tests
    // that want the targetSize to be 0, we set it to something more reasonable
    math.max(16 * 1024, batchSize)
  }
}

case class GpuSortExec(
    gpuSortOrder: Seq[SortOrder],
    global: Boolean,
    child: SparkPlan,
    sortType: SortExecType)(cpuSortOrder: Seq[SortOrder])
  extends ShimUnaryExecNode with GpuExec {

  override def otherCopyArgs: Seq[AnyRef] = cpuSortOrder :: Nil

  override def childrenCoalesceGoal: Seq[CoalesceGoal] = sortType match {
    case FullSortSingleBatch => Seq(RequireSingleBatch)
    case OutOfCoreSort | SortEachBatch => Seq(null)
    case t => throw new IllegalArgumentException(s"Unexpected Sort Type $t")
  }

  override def output: Seq[Attribute] = child.output

  override def outputOrdering: Seq[SortOrder] = cpuSortOrder

  // sort performed is local within a given partition so will retain
  // child operator's partitioning
  override def outputPartitioning: Partitioning = child.outputPartitioning

  override def requiredChildDistribution: Seq[Distribution] =
    if (global) OrderedDistribution(cpuSortOrder) :: Nil else UnspecifiedDistribution :: Nil

  override def outputBatching: CoalesceGoal = sortType match {
    // We produce a single batch if we know that our input will be a single batch
    case FullSortSingleBatch => RequireSingleBatch
    case _ => null
  }

  override def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException(s"Row-based execution should not occur for $this")

  override lazy val additionalMetrics: Map[String, GpuMetric] =
    Map(
      OP_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_OP_TIME),
      SORT_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_SORT_TIME))

  private lazy val targetSize = GpuSortExec.targetSize(conf)

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val sorter = new GpuSorter(gpuSortOrder, output)

    val sortTime = gpuLongMetric(SORT_TIME)
    val opTime = gpuLongMetric(OP_TIME)
    val outputBatch = gpuLongMetric(NUM_OUTPUT_BATCHES)
    val outputRows = gpuLongMetric(NUM_OUTPUT_ROWS)
    val outOfCore = sortType == OutOfCoreSort
    val singleBatch = sortType == FullSortSingleBatch
    child.executeColumnar().mapPartitions { cbIter =>
      if (outOfCore) {
        val iter = GpuOutOfCoreSortIterator(cbIter, sorter,
          targetSize, opTime, sortTime, outputBatch, outputRows)
        onTaskCompletion(iter.close())
        iter
      } else {
        GpuSortEachBatchIterator(cbIter, sorter, singleBatch,
          opTime, sortTime, outputBatch, outputRows)
      }
    }
  }
}

case class GpuSortEachBatchIterator(
    iter: Iterator[ColumnarBatch],
    sorter: GpuSorter,
    singleBatch: Boolean,
    opTime: GpuMetric = NoopMetric,
    sortTime: GpuMetric = NoopMetric,
    outputBatches: GpuMetric = NoopMetric,
    outputRows: GpuMetric = NoopMetric) extends Iterator[ColumnarBatch] {
  override def hasNext: Boolean = iter.hasNext

  override def next(): ColumnarBatch = {
    if (!hasNext) {
      throw new NoSuchElementException()
    }
    val scb = closeOnExcept(iter.next()) { cb =>
      SpillableColumnarBatch(cb, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
    }
    val ret = sorter.fullySortBatchAndCloseWithRetry(scb, sortTime, opTime)
    opTime.ns {
      outputBatches += 1
      outputRows += ret.numRows()
      if (singleBatch) {
        GpuColumnVector.tagAsFinalBatch(ret)
      } else {
        ret
      }
    }
  }
}

/**
 * Create an iterator that will sort each batch as it comes in. It will keep any projected
 * columns in place after doing the sort on the assumption that you want to possibly combine
 * them in some way afterwards.
 */
object GpuSpillableProjectedSortEachBatchIterator {
  def apply(
      iter: Iterator[ColumnarBatch],
      sorter: GpuSorter,
      opTime: GpuMetric = NoopMetric,
      sortTime: GpuMetric = NoopMetric): Iterator[SpillableColumnarBatch] = {
    val spillableIter = iter.flatMap { cb =>
        // Filter out empty batches and make them spillable
        if (cb.numRows() > 0) {
          Some(SpillableColumnarBatch(cb, SpillPriorities.ACTIVE_ON_DECK_PRIORITY))
        } else {
          cb.close()
          None
        }
    }

    val sortedBatchIter = spillableIter.flatMap { scb =>
      withRetry(scb, splitSpillableInHalfByRows) { attemptScb =>
        opTime.ns {
          val sortedTbl = withResource(attemptScb.getColumnarBatch()) { attemptCb =>
            sorter.appendProjectedAndSort(attemptCb, sortTime)
          }
          withResource(sortedTbl) { _ =>
            closeOnExcept(GpuColumnVector.from(sortedTbl, sorter.projectedBatchTypes)) { cb =>
              SpillableColumnarBatch(cb, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
            }
          }
        }
      }
    }
    sortedBatchIter
  }
}

/**
 * Holds data for the out of core sort. It includes the batch of data and the first row in that
 * batch so we can sort the batches.
 */
case class OutOfCoreBatch(buffer: SpillableColumnarBatch,
    firstRow: UnsafeRow) extends AutoCloseable {
  override def close(): Unit = buffer.close()
}

/**
 * Data that the out of core sort algorithm has not finished sorting. This acts as a priority
 * queue with each batch sorted by the first row in that batch.
 */
class Pending(cpuOrd: LazilyGeneratedOrdering) extends AutoCloseable {
  private val pending = new PriorityQueue[OutOfCoreBatch](new Comparator[OutOfCoreBatch]() {
    override def compare(a: OutOfCoreBatch, b: OutOfCoreBatch): Int =
      cpuOrd.compare(a.firstRow, b.firstRow)
  })
  private var pendingSize = 0L

  def add(batch: OutOfCoreBatch): Unit = {
    pendingSize += batch.buffer.sizeInBytes
    pending.add(batch)
  }

  def storedSize: Long = pendingSize

  def size(): Int = pending.size()

  def poll(): OutOfCoreBatch = {
    val ret = pending.poll()
    if (ret != null) {
      pendingSize -= ret.buffer.sizeInBytes
    }
    ret
  }

  def peek(): OutOfCoreBatch = pending.peek()

  def isEmpty: Boolean = pending.isEmpty

  override def close(): Unit = pending.forEach(_.close())
}

/**
 * Sorts incoming batches of data spilling if needed.
 * <br/>
 * The algorithm for this is a modified version of an external merge sort with multiple passes for
 * large data.
 * https://en.wikipedia.org/wiki/External_sorting#External_merge_sort
 * <br/>
 * The main difference is that we cannot stream the data when doing a merge sort. So, we instead
 * divide the data into batches that are small enough that we can do a merge sort on N batches
 * and still fit the output within the target batch size. When merging batches instead of
 * individual rows we cannot assume that all of the resulting data is globally sorted. Hopefully,
 * most of it is globally sorted but we have to use the first row from the next pending batch to
 * determine the cutoff point between globally sorted data and data that still needs to be merged
 * with other batches. The globally sorted portion is put into a sorted queue while the rest of
 * the merged data is split and put back into a pending queue.  The process repeats until we have
 * enough data to output.
 */
case class GpuOutOfCoreSortIterator(
    iter: Iterator[ColumnarBatch],
    sorter: GpuSorter,
    targetSize: Long,
    opTime: GpuMetric,
    sortTime: GpuMetric,
    outputBatches: GpuMetric,
    outputRows: GpuMetric) extends Iterator[ColumnarBatch]
    with AutoCloseable with Logging {

  /**
   * This has already sorted the data, and it still has the projected columns in it that need to
   * be removed before it is returned.
   */
  private val alreadySortedIter =
    GpuSpillableProjectedSortEachBatchIterator(iter, sorter, opTime, sortTime)

  private val cpuOrd = new LazilyGeneratedOrdering(sorter.cpuOrdering)
  // A priority queue of data that is not merged yet.
  private val pending = new Pending(cpuOrd)

  // data that has been determined to be globally sorted and is waiting to be output.
  private val sorted = new LinkedList[SpillableColumnarBatch]()
  // how much data, in bytes, that is stored in `sorted`
  private var sortedSize = 0L

  override def hasNext: Boolean = !sorted.isEmpty || !pending.isEmpty || alreadySortedIter.hasNext

  // Use types for the UnsafeProjection otherwise we need to have CPU BoundAttributeReferences
  // used for converting between columnar data and rows (to get the first row in each batch).
  private lazy val unsafeProjection = UnsafeProjection.create(sorter.projectedBatchTypes)
  // Used for converting between rows and columns when we have to put a cuttoff on the GPU
  // to know how much of the data after a merge sort is fully sorted.
  private lazy val projectedSchema = TrampolineUtil.fromAttributes(sorter.projectedBatchSchema)
  private lazy val converters = new GpuRowToColumnConverter(projectedSchema)

  // This is used in "splitTableAfterSort".
  // We need to figure out how to split up the data into reasonable batches. We could try and do
  // something really complicated and figure out how much data get per batch, but in practice
  // we really only expect to see one or two batches worth of data come in, so lets optimize
  // for that case and set the targetBatchSize to always be 1/8th the targetSize.
  private val pendingBatchSize = targetSize >> 3 // => targetSize/8

  private var realBatchSize = math.max(targetSize, 1)

  /**
   * Convert the boundaries (first rows for each batch) into unsafe rows for use later on.
   */
  private def convertBoundaries(tab: Table): Array[UnsafeRow] = {
    import scala.collection.JavaConverters._
    val cb = withResource(new NvtxRange("COPY BOUNDARIES", NvtxColor.PURPLE)) { _ =>
      val hostCols = GpuColumnVector.extractColumns(tab, sorter.projectedBatchTypes).toSeq
        .safeMap(_.copyToHost())
      new ColumnarBatch(hostCols.toArray, tab.getRowCount.toInt)
    }
    withResource(cb) { cb =>
      withResource(new NvtxRange("TO UNSAFE ROW", NvtxColor.RED)) { _ =>
        cb.rowIterator().asScala.map(unsafeProjection(_).copy()).toArray
      }
    }
  }

  private final def splitAfterSort(cb: ColumnarBatch,
      sortedOffset: Int = -1): (Option[ColumnarBatch], Seq[OutOfCoreBatch]) = {
    withResource(GpuColumnVector.from(cb)) { table =>
      splitTableAfterSort(table, sortedOffset)
    }
  }

  /**
   * A rather complex function. It will take a sorted table (either the output of a regular sort or
   * a merge sort), split it up, and place the split portions into the proper queues. If
   * sortedOffset >= 0 then everything below that offset is considered to be fully sorted and is
   * returned as an option of "SpillableColumnarBatch". Everything else is spilt into smaller
   * batches as determined by this function and returned as a seq of "OutOfCoreBatch".
   * Call `saveSplitResult` to place them into the cache correspondingly.
   */
  private final def splitTableAfterSort(sortedTbl: Table,
      sortedOffset: Int): (Option[ColumnarBatch], Seq[OutOfCoreBatch]) = {
    val rows = sortedTbl.getRowCount.toInt
    val memSize = GpuColumnVector.getTotalDeviceMemoryUsed(sortedTbl)
    val averageRowSize = memSize.toDouble/rows
    // Protect ourselves from large rows when there are small targetSizes
    val targetRowCount = Math.max((pendingBatchSize/averageRowSize).toInt, 1024)

    var sortedCb: Option[ColumnarBatch] = None
    val pendingObs: ArrayBuffer[OutOfCoreBatch] = ArrayBuffer.empty
    if (sortedOffset == rows) {
      // The entire thing is sorted
      sortedCb = Some(GpuColumnVector.from(sortedTbl, sorter.projectedBatchTypes))
    } else {
      val hasFullySortedData = sortedOffset > 0
      val splitIndexes = if (hasFullySortedData) {
        sortedOffset until rows by targetRowCount
      } else {
        targetRowCount until rows by targetRowCount
      }
      // Get back the first row so we can sort the batches
      val lowerGatherIndexes = if (hasFullySortedData) {
        // The first batch is sorted so don't gather a row for it
        splitIndexes
      } else {
        Seq(0) ++ splitIndexes
      }

      val lowerBoundaries =
        withResource(new NvtxRange("lower boundaries", NvtxColor.ORANGE)) { _ =>
          withResource(ColumnVector.fromInts(lowerGatherIndexes: _*)) { gatherMap =>
            withResource(sortedTbl.gather(gatherMap)) { boundariesTab =>
              convertBoundaries(boundariesTab)
            }
          }
        }

      withResource(sortedTbl.contiguousSplit(splitIndexes: _*)) { splits =>
        var currentSplit = 0
        val stillPending = if (hasFullySortedData) {
          val ct = splits(currentSplit)
          splits(currentSplit) = null
          val cb = withResource(ct) { _ =>
            GpuColumnVectorFromBuffer.from(ct, sorter.projectedBatchTypes)
          }
          currentSplit += 1
          sortedCb = Some(cb)
          splits.slice(1, splits.length)
        } else {
          splits
        }

        closeOnExcept(sortedCb) { _ =>
          assert(lowerBoundaries.length == stillPending.length)
          closeOnExcept(pendingObs) { _ =>
            stillPending.zip(lowerBoundaries).foreach {
              case (ct: ContiguousTable, lower: UnsafeRow) =>
                splits(currentSplit) = null
                currentSplit += 1
                if (ct.getRowCount > 0) {
                  val sp = SpillableColumnarBatch(ct, sorter.projectedBatchTypes,
                    SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
                  pendingObs += OutOfCoreBatch(sp, lower)
                } else {
                  ct.close()
                }
            }
          }
        }
      }
    }
    (sortedCb, pendingObs.toSeq)
  }

  /** Save the splitting result returned from `splitAfterSort` into the cache */
  private final def saveSplitResult(
      result: (Option[ColumnarBatch], Seq[OutOfCoreBatch])): Unit = {
    val (sortedSp, pendingObs) = result
    closeOnExcept(pendingObs) { _ =>
      sortedSp.foreach { cb =>
        val sp = SpillableColumnarBatch(cb, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
        sortedSize += sp.sizeInBytes
        sorted.add(sp)
      }
    }
    pendingObs.foreach(pending.add)
  }

  /**
   * Take a single sorted batch from the `alreadySortedIter`, split it up and store them for
   * merging.
   */
  private final def splitOneSortedBatch(scb: SpillableColumnarBatch): Unit = {
    withResource(new NvtxWithMetrics("split input batch", NvtxColor.CYAN, opTime)) { _ =>
      val retIter = withRetry(scb, splitSpillableInHalfByRows) { attempt =>
        onFirstPassSplit()
        withResource(attempt.getColumnarBatch()) { cb =>
          splitAfterSort(cb)
        }
      }
      while(retIter.hasNext) {
        saveSplitResult(retIter.next())
      }
    }
  }

  /**
   * First pass through the data. Conceptually we are going to read in all of the batches, that are
   * already sorted and split them up into smaller chunks for later merge sorting. But we are
   * only going to do that if we have more than one batch to sort.
   */
  private final def firstPassReadBatches(scb: SpillableColumnarBatch): Unit = {
    splitOneSortedBatch(scb)
    while (alreadySortedIter.hasNext) {
      splitOneSortedBatch(alreadySortedIter.next())
    }
  }

  private def splitTargetSizeInHalf(
      name: String)(inputSize: AutoCloseableTargetSize): Seq[AutoCloseableTargetSize] = {
    val halfSize = splitTargetSizeInHalfGpu(inputSize)
    assert(halfSize.length == 1)
    // cache the size for the following operations
    logInfo(s"$name: change the target batch size from $realBatchSize to" +
      s" ${halfSize.head.targetSize}")
    realBatchSize = halfSize.head.targetSize
    halfSize
  }

  /**
   * Merge sort enough data that we can output a batch and put it in the sorted queue.
   * @return if we can determine that we can return a final batch without putting it in the sorted
   *         queue then optionally return it.
   */
  private final def mergeSortEnoughToOutput(): Option[ColumnarBatch] = {
    logDebug("start a mergeSortEnoughToOutput ...")
    val splitFunc = splitTargetSizeInHalf("merge-sort")(_)
    // Now get enough sorted data to return
    while (!pending.isEmpty && sortedSize < realBatchSize) {
      // Keep going until we have enough data to return
      val pendingSort = new ArrayBuffer[OutOfCoreBatch]()
      val retryResult = closeOnExcept(pendingSort.toSeq) { _ =>
        val initBatchSize = AutoCloseableTargetSize(realBatchSize, 1)
        withRetry(initBatchSize, splitFunc) { attemptSize =>
          // Retry steps:
          //   1) Restore the remaining batches in pendingSort to pending.
          //   2) Collect batches for merge sort according to attemptSize.
          //   3) Execute merge sort and split the result accordingly.
          // 1)
          pendingSort.indices.reverse.foreach(i => pending.add(pendingSort.remove(i)))
          // 2)
          var remainingSize = attemptSize.targetSize
          while (!pending.isEmpty && (pendingSort.isEmpty ||
              remainingSize - pending.peek().buffer.sizeInBytes >= 0)) {
            pendingSort.append(pending.poll())
            remainingSize -= pendingSort.last.buffer.sizeInBytes
          }
          // 3)
          val mergedBatch = sorter.mergeSort(pendingSort.map(_.buffer).toSeq, sortTime)
          withResource(mergedBatch) { _ =>
            val sortSplitOffset = if (pending.isEmpty) {
              // No need to split it
              mergedBatch.numRows()
            } else {
              // The data is only fully sorted if there is nothing pending that is smaller
              // than it so get the next "smallest" row that is pending.
              val cutoffCb = converters.convertBatch(Array(pending.peek().firstRow),
                projectedSchema)
              val offsetCol = withResource(cutoffCb)(sorter.upperBound(mergedBatch, _))
              withResource(withResource(offsetCol)(_.copyToHost())) { offsetHostCol =>
                assert(offsetHostCol.getRowCount == 1)
                offsetHostCol.getInt(0)
              }
            }

            val mergedSize = GpuColumnVector.getTotalDeviceMemoryUsed(mergedBatch)
            if (sortSplitOffset == mergedBatch.numRows() && sorted.isEmpty &&
                (mergedSize >= targetSize || pending.isEmpty)) {
              // This is a special case where we have everything we need to output already,
              // so why bother with another contiguous split just to put it into the queue.
              withResource(GpuColumnVector.from(mergedBatch)) { mergedTbl =>
                closeOnExcept(sorter.removeProjectedColumns(mergedTbl)) { ret =>
                  // Get the finally batch, ok to close the input for this merge-sort
                  pendingSort.safeClose()
                  pendingSort.clear()
                  Left(ret)
                }
              }
            } else {
              onMergeSortSplit() // This is for test only
              val splitRet@(batch, pendingCbs) = splitAfterSort(mergedBatch, sortSplitOffset)
              closeOnExcept(batch.toSeq ++ pendingCbs) { _ =>
                // Split done, ok to close the input for this merge-sort
                pendingSort.safeClose()
                pendingSort.clear()
              }
              Right(splitRet)
            }
          }
        }.next()
      }
      retryResult match {
        case Left(retBatch) => return Some(retBatch)
        case Right(splitResult) => saveSplitResult(splitResult)
      }
    }
    None
  }

  /**
   * Take data from the sorted queue and return a final batch that can be returned
   * @return a batch that can be returned.
   */
  private final def concatOutput(): ColumnarBatch = {
    // combine all the sorted data into a single batch
    val splitFunc = splitTargetSizeInHalf("concat-output")(_)
    val initBatchSize = AutoCloseableTargetSize(realBatchSize, 1)
    withRetry(initBatchSize, splitFunc) { attemptSize =>
      var idx = 0
      var remainingSize = attemptSize.targetSize
      val spillCbs = ArrayBuffer[SpillableColumnarBatch]()
      while (idx < sorted.size() &&
          (remainingSize - sorted.get(idx).sizeInBytes >= 0 || spillCbs.isEmpty)) {
        spillCbs += sorted.get(idx)
        remainingSize -= spillCbs.last.sizeInBytes
        idx += 1
      }
      sortedSize -= math.max(attemptSize.targetSize - remainingSize, 0)
      onConcatOutput() // this call is for test only
      val concated = if (spillCbs.length == 1) { // We cannot concat a single table
        withResource(spillCbs.head.getColumnarBatch())(GpuColumnVector.from)
      } else {
        val tables = spillCbs.safeMap { sp =>
          withResource(sp.getColumnarBatch())(GpuColumnVector.from)
        }
        withResource(tables)(_ => Table.concatenate(tables: _*))
      }
      withResource(concated) { _ =>
        // ok to remove the input for this concatenation.
        spillCbs.indices.foreach(_ => sorted.removeFirst())
        spillCbs.safeClose()
        sorter.removeProjectedColumns(concated)
      }
    }.next()
  }

  override def next(): ColumnarBatch = {
    if (sorter.projectedBatchSchema.isEmpty) {
      // special case, no columns just rows
      withRetryNoSplit(alreadySortedIter.next()) { scb =>
        // This should have no columns so no need to remove anything from the projected data
        scb.getColumnarBatch()
      }
    } else {
      if (pending.isEmpty && sorted.isEmpty) {
        closeOnExcept(alreadySortedIter.next()) { scb =>
          if (!alreadySortedIter.hasNext) {
            sorted.add(scb)
          } else {
            firstPassReadBatches(scb)
          }
        }
      }
      withResource(new NvtxWithMetrics("Sort next output batch", NvtxColor.CYAN, opTime)) { _ =>
        val ret = mergeSortEnoughToOutput().getOrElse(concatOutput())

        outputBatches += 1
        outputRows += ret.numRows()
        // We already read in all of the data so calling hasNext is cheap
        if (hasNext) {
          ret
        } else {
          GpuColumnVector.tagAsFinalBatch(ret)
        }
      }
    }
  }

  override def close(): Unit = {
    sorted.forEach(_.close())
    pending.close()
  }

  /** Callbacks designed for unit tests only. Don't do any heavy things inside. */
  protected def onFirstPassSplit(): Unit = {}
  protected def onMergeSortSplit(): Unit = {}
  protected def onConcatOutput(): Unit = {}
}
