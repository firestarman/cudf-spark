---
layout: page
title: Shuffle Metrics
nav_order: 15
parent: Developer Overview
---
# Shuffle Metrics: SparkRapidsShuffleDiskSavingsEvent

When using MULTITHREADED shuffle mode with `spark.rapids.shuffle.multithreaded.skipMerge=true`, the
NVIDIA cuDF plugin for Apache Spark emits `SparkRapidsShuffleDiskSavingsEvent` to the Spark event
log.  This document explains how to interpret and aggregate these events.

## Event Format

Each executor posts its own event when it cleans up shuffle data. A single shuffle may have
multiple events in the eventlog (one per executor that participated in the shuffle write).

Event format in eventlog (JSON):
```json
{"Event":"com.nvidia.spark.rapids.SparkRapidsShuffleDiskSavingsEvent",
 "shuffleId":0,"bytesFromMemory":7868,"bytesFromDisk":0}
```

## Why Custom Events Instead of Task Metrics

Spark task metrics are committed when a task completes. However, shuffle data lifecycle
extends beyond task completion - buffers may be spilled to disk after a task finishes but
before the shuffle data is read. The final `bytesFromMemory` vs `bytesFromDisk` statistics
can only be determined when shuffle cleanup occurs (after the SQL query completes), at
which point task metrics are no longer updatable.

## Field Descriptions

| Field | Description |
|-------|-------------|
| `shuffleId` | The Spark shuffle ID |
| `bytesFromMemory` | Bytes kept in memory and never written to disk (actual disk I/O savings) |
| `bytesFromDisk` | Bytes spilled to disk due to memory pressure |

The sum of `bytesFromMemory` across all events should approximately match the total
"Shuffle Bytes Written" reported in task metrics.

## Aggregating Events

To get application-wide totals from an eventlog:

```bash
grep "SparkRapidsShuffleDiskSavingsEvent" eventlog | \
  jq -s '{
    totalBytesFromMemory: (map(.bytesFromMemory) | add),
    totalBytesFromDisk: (map(.bytesFromDisk) | add),
    diskSavingsBytes: (map(.bytesFromMemory) | add)
  }'
```

## Timing Considerations

The cleanup mechanism uses a polling model where executors poll the driver every 1 second.
For short-running applications or scripts, the session may exit before executors have a
chance to poll and report their statistics.

To ensure all events are captured, add a short delay before exiting:

```scala
// After your last query completes
Thread.sleep(2000)  // Wait for executor cleanup polling
spark.stop()
```

For long-running applications or interactive sessions (spark-shell, notebooks), this is
typically not an issue as there is enough time between queries for cleanup to complete.

## Async shuffle read

`spark.rapids.sql.asyncRead.shuffle.enabled` defaults to `true`. In Kudo GPU read mode,
the next batch of serialized host input is buffered on a background thread. GPU transfer,
deserialization, and merge remain on the task thread, using the GPU semaphore and existing
OOM retry/split handling. Set the option to `false` at application startup to disable prefetch.

With `spark.rapids.sql.metrics.level=DEBUG`, the SQL UI exposes `read batch time (async)`,
`read batch semaphore time (async)`, and `background job launch time` on shuffle coalescing
operators (with side prefixes on joins). The launch metric measures submission-to-start delay
of consumed background jobs. A nonzero value confirms background work; a zero value can mean
that the partition fit in one batch and no prefetch was needed. Read time is measured on the
consumer thread; in Kudo GPU mode it also includes GPU merge. Semaphore time measures the
portion spent holding the GPU semaphore, not the time waiting to acquire it.

The background thread name starts with `async GPU shuffle buffer thread for `. Use a profiler
to verify overlap with GPU merge or downstream kernels; timing metrics alone do not prove
overlap or a performance improvement.
