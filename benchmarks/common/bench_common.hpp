// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// The three timing modes shared by the GPU micro-benchmarks. All three take
// the per-iteration work as a callable so a raw backend call and a gcxx
// wrapper slot in identically; the callable owns any DoNotOptimize /
// ClobberMemory guards on its results.
#pragma once

#include <chrono>
#include <cstdint>
#include <utility>

#include <benchmark/benchmark.h>

#include <gcxx/runtime/event.hpp>
#include <gcxx/runtime/stream.hpp>

namespace bench {

  // Host-side issue cost: fn() must be enqueue-only (no per-iteration sync);
  // device work is drained once, outside the timed loop.
  template <typename Fn>
  auto issue_only(benchmark::State& state, gcxx::Stream& stream, Fn&& fn)
    -> void {
    for (auto _ : state) {
      fn();
    }
    stream.sync();
  }

  // End-to-end: every iteration completes on the device before the next one
  // is issued; host issue cost and GPU execution are both counted.
  template <typename Fn>
  auto with_sync(benchmark::State& state, gcxx::Stream& stream, Fn&& fn)
    -> void {
    for (auto _ : state) {
      fn();
      stream.sync();
    }
  }

  // Device-side duration via an event pair around fn(); requires
  // ->UseManualTime() on the benchmark registration. Excludes the host issue
  // cost (which with_sync counts) and the sync latency (which both other
  // modes count); roughly issue_only + with_sync - gpu_time == overhead.
  template <typename Fn>
  auto gpu_time(benchmark::State& state, gcxx::Stream& stream, Fn&& fn)
    -> void {
    gcxx::Event start;
    gcxx::Event stop;
    for (auto _ : state) {
      start.recordInStream(stream);
      fn();
      stop.recordInStream(stream);
      stream.sync();
      state.SetIterationTime(
        stop.elapsedTimeSince<std::chrono::duration<double>>(start).count());
    }
  }

  // items/s == FLOP/s when flops_per_iter = 2*m*k*n (dense gemm).
  auto set_flops_counter(benchmark::State& state, double flopsPerIter) -> void {
    state.SetItemsProcessed(static_cast<std::int64_t>(
      flopsPerIter * static_cast<double>(state.iterations())));
  }

}  // namespace bench
