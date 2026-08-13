// SPDX-License-Identifier: BSD-3-Clause AND GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample illustrates the usage of CUDA events for both GPU timing and
 * overlapping CPU and GPU execution.  Events are inserted into a stream
 * of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
 * perform computations while GPU is executing (including DMA memcopies
 * between the host and device).  CPU can query CUDA events to determine
 * whether GPU has completed tasks.
 */

#include <fmt/format.h>
#include <gcxx/api.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>

// CUDA Kernel Device code: in-place increment of every element by inc_value.
__global__ void increment_kernel(gcxx::span<int> g_data, int inc_value) {
  int idx     = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx] + inc_value;
}

// Verify every element of the (host-accessible) buffer equals x.
bool correct_output(const gcxx::span<int>& data, int x) {
  std::size_t i{0};
  for (const auto& val : data) {
    if (val != x) {
      fmt::print("Error! data[{}] = {}, ref = {}\n", i, val, x);
      return false;
    }
    ++i;
  }
  return true;
}

int main(int argc, char* argv[]) {
  (void)argc;
  fmt::print("[{}] - Starting...\n", argv[0]);

  // Pick the default device and report its name.
  auto devhand        = gcxx::Device::get();
  const auto devProps = devhand.getDeviceProp();
  fmt::print("CUDA device [{}]\n", devProps.name);

  constexpr int n     = 16 * 1024 * 1024;
  constexpr int value = 26;

  // Non-blocking stream shared by every pool allocation, copy, and kernel.
  gcxx::Stream str;

  auto device_pool = gcxx::device_default_memory_pool(devhand);
  auto pinned_pool = gcxx::pinned_default_memory_pool();

  // Allocate host (pinned) and device memory.
  gcxx::host_buffer<int> a(str, pinned_pool, n);
  gcxx::device_buffer<int> d_a(str, device_pool, n);

  gcxx::span a_span(a);
  gcxx::span d_a_span(d_a);

  // a is host (pinned) memory -- zero it with a host memset. gcxx::Memset is
  // cudaMemsetAsync and only accepts device pointers, so it would fail here
  // with cudaErrorInvalidValue.
  std::memset(a_span.data(), 0, a_span.size_bytes());
  // Fill the device buffer with 0xFF (matches the upstream cudaMemset(d_a,255)).
  gcxx::Memset(str, d_a_span, 255);

  // Kernel launch configuration.
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(n / threads.x, 1);

  devhand.Synchronize();
  float gpu_time = 0.0f;

  // Asynchronously issue work to the GPU. Measure both the GPU time (between
  // the recorded events) and the CPU time spent issuing the async calls.
  auto cpu_start = std::chrono::steady_clock::now();

  auto start = str.RecordEvent();
  gcxx::Copy(str, d_a_span, a_span);  // H2D
  gcxx::launch::Kernel(str, blocks, threads, 0, &increment_kernel, d_a_span, value);
  gcxx::Copy(str, a_span, d_a_span);  // D2H
  auto stop = str.RecordEvent();

  auto cpu_end = std::chrono::steady_clock::now();

  // Have the CPU do some work while waiting for the GPU stage to finish.
  unsigned long int counter = 0;
  while (!stop.HasOccurred()) {
    counter++;
  }

  gpu_time = stop.ElapsedTimeSince<gcxx::milliSec>(start).count();

  // Print the CPU and GPU times.
  fmt::print("time spent executing by the GPU: {:.2f}\n", gpu_time);
  fmt::print("time spent by CPU in CUDA calls: {:.2f}\n",
             std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count());
  fmt::print("CPU executed {} iterations while waiting for GPU to finish\n", counter);

  // Check the output for correctness.
  bool bFinalResults = correct_output(a_span, value);

  // Resources (events, host/device buffers) are released automatically by RAII.
  return bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE;
}
