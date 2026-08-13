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
 * This sample demonstrates the use of OpenMP (or pthreads) and streams with
 * Unified Memory on a single GPU. A pool of tasks is partitioned across host
 * threads: small tasks are run on the CPU while large tasks are dispatched to
 * the GPU, with each thread owning its own stream + BLAS handle. Unified Memory
 * attachments (cudaMemAttachHost / cudaMemAttachSingle) are what let the CPU and
 * the per-thread GPU streams safely share the same allocations.
 *
 * GCXX port: managed memory, streams, BLAS handles, device capability checks,
 * the host-side reference gemv, the per-stream memory attachment
 * (gcxx::StreamView::AttachMemAsync) and the device matrix-vector product
 * (gcxx::blas::gemv) are all mapped onto GCXX. Note that stream memory
 * attachment is a CUDA-only concept; on the HIP backend AttachMemAsync is a
 * no-op, so the host/device routing there relies on unified memory being
 * concurrently accessible (each task owns its own buffers, so there is no race).
 */

#include <fmt/format.h>
#include <gcxx/api.hpp>
#include <gcxx/blas_api.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#ifdef USE_PTHREADS
#include <pthread.h>
#else
#include <omp.h>
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
void   srand48(long seed) { srand((unsigned int)seed); }  // NOLINT
double drand48() { return double(rand()) / RAND_MAX; }
#endif

// Column-major matrix / contiguous vector mdspan aliases (double, int index).
template <class T, class IndexT>
using mat_t = gcxx::mdspan<T, gcxx::dextents<IndexT, 2>, gcxx::layout_left,
                           gcxx::default_accessor<T>>;
template <class T, class IndexT>
using vec_t = gcxx::mdspan<T, gcxx::dextents<IndexT, 1>, gcxx::layout_left,
                           gcxx::default_accessor<T>>;

template <typename T>
struct Task {
  unsigned int size{};
  unsigned int id{};
  gcxx::device_managed_ptr<T> data{};
  gcxx::device_managed_ptr<T> result{};
  gcxx::device_managed_ptr<T> vector{};

  Task() = default;

  // Allocates unified-memory buffers for an size x size matrix and two
  // length-size vectors, then fills them with random data. Mirrors the
  // cudaMallocManaged allocations in the upstream sample.
  void allocate(unsigned int s, unsigned int unique_id) {
    id   = unique_id;
    size = s;
    data   = gcxx::make_device_managed_unique_ptr<T>(
      static_cast<std::size_t>(size) * size);
    result = gcxx::make_device_managed_unique_ptr<T>(size);
    vector = gcxx::make_device_managed_unique_ptr<T>(size);
    gcxx::Device::Synchronize();

    T* d = data.get();
    for (unsigned int i = 0; i < size * size; ++i) {
      d[i] = static_cast<T>(drand48());
    }

    T* r = result.get();
    T* v = vector.get();
    for (unsigned int i = 0; i < size; ++i) {
      r[i] = static_cast<T>(0);
      v[i] = static_cast<T>(drand48());
    }
  }
};

#ifdef USE_PTHREADS
struct threadData_t {
  int                                  tid;
  gcxx::DeviceHandle*                  devhand;
  std::vector<Task<double>>*           taskListPtr;
  std::vector<gcxx::Stream>*           streams;
  std::vector<gcxx::blas::BlasHandle>* handles;
  int                                  taskStart;
  int                                  taskSize;
};
using threadData = threadData_t;
#endif

// Host-side reference matrix-vector product (kept verbatim from the upstream
// sample, including its unused m/alpha parameters). Operates directly on the
// host-accessible unified-memory pointers.
template <typename T>
void gemv(int m, int n, T alpha, T* A, T* x, T beta, T* result) {
  (void)m;
  (void)alpha;
  for (int i = 0; i < n; i++) {
    result[i] *= beta;
    for (int j = 0; j < n; j++) {
      result[i] += A[i * n + j] * x[j];
    }
  }
}

// Runs a single task either on the host (small) or on the device (large).
// `dev_stream` is this thread's GPU stream, `host_stream` is the shared stream
// used to stage host-side access (streams[0] in the upstream sample).
template <typename T>
void run_task(Task<T>& t, gcxx::blas::BlasHandleView handle,
              gcxx::StreamView dev_stream, gcxx::StreamView host_stream,
              int tid) {
  if (t.size < 100) {
    fmt::print("Task [{:>3d}], thread [{:>3d}] executing on host   ({})\n", t.id, tid,
               t.size);

    // Attach the buffers to the host so the CPU may touch them while the device
    // runs other tasks concurrently, then wait for the attachment to take hold.
    host_stream.AttachMemAsync(gcxx::span<T>(t.data.get(),
                                  static_cast<std::size_t>(t.size) * t.size),
                               gcxx::flags::memAttach::Host);
    host_stream.AttachMemAsync(gcxx::span<T>(t.vector.get(), t.size),
                               gcxx::flags::memAttach::Host);
    host_stream.AttachMemAsync(gcxx::span<T>(t.result.get(), t.size),
                               gcxx::flags::memAttach::Host);
    host_stream.Synchronize();
    gemv(static_cast<int>(t.size), static_cast<int>(t.size), T{1},
         t.data.get(), t.vector.get(), T{0}, t.result.get());
  } else {
    fmt::print("Task [{:>3d}], thread [{:>3d}] executing on device ({})\n", t.id, tid,
               t.size);

    handle.setStream(dev_stream);

    // Attach each buffer to this stream exclusively so the device may access it
    // only from here, then dispatch the matrix-vector product via gcxx::blas.
    dev_stream.AttachMemAsync(gcxx::span<T>(t.data.get(),
                                 static_cast<std::size_t>(t.size) * t.size),
                              gcxx::flags::memAttach::Single);
    dev_stream.AttachMemAsync(gcxx::span<T>(t.vector.get(), t.size),
                              gcxx::flags::memAttach::Single);
    dev_stream.AttachMemAsync(gcxx::span<T>(t.result.get(), t.size),
                              gcxx::flags::memAttach::Single);

    const int       n = static_cast<int>(t.size);
    mat_t<T, int>   A(t.data.get(), n, n);
    vec_t<T, int>   x(t.vector.get(), n);
    vec_t<T, int>   y(t.result.get(), n);
    gcxx::blas::gemv(handle, T{1}, A, x, T{0}, y);
  }
}

#ifdef USE_PTHREADS
void* execute(void* inpArgs) {
  auto* dataPtr = static_cast<threadData*>(inpArgs);
  dataPtr->devhand->makeCurrent();

  for (int i = 0; i < dataPtr->taskSize; i++) {
    Task<double>& t = (*dataPtr->taskListPtr)[dataPtr->taskStart + i];
    run_task<double>(t, (*dataPtr->handles)[dataPtr->tid + 1],
                     (*dataPtr->streams)[dataPtr->tid + 1].get(),
                     (*dataPtr->streams)[0].get(), dataPtr->tid);
  }

  pthread_exit(nullptr);
  return nullptr;
}
#endif

template <typename T>
void initialise_tasks(std::vector<Task<T>>& TaskList) {
  for (unsigned int i = 0; i < TaskList.size(); i++) {
    int size = std::max(static_cast<int>(drand48() * 1000.0), 64);
    TaskList[i].allocate(static_cast<unsigned int>(size), i);
  }
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  fmt::print("[{}] - Starting...\n", argv[0]);

  // Pick the default device and report it.
  auto        devhand  = gcxx::Device::get();
  const auto  devProps = devhand.getDeviceProp();
  fmt::print("GPU device [{}]\n", devProps.name);

  if (!devhand.attribute(gcxx::dev_attr::managed_memory)) {
    fmt::print(stderr, "Unified Memory not supported on this device\n");
    return EXIT_SUCCESS;
  }

  // "Prohibited" compute mode is value 2 in both CUDA (cudaComputeModeProhibited)
  // and HIP (hipComputeModeProhibited); GCXX does not yet wrap a named enum.
  constexpr int kComputeModeProhibited = 2;
  if (devhand.attribute(gcxx::dev_attr::compute_mode) == kComputeModeProhibited) {
    fmt::print(
      stderr,
      "This sample requires a device in either default or process exclusive mode\n");
    return EXIT_SUCCESS;
  }

  int seed = static_cast<int>(std::time(nullptr));
  srand48(seed);

  const int nthreads = 4;

  // streams[0] is the shared host-access stream; streams[1..nthreads] are the
  // per-thread device streams, paired with handles[1..nthreads].
  std::vector<gcxx::Stream>           streams(nthreads + 1);
  std::vector<gcxx::blas::BlasHandle> handles(nthreads + 1);

  const unsigned int        N = 40;
  std::vector<Task<double>> TaskList(N);
  initialise_tasks(TaskList);

  fmt::print("Executing tasks on host / device\n");

#ifdef USE_PTHREADS
  std::vector<pthread_t>  threads(nthreads);
  std::vector<threadData> inputs(nthreads);

  // Even partition of tasks across threads; the last thread picks up remainder.
  const int perThread = static_cast<int>(TaskList.size()) / nthreads;
  const int remainder = static_cast<int>(TaskList.size()) % nthreads;
  int       cursor    = 0;
  for (int i = 0; i < nthreads; i++) {
    inputs[i].tid         = i;
    inputs[i].devhand     = &devhand;
    inputs[i].taskListPtr = &TaskList;
    inputs[i].streams     = &streams;
    inputs[i].handles     = &handles;
    inputs[i].taskStart   = cursor;
    inputs[i].taskSize    = perThread + (i == nthreads - 1 ? remainder : 0);
    cursor += inputs[i].taskSize;

    pthread_create(&threads[i], nullptr, &execute, &inputs[i]);
  }
  for (int i = 0; i < nthreads; i++) {
    pthread_join(threads[i], nullptr);
  }
#else
  omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(TaskList.size()); i++) {
    devhand.makeCurrent();
    int tid = omp_get_thread_num();
    run_task<double>(TaskList[i], handles[tid + 1], streams[tid + 1].get(),
                     streams[0].get(), tid);
  }
#endif

  devhand.Synchronize();

  // Streams, handles and tasks all release their device resources via RAII.

  fmt::print("All Done!\n");
  return EXIT_SUCCESS;
}
