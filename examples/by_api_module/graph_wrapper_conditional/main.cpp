// SPDX-License-Identifier: GPL-3.0-or-later AND BSD-3-Clause
// Copyright (C) 2026 Sriram Katta
//
// Portions of this file are derived from NVIDIA CUDA sample code
// (BSD-3-Clause, original notice preserved below). Modifications
// and additions are licensed under GPL-3.0-or-later.
/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

// Demonstrates conditional graph nodes with a series of simple example
// graphs; see the CUDA programming guide on conditional graph nodes.

// System includes
#include <cassert>
#include <cstdio>
#include <utility>
#include <vector>

#include <gcxx/api.hpp>

__global__ void ifGraphKernelA(
  char* dPtr, gcxx::GraphView::deviceGraphConditionalHandle_t handle) {
  // In this example, condition is set if *dPtr is odd
  unsigned int value = *dPtr & 0x01;
  gcxx::Graph::setConditional(handle, value);
  printf("GPU: Handle set to %d\n", value);
}

// This kernel will only be executed if the condition is true.
__global__ void ifGraphKernelC() {
  printf("GPU: Hello from the GPU! The condition was true.\n");
}

void simpleIfGraph() {

  // Allocate a byte of device memory to use as input.
  auto dPtr_raii = gcxx::make_device_unique_ptr<char>(1);
  char* dPtr     = dPtr_raii.get();

  printf("simpleIfGraph: Building graph...\n");
  gcxx::Graph graph;

  // Create conditional handle.
  auto condHandle = graph.createConditionalHandle(0);

  auto kernelparam = gcxx::KernelParamsBuilder()
                       .setKernel(ifGraphKernelA)
                       .setGridDim(1)
                       .setBlockDim(1)
                       .setArgs(dPtr, condHandle)
                       .build();
  auto kernelNode = graph.addNode(kernelparam);

  auto [conditionalNode, bodyGraph] = graph.addIfNode(condHandle, {kernelNode});


  auto kernel2 = gcxx::KernelParamsBuilder()
                   .setKernel(ifGraphKernelC)
                   .setGridDim(1)
                   .setBlockDim(1)
                   .build();
  auto kernelnode1 = bodyGraph.addNode(kernel2);


  auto graphExec = graph.Instantiate();

  // Initialize device memory and launch the graph
  gcxx::Memset(dPtr, 0, 1);
  printf("Host: Launching graph with device memory set to 0\n");
  graphExec.Launch();
  gcxx::Device::Synchronize();

  // Initialize device memory and launch the graph
  gcxx::Memset(dPtr, 1, 1);
  printf("Host: Launching graph with device memory set to 1\n");
  graphExec.Launch();
  gcxx::Device::Synchronize();


  printf("simpleIfGraph: Complete\n\n");
}

// Do-while while-node (default-true cond); body filled via stream capture.
__global__ void doWhileEmptyKernel() {
  printf("GPU: doWhileEmptyKernel()\n");
  return;
}

__global__ void doWhileLoopKernel(
  char* dPtr, gcxx::GraphView::deviceGraphConditionalHandle_t handle) {
  if (--(*dPtr) == 0) {
    gcxx::Graph::setConditional(handle, 0);
  }
  printf("GPU: counter = %d\n", *dPtr);
}

void simpleDoWhileGraph() {
  auto dPtr_raii = gcxx::make_device_unique_ptr<char>(1);
  char* dPtr     = dPtr_raii.get();

  printf("simpleDoWhileGraph: Building graph...\n");
  gcxx::Graph graph;

  auto handle = graph.createConditionalHandle(
    1, gcxx::flags::graphConditionalHandle::Default);

  auto [conditionalNode, bodyGraph] = graph.addWhileNode(handle);

  gcxx::Stream captureStream;

  captureStream.beginCaptureToGraph(bodyGraph,
                                    gcxx::flags::streamCaptureMode::Global);
  gcxx::launch::Kernel(captureStream, {1}, {1}, 0, doWhileEmptyKernel);
  gcxx::launch::Kernel(captureStream, {1}, {1}, 0, doWhileEmptyKernel);
  gcxx::launch::Kernel(captureStream, {1}, {1}, 0, doWhileLoopKernel, dPtr,
                       handle);

  captureStream.endCaptureToGraph(bodyGraph);
  auto graphExec = graph.Instantiate();

  // Initialize device memory and launch the graph
  gcxx::Memset(dPtr, 10, 1);
  printf("Host: Launching graph with loop counter set to 10\n");
  graphExec.Launch();
  gcxx::Device::Synchronize();

  printf("simpleDoWhileGraph: Complete\n\n");
}

// While node via capture: pre-loop kernel A sets cond, body C, then D.
__global__ void capturedWhileKernel(
  char* dPtr, gcxx::GraphView::deviceGraphConditionalHandle_t handle) {
  printf("GPU: counter = %d\n", *dPtr);
  if (*dPtr) {
    (*dPtr)--;
  }
  gcxx::Graph::setConditional(handle, *dPtr);
}

__global__ void capturedWhileEmptyKernel() {
  printf("GPU: capturedWhileEmptyKernel()\n");
  return;
}

void capturedWhileGraph() {
  gcxx::GraphView::deviceGraphConditionalHandle_t handle = 0;

  auto dPtr_raii = gcxx::make_device_unique_ptr<char>(1);
  char* dPtr     = dPtr_raii.get();

  printf("capturedWhileGraph: Building graph...\n");
  gcxx::Stream captureStream;

  captureStream.beginCapture(gcxx::flags::streamCaptureMode::Global);

  {
    auto [status, uniqueID, graph, dependencies, numDependencies] =
      captureStream.getCaptureInfo();
    [[maybe_unused]] auto _ = uniqueID;  // Suppress unused warning

    handle = graph.createConditionalHandle(
      0, gcxx::flags::graphConditionalHandle::Default);
    gcxx::launch::Kernel(captureStream, {1}, {1}, 0, capturedWhileKernel, dPtr,
                         handle);
  }

  // Insert kernel node A

  // Obtain the handle for node A (get updated dependencies after launch).
  auto captureInfo2 = captureStream.getCaptureInfo();

  // Insert conditional node B; wrap the capture's raw dependency handles as
  // node views.
  std::vector<gcxx::GraphNodeView> dependencies(
    captureInfo2.pDependencies,
    captureInfo2.pDependencies + captureInfo2.pDependenciescount);
  auto [conditionalNode, bodyGraph] =
    captureInfo2.graph.addWhileNode(handle, dependencies);

  captureStream.updateCaptureDependencies(
    gcxx::flags::StreamUpdateCaptureDependencies::Set, &conditionalNode, 1);

  // Insert kernel node D
  gcxx::launch::Kernel(captureStream, {1}, {1}, 0, capturedWhileEmptyKernel);

  auto graph = captureStream.endCapture();

  // Populate conditional body graph using stream capture
  gcxx::Stream bodyStream;

  bodyStream.beginCaptureToGraph(bodyGraph,
                                 gcxx::flags::streamCaptureMode::Global);

  // Insert kernel node C
  gcxx::launch::Kernel(bodyStream, {1}, {1}, 0, capturedWhileKernel, dPtr,
                       handle);


  bodyStream.endCaptureToGraph(bodyGraph);

  auto graphExec = graph.Instantiate();

  gcxx::Memset(dPtr, 0, 1);
  printf("Host: Launching graph with loop counter set to 0\n");
  graphExec.Launch();
  gcxx::Device::Synchronize();

  int n = 6;
  gcxx::Memset(dPtr, n, 1);
  printf("Host: Launching graph with loop counter set to %d\n", n);
  graphExec.Launch();
  gcxx::Device::Synchronize();

  printf("capturedWhileGraph: Complete\n\n");
}

// If/else conditional node: two body graphs, true and false; CUDA >= 12.8.
__global__ void ifGraphKernelD() {
  printf("GPU: Hello from the GPU! The condition was false.\n");
}

void simpleIfElseGraph() {
  gcxx::Graph graph;

  auto dptr_raii = gcxx::make_device_unique_ptr<char>(1);
  char* dPtr     = dptr_raii.get();

  printf("simpleIfElseGraph: Building graph...\n");

  // Create conditional handle.
  auto handle = graph.createConditionalHandle(0);

  // Use a kernel upstream of the conditional to set the handle value.
  auto kernparam = gcxx::KernelParamsBuilder()
                     .setKernel(ifGraphKernelA)
                     .setGridDim(1)
                     .setBlockDim(1)
                     .setArgs(dPtr, handle)
                     .build();
  auto kernNode = graph.addNode(kernparam);

  auto [ifelsenode, IfGraphBody, Elsegraphbody] =
    graph.addIfElseNode(handle, {kernNode});

  // Populate the if-branch body (executed when the condition is true).
  auto kern2 = gcxx::KernelParamsBuilder()
                 .setKernel(ifGraphKernelC)
                 .setGridDim(1)
                 .setBlockDim(1)
                 .build();
  auto truenode = IfGraphBody.addNode(kern2);

  auto falsekern = gcxx::KernelParamsBuilder()
                     .setKernel(ifGraphKernelD)
                     .setGridDim(1)
                     .setBlockDim(1)
                     .build();
  auto falsenode = Elsegraphbody.addNode(falsekern);

  auto graphExec = graph.Instantiate();

  // // Initialize device memory and launch the graph
  gcxx::Memset(dPtr, 0, 1);
  printf("Host: Launching graph with loop counter set to 0\n");
  graphExec.Launch();
  gcxx::Device::Synchronize();

  int n = 1;
  gcxx::Memset(dPtr, n, 1);
  printf("Host: Launching graph with loop counter set to %d\n", n);
  graphExec.Launch();
  gcxx::Device::Synchronize();


  printf("simpleIfElseGraph: Complete\n\n");
}

// Switch conditional node with four case bodies; requires CUDA >= 12.8.
__global__ void switchGraphKernelA(
  char* dPtr, gcxx::GraphView::deviceGraphConditionalHandle_t handle) {
  unsigned int value = *dPtr;
  gcxx::Graph::setConditional(handle, value);
  printf("GPU: Handle set to %d\n", value);
}

__global__ void switchGraphKernelC() {
  printf("GPU: Hello from switchGraphKernelC(), running on the GPU!\n");
}

__global__ void switchGraphKernelD() {
  printf("GPU: Hello from switchGraphKernelD(), running on the GPU!\n");
}

__global__ void switchGraphKernelE() {
  printf("GPU: Hello from switchGraphKernelE(), running on the GPU!\n");
}

__global__ void switchGraphKernelF() {
  printf("GPU: Hello from switchGraphKernelF(), running on the GPU!\n");
}

void simpleSwitchGraph() {
  gcxx::Graph graph;

  auto dptr_raii = gcxx::make_device_unique_ptr<char>(1);
  char* dPtr     = dptr_raii.get();

  printf("simpleSwitchGraph: Building graph...\n");

  auto handle = graph.createConditionalHandle(
    0, gcxx::flags::graphConditionalHandle::Default);

  // Use a kernel upstream of the conditional to set the handle value.
  auto kern1 = gcxx::KernelParamsBuilder()
                 .setKernel(switchGraphKernelA)
                 .setGridDim(1)
                 .setBlockDim(1)
                 .setArgs(dPtr, handle)
                 .build();
  auto kernelNode = graph.addNode(kern1);

  auto [condNode, casevector] = graph.addSwitchNode(handle, 4);

  // Populate the four graph bodies within the SWITCH conditional graph.
  auto kernswitchC = gcxx::KernelParamsBuilder()
                       .setKernel(switchGraphKernelC)
                       .setGridDim(1)
                       .setBlockDim(1)
                       .build();
  std::ignore = casevector[0].addNode(kernswitchC);

  auto kernswitchD = gcxx::KernelParamsBuilder()
                       .setKernel(switchGraphKernelD)
                       .setGridDim(1)
                       .setBlockDim(1)
                       .build();
  std::ignore = casevector[1].addNode(kernswitchD);

  auto kernswitchE = gcxx::KernelParamsBuilder()
                       .setKernel(switchGraphKernelE)
                       .setGridDim(1)
                       .setBlockDim(1)
                       .build();
  std::ignore = casevector[2].addNode(kernswitchE);

  auto kernswitchF = gcxx::KernelParamsBuilder()
                       .setKernel(switchGraphKernelF)
                       .setGridDim(1)
                       .setBlockDim(1)
                       .build();
  std::ignore = casevector[3].addNode(kernswitchF);

  auto graphExec = graph.Instantiate();

  for (char i = 0; i < 5; i++) {
    gcxx::Memset(dPtr, i, 1);
    printf("Host: Launching graph with device memory set to %d\n", i);
    graphExec.Launch();
    gcxx::Device::Synchronize();
  }

  printf("simpleSwitchGraph: Complete\n\n");
}

int main(int argc, char** argv) {
#if GCXX_CUDA_MODE()
  int driverVersion = 0;

  cudaDriverGetVersion(&driverVersion);
  printf("Driver version is: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);

  if (driverVersion < 12030) {
    printf(
      "Skipping execution as driver does not support Graph Conditional "
      "Nodes\n");
    return 0;
  }

  simpleIfGraph();
  simpleDoWhileGraph();
  capturedWhileGraph();

  if (driverVersion < 12080) {
    printf(
      "Skipping execution as driver does not support if/else and switch type "
      "Graph Conditional Nodes\n");
    return 0;
  }

  simpleIfElseGraph();
  simpleSwitchGraph();
#else
#warning "Hip doesn't have conditional graph support"
#endif

  return 0;
}
