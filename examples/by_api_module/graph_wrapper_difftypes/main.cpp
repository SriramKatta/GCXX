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

/*
 * This file demonstrates the usage of conditional graph nodes with
 * a series of *simple* example graphs.
 *
 * For more information on conditional nodes, see the programming guide:
 *
 *   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#conditional-graph-nodes
 *
 */

// System includes
#include <cassert>
#include <cstdio>
#include <utility>

#include <gcxx/api.hpp>

__global__ void ifGraphKernelA(char* dPtr,
                               gcxx::deviceGraphConditionalHandle_t handle) {
  // In this example, condition is set if *dPtr is odd
  unsigned int value = *dPtr & 0x01;
  gcxx::Graph::SetConditional(handle, value);
  printf("GPU: Handle set to %d\n", value);
}

// This kernel will only be executed if the condition is true
__global__ void ifGraphKernelC(void) {
  printf("GPU: Hello from the GPU! The condition was true.\n");
}

// Setup and launch the graph
void simpleIfGraph(void) {

  // Allocate a byte of device memory to use as input
  auto dPtr_raii = gcxx::memory::make_device_unique_ptr<char>(1);
  char* dPtr     = dPtr_raii.get();
  // checkCudaErrors(cudaMalloc((void **)&dPtr, 1));

  printf("simpleIfGraph: Building graph...\n");
  // cudaGraphCreate(&graph, 0);
  gcxx::Graph graph;

  // Create conditional handle.
  // cudaGraphConditionalHandleCreate(&condHandle, graph);
  auto condHandle = graph.CreateConditionalHandle(0);

  // Use a kernel upstream of the conditional to set the handle value
  // cudaGraphNodeParams params = {cudaGraphNodeTypeKernel};
  // params.kernel.func         = (void*)ifGraphKernelA;
  // params.kernel.blockDim.x   = params.kernel.blockDim.y =
  //   params.kernel.blockDim.z = 1;
  // params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z
  // =
  //   1;
  // params.kernel.kernelParams = kernelArgs;
  // kernelArgs[0]              = &dPtr;
  // kernelArgs[1]              = &condHandle;
  // checkCudaErrors(cudaGraphAddNode(&kernelNode, graph, NULL, 0, &params));
  auto kernelparam = gcxx::KernelParamsBuilder()
                       .setKernel(ifGraphKernelA)
                       .setArgs(dPtr, condHandle)
                       .build<2>();
  auto kernelNode = graph.AddKernelNode(kernelparam);

  // cudaGraphNodeParams cParams = {cudaGraphNodeTypeConditional};
  // cParams.conditional.handle  = condHandle;
  // cParams.conditional.type    = cudaGraphCondTypeIf;
  // cParams.conditional.size    = 1;
  // checkCudaErrors(cudaGraphAddNode(&conditionalNode, graph.getRawGraph(),
  //                                  &kernelNode, 1, &cParams));

  auto [conditionalNode, bodyGraph] = graph.AddIfNode(condHandle);

  // gcxx::GraphView bodyGraph = cParams.conditional.phGraph_out[0];

  // Populate the body of the conditional node
  // cudaGraphNode_t bodyNode;
  // params.kernel.func         = (void*)ifGraphKernelC;
  // params.kernel.kernelParams = nullptr;
  // checkCudaErrors(cudaGraphAddNode(&bodyNode, bodyGraph, NULL, 0, &params));

  auto kernel2 =
    gcxx::KernelParamsBuilder().setKernel(ifGraphKernelC).build<0>();
  auto kernelnode1 = bodyGraph.AddKernelNode(kernel2);


  // checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  auto graphExec = graph.Instantiate();

  // Initialize device memory and launch the graph
  // checkCudaErrors(cudaMemset(dPtr, 0, 1));  // Set dPtr to 0
  gcxx::memory::Memset(dPtr, 0, 1);
  printf("Host: Launching graph with device memory set to 0\n");
  graphExec.Launch();
  gcxx::Device::Synchronize();

  // Initialize device memory and launch the graph
  // checkCudaErrors(cudaMemset(dPtr, 1, 1));  // Set dPtr to 1
  gcxx::memory::Memset(dPtr, 1, 1);
  printf("Host: Launching graph with device memory set to 1\n");
  graphExec.Launch();
  gcxx::Device::Synchronize();


  printf("simpleIfGraph: Complete\n\n");
}

/*
 * Create a graph containing a single conditional while node.
 * The default value of the conditional variable is set to true, so this
 * effectively becomes a do-while loop as the conditional body will always
 * execute at least once. The body of the conditional contains 3 kernel nodes:
 * A [ B -> C -> D ]
 * Nodes B and C are just dummy nodes for demonstrative purposes. Node D
 * will decrement a device memory location and set the condition value to false
 * when the value reaches zero, terminating the loop.
 * In this example, stream capture is used to populate the conditional body.
 */

// This kernel will only be executed if the condition is true
__global__ void doWhileEmptyKernel(void) {
  printf("GPU: doWhileEmptyKernel()\n");
  return;
}

__global__ void doWhileLoopKernel(char* dPtr,
                                  gcxx::deviceGraphConditionalHandle_t handle) {
  if (--(*dPtr) == 0) {
    cudaGraphSetConditional(handle, 0);
  }
  printf("GPU: counter = %d\n", *dPtr);
}

void simpleDoWhileGraph(void) {
  // cudaGraph_t graph;
  // cudaGraphExec_t graphExec;
  // cudaGraphNode_t conditionalNode;

  // Allocate a byte of device memory to use as input
  // checkCudaErrors(cudaMalloc((void**)&dPtr, 1));
  auto dPtr_raii = gcxx::memory::make_device_unique_ptr<char>(1);
  char* dPtr     = dPtr_raii.get();

  printf("simpleDoWhileGraph: Building graph...\n");
  // checkCudaErrors(cudaGraphCreate(&graph, 0));
  gcxx::Graph graph;

  auto handle = graph.CreateConditionalHandle(
    1, gcxx::flags::graphConditionalHandle::Default);
  // checkCudaErrors(cudaGraphConditionalHandleCreate(&handle, graph, 1,
  //                                                  cudaGraphCondAssignDefault));

  // cudaGraphNodeParams cParams = {cudaGraphNodeTypeConditional};
  // cParams.conditional.handle  = handle;
  // cParams.conditional.type    = cudaGraphCondTypeWhile;
  // cParams.conditional.size    = 1;
  // checkCudaErrors(cudaGraphAddNode(&conditionalNode, graph, NULL, 0,
  // &cParams));

  auto [conditionalNode, bodyGraph] = graph.AddWhileNode(handle);

  // cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  gcxx::Stream captureStream;
  // checkCudaErrors(cudaStreamCreate(&captureStream));

  // checkCudaErrors(cudaStreamBeginCaptureToGraph(captureStream, bodyGraph,
  //                                               nullptr, nullptr, 0,
  //                                               cudaStreamCaptureModeGlobal));
  captureStream.BeginCaptureToGraph(bodyGraph,
                                    gcxx::flags::streamCaptureMode::Global);
  // doWhileEmptyKernel<<<1, 1, 0, captureStream>>>();
  // doWhileEmptyKernel<<<1, 1, 0, captureStream>>>();
  // doWhileLoopKernel<<<1, 1, 0, captureStream>>>(dPtr, handle);
  gcxx::launch::Kernel(captureStream, {1}, {1}, 0, doWhileEmptyKernel);
  gcxx::launch::Kernel(captureStream, {1}, {1}, 0, doWhileEmptyKernel);
  gcxx::launch::Kernel(captureStream, {1}, {1}, 0, doWhileLoopKernel, dPtr,
                       handle);

  // checkCudaErrors(cudaStreamEndCapture(captureStream, nullptr));
  captureStream.EndCaptureToGraph(bodyGraph);
  // checkCudaErrors(cudaStreamDestroy(captureStream));

  // checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  auto graphExec = graph.Instantiate();

  // Initialize device memory and launch the graph
  // checkCudaErrors(cudaMemset(dPtr, 10, 1));  // Set dPtr to 10
  gcxx::memory::Memset(dPtr, 10, 1);
  printf("Host: Launching graph with loop counter set to 10\n");
  // checkCudaErrors(cudaGraphLaunch(graphExec, 0));
  // checkCudaErrors(cudaDeviceSynchronize());
  graphExec.Launch();
  gcxx::Device::Synchronize();

  // Cleanup
  // checkCudaErrors(cudaGraphExecDestroy(graphExec));
  // checkCudaErrors(cudaGraphDestroy(graph));
  // checkCudaErrors(cudaFree(dPtr));

  printf("simpleDoWhileGraph: Complete\n\n");
}

/*
 * Create a graph containing a conditional while loop using stream capture.
 * This demonstrates how to insert a conditional node into a stream which is
 * being captured. The graph consists of a kernel node, A, followed by a
 * conditional while node, B, followed by a kernel node, D. The conditional
 * body is populated by a single kernel node, C:
 *
 * A -> B [ C ] -> D
 *
 * The same kernel will be used for both nodes A and C. This kernel will test
 * a device memory location and set the condition when the location is non-zero.
 * We must run the kernel before the loop as well as inside the loop in order
 * to behave like a while loop as opposed to a do-while loop. We need to
 * evaluate the device memory location before the conditional node is evaluated
 * in order to set the condition variable properly. Because we're using a kernel
 * upstream of the conditional node, there is no need to use the handle default
 * value to initialize the conditional value.
 */

__global__ void capturedWhileKernel(
  char* dPtr, gcxx::deviceGraphConditionalHandle_t handle) {
  printf("GPU: counter = %d\n", *dPtr);
  if (*dPtr) {
    (*dPtr)--;
  }
  cudaGraphSetConditional(handle, *dPtr);
}

__global__ void capturedWhileEmptyKernel(void) {
  printf("GPU: capturedWhileEmptyKernel()\n");
  return;
}

void capturedWhileGraph(void) {
  gcxx::deviceGraphConditionalHandle_t handle;
  // cudaGraph_t graph;
  // cudaGraphExec_t graphExec;

  // Allocate a byte of device memory to use as input
  // checkCudaErrors(cudaMalloc((void **)&dPtr, 1));
  auto dPtr_raii = gcxx::memory::make_device_unique_ptr<char>(1);
  char* dPtr     = dPtr_raii.get();

  printf("capturedWhileGraph: Building graph...\n");
  // cudaStream_t captureStream;
  // checkCudaErrors(cudaStreamCreate(&captureStream));
  gcxx::Stream captureStream;

  // checkCudaErrors(cudaStreamBeginCapture(captureStream,
  // cudaStreamCaptureModeGlobal));
  captureStream.BeginCapture(gcxx::flags::streamCaptureMode::Global);

  // Obtain the handle of the graph
  // checkCudaErrors(cudaStreamGetCaptureInfo(captureStream, &status, NULL,
  // &graph.getRawGraph(), &dependencies, &numDependencies));
  // const deviceGraphNode_t* pDependencies;
  {
    auto [status, uniqueID, graph, dependencies, numDependencies] =
      captureStream.GetCaptureInfo();
    [[maybe_unused]] auto _ = uniqueID;  // Suppress unused warning

    // Create the conditional handle
    // checkCudaErrors(cudaGraphConditionalHandleCreate(&handle, graph));
    handle = graph.CreateConditionalHandle(
      0, gcxx::flags::graphConditionalHandle::Default);
    capturedWhileKernel<<<1, 1, 0, captureStream>>>(dPtr, handle);
  }

  // Insert kernel node A

  // Obtain the handle for node A (get updated dependencies after kernel launch)
  auto captureInfo2    = captureStream.GetCaptureInfo();
  auto dependencies    = captureInfo2.pDependencies;
  auto numDependencies = captureInfo2.pDependenciescount;

  // Insert conditional node B
  auto [conditionalNode, bodyGraph] =
    captureInfo2.graph.AddWhileNode(handle, dependencies, numDependencies);

  // checkCudaErrors(cudaStreamUpdateCaptureDependencies(
  // captureStream, &conditionalNode, 1, cudaStreamSetCaptureDependencies));
  captureStream.UpdateCaptureDependencies(
    gcxx::flags::StreamUpdateCaptureDependencies::Set, &conditionalNode, 1);

  // Insert kernel node D
  capturedWhileEmptyKernel<<<1, 1, 0, captureStream>>>();

  auto graph = captureStream.EndCapture();

  // Populate conditional body graph using stream capture
  gcxx::Stream bodyStream;

  bodyStream.BeginCaptureToGraph(bodyGraph,
                                 gcxx::flags::streamCaptureMode::Global);

  // Insert kernel node C
  capturedWhileKernel<<<1, 1, 0, bodyStream>>>(dPtr, handle);

  bodyStream.EndCaptureToGraph(bodyGraph);

  auto graphExec = graph.Instantiate();

  gcxx::memory::Memset(dPtr, 0, 1);
  printf("Host: Launching graph with loop counter set to 0\n");
  graphExec.Launch();
  gcxx::Device::Synchronize();

  int n = 6;
  gcxx::memory::Memset(dPtr, n, 1);
  printf("Host: Launching graph with loop counter set to %d\n", n);
  graphExec.Launch();
  gcxx::Device::Synchronize();

  printf("capturedWhileGraph: Complete\n\n");
}

/*
 * Create a graph containing two nodes.
 * The first node, A, is a kernel and the second node, B, is a conditional IF
 * node containing two graphs. The first graph within the conditional will be
 * executed when the condition is true, while the second graph will be executed
 * when the conditional is false. The kernel sets the condition variable to true
 * if a device memory location contains an odd number. Otherwise the condition
 * variable is set to false. There is a single kernel(C & D) within each
 * conditional body which prints a message.
 *
 * A -> B [ C | D ]
 *
 * This example requires CUDA >= 12.8.
 */

// This kernel will only be executed if the condition is false
__global__ void ifGraphKernelD(void) {
  printf("GPU: Hello from the GPU! The condition was false.\n");
}

// Setup and launch the graph
void simpleIfElseGraph(void) {
  gcxx::Graph graph;

  auto dptr_raii = gcxx::memory::make_device_unique_ptr<char>(1);
  char* dPtr     = dptr_raii.get();

  printf("simpleIfElseGraph: Building graph...\n");

  // Create conditional handle.
  auto handle = graph.CreateConditionalHandle(0);

  // Use a kernel upstream of the conditional to set the handle value
  auto kernparam = gcxx::KernelParamsBuilder()
                     .setKernel(ifGraphKernelA)
                     .setArgs(dPtr, handle)
                     .build<2>();
  auto kernnode = graph.AddKernelNode(kernparam);

  auto [ifelsenode, IfGraphBody, Elsegraphbody] =
    graph.AddIfElseNode(handle, &kernnode, 1);

  // Populate the body of the first graph in the conditional node, executed if
  auto kern2 = gcxx::KernelParamsBuilder().setKernel(ifGraphKernelC).build<0>();
  auto truenode = IfGraphBody.AddKernelNode(kern2);

  auto falsekern =
    gcxx::KernelParamsBuilder().setKernel(ifGraphKernelD).build<0>();
  auto falsenode = Elsegraphbody.AddKernelNode(falsekern);

  auto graphExec = graph.Instantiate();

  // // Initialize device memory and launch the graph
  gcxx::memory::Memset(dPtr, 0, 1);
  printf("Host: Launching graph with loop counter set to 0\n");
  graphExec.Launch();
  gcxx::Device::Synchronize();

  int n = 1;
  gcxx::memory::Memset(dPtr, n, 1);
  printf("Host: Launching graph with loop counter set to %d\n", n);
  graphExec.Launch();
  gcxx::Device::Synchronize();


  printf("simpleIfElseGraph: Complete\n\n");
}

/*
 * Create a graph containing two nodes.
 * The first node, A, is a kernel and the second node, B, is a conditional
 * SWITCH node containing four graphs. The nth graph within the conditional will
 * be executed when the condition is n. If conditional >= n, no graph will be
 * executed. Kernel A sets the condition variable to the value stored in a
 * device memory location. This device location is updated from the host with
 * each launch to demonstrate the behavior. There is a single kernel(nodes C, D,
 * E and F) within each conditional body which prints a message.
 *
 * A -> B [ C | D | E | F ]
 *
 * This example requires CUDA >= 12.8.
 */

__global__ void switchGraphKernelA(char* dPtr,
                                   cudaGraphConditionalHandle handle) {
  unsigned int value = *dPtr;
  cudaGraphSetConditional(handle, value);
  printf("GPU: Handle set to %d\n", value);
}

__global__ void switchGraphKernelC(void) {
  printf("GPU: Hello from switchGraphKernelC(), running on the GPU!\n");
}

__global__ void switchGraphKernelD(void) {
  printf("GPU: Hello from switchGraphKernelD(), running on the GPU!\n");
}

__global__ void switchGraphKernelE(void) {
  printf("GPU: Hello from switchGraphKernelE(), running on the GPU!\n");
}

__global__ void switchGraphKernelF(void) {
  printf("GPU: Hello from switchGraphKernelF(), running on the GPU!\n");
}

// Setup and launch the graph
void simpleSwitchGraph(void) {
  gcxx::Graph graph;

  auto dptr_raii = gcxx::memory::make_device_unique_ptr<char>(1);
  char* dPtr     = dptr_raii.get();

  printf("simpleSwitchGraph: Building graph...\n");

  auto handle = graph.CreateConditionalHandle(
    0, gcxx::flags::graphConditionalHandle::Default);

  // Use a kernel upstream of the conditional to set the handle value
  auto kern1 = gcxx::KernelParamsBuilder()
                 .setKernel(switchGraphKernelA)
                 .setArgs(dPtr, handle)
                 .build<2>();
  auto kernelNode = graph.AddKernelNode(kern1);

  auto [condNode, casevector] = graph.AddSwitchNode(handle, 4);

  // Populate the four graph bodies within the SWITCH conditional graph
  auto kernswitchC =
    gcxx::KernelParamsBuilder().setKernel(switchGraphKernelC).build<0>();
  std::ignore = casevector[0].AddKernelNode(kernswitchC);

  auto kernswitchD =
    gcxx::KernelParamsBuilder().setKernel(switchGraphKernelD).build<0>();
  std::ignore = casevector[1].AddKernelNode(kernswitchD);

  auto kernswitchE =
    gcxx::KernelParamsBuilder().setKernel(switchGraphKernelE).build<0>();
  std::ignore = casevector[2].AddKernelNode(kernswitchE);

  auto kernswitchF =
    gcxx::KernelParamsBuilder().setKernel(switchGraphKernelF).build<0>();
  std::ignore = casevector[3].AddKernelNode(kernswitchF);

  auto graphExec = graph.Instantiate();

  for (char i = 0; i < 5; i++) {
    gcxx::memory::Memset(dPtr, i, 1);
    printf("Host: Launching graph with device memory set to %d\n", i);
    graphExec.Launch();
    gcxx::Device::Synchronize();
  }

  printf("simpleSwitchGraph: Complete\n\n");
}

int main(int argc, char** argv) {
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

  return 0;
}
