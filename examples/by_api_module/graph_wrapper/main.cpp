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

#include <cmath>
#include <cstdio>
#include <vector>


#include <gcxx/api.hpp>
#include <gcxx/cooperative_groups.hpp>


namespace cg = cooperative_groups;

enum { THREADS_PER_BLOCK = 512, GRAPH_LAUNCH_ITERATIONS = 3 };

using callBackData_t = struct callBackData {
  const char* fn_name;
  double* data;
};

__global__ void reduce(float* inputVec, double* outputVec, size_t inputSize,
                       size_t outputSize) {
  __shared__ double tmp[THREADS_PER_BLOCK];

  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid     = blockIdx.x * blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
    temp_sum += (double)inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  double beta = temp_sum;
  double temp = NAN;

  for (int i = tile32.size() / 2; i > 0; i >>= 1) {
    if (tile32.thread_rank() < i) {
      temp = tmp[cta.thread_rank() + i];
      beta += temp;
      tmp[cta.thread_rank()] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);

  if (cta.thread_rank() == 0 && blockIdx.x < outputSize) {
    beta = 0.0;
    for (int i = 0; i < cta.size(); i += tile32.size()) {
      beta += tmp[i];
    }
    outputVec[blockIdx.x] = beta;
  }
}

__global__ void reduceFinal(double* inputVec, double* result,
                            size_t inputSize) {
  __shared__ double tmp[THREADS_PER_BLOCK];

  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid     = blockIdx.x * blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
    temp_sum += (double)inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  // do reduction in shared mem
  if ((blockDim.x >= 512) && (cta.thread_rank() < 256)) {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 256];
  }

  cg::sync(cta);

  if ((blockDim.x >= 256) && (cta.thread_rank() < 128)) {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 128];
  }

  cg::sync(cta);

  if ((blockDim.x >= 128) && (cta.thread_rank() < 64)) {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 64];
  }

  cg::sync(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockDim.x >= 64)
      temp_sum += tmp[cta.thread_rank() + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      temp_sum += tile32.shfl_down(temp_sum, offset);
    }
  }
  // write result for this block to global mem
  if (cta.thread_rank() == 0)
    result[0] = temp_sum;
}

void init_input(float* a, size_t size) {
  for (size_t i = 0; i < size; i++)
    a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

void GCXXRT_CB myHostNodeCallback(void* data) {
  // Check status of GPU after stream operations are done
  auto* tmp = (callBackData_t*)(data);
  // checkCudaErrors(tmp->status);

  auto* result   = (double*)(tmp->data);
  char* function = (char*)(tmp->fn_name);
  printf("[%s] Host callback final reduced sum = %lf\n", function, *result);
  *result = 0.0;  // reset the result
}

void deviceGraphsManual(float* inputVec_h, float* inputVec_d,
                        double* outputVec_d, double* result_d, size_t inputSize,
                        size_t numOfBlocks) {
  gcxx::Stream streamForGraph;
  gcxx::Graph graph;
  std::vector<gcxx::deviceGraphNode_t> nodeDependencies;
  gcxx::deviceGraphNode_t memcpyNode = nullptr, memsetNode = nullptr;
  double result_h = 0.0;

  auto memcpy3d1 =
    gcxx::Memcpy3DParamsBuilder()
      .setSrcPtr(gcxx::memory::makePitchedPtr<float>(inputVec_h, inputSize,
                                                     inputSize, 1))
      .setDstPtr(gcxx::memory::makePitchedPtr<float>(inputVec_d, inputSize,
                                                     inputSize, 1))
      .setExtent(gcxx::memory::makeExtent<float>(inputSize, 1, 1))
      .build();


  memcpyNode = graph.AddMemcpyNode(memcpy3d1);

  auto memset1 = gcxx::MemsetParamsBuilder()
                   .setPtr(outputVec_d)
                   .setElemetSize(sizeof(float))
                   .setWidth(numOfBlocks * 2)
                   .build();

  memsetNode = graph.AddMemsetNode(memset1);

  nodeDependencies.push_back(memsetNode);
  nodeDependencies.push_back(memcpyNode);

  auto k1build = gcxx::KernelParamsBuilder()
                   .setKernel(reduce)
                   .setBlockDim(numOfBlocks)
                   .setGridDim(THREADS_PER_BLOCK)
                   .setArgs(inputVec_d, outputVec_d, inputSize, numOfBlocks)
                   .build<4>();

  auto kernelNode = graph.AddKernelNode(k1build, nodeDependencies);

  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);

  auto memset2 = gcxx::MemsetParamsBuilder()
                   .setElemetSize(sizeof(float))
                   .setPtr(result_d)
                   .setWidth(2)
                   .build();

  memsetNode = graph.AddMemsetNode(memset2);
  nodeDependencies.push_back(memsetNode);

  auto k2builder = gcxx::KernelParamsBuilder()
                     .setKernel(reduceFinal)
                     .setBlockDim(THREADS_PER_BLOCK)
                     .setArgs(outputVec_d, result_d, numOfBlocks)
                     .build<3>();
  auto k2 = k2builder.getRawParams();

  kernelNode = graph.AddKernelNode(k2builder, nodeDependencies);
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);

  auto memcpy3d2 =
    gcxx::Memcpy3DParamsBuilder()
      .setSrcPtr(gcxx::memory::makePitchedPtr<double>(result_d, 1, 1, 1))
      .setDstPtr(gcxx::memory::makePitchedPtr<double>(&result_h, 1, 1, 1))
      .setExtent(gcxx::memory::makeExtent<double>(1, 1, 1))
      .build();

  memcpyNode = graph.AddMemcpyNode(memcpy3d2, nodeDependencies);
  nodeDependencies.clear();
  nodeDependencies.push_back(memcpyNode);

  callBackData_t hostFnData;
  hostFnData.data    = &result_h;
  hostFnData.fn_name = "deviceGraphsManual";
  // cudaHostNodeParams hostParams = {0};
  // hostParams.fn                                   = myHostNodeCallback;
  // hostParams.userData = &hostFnData;

  auto hostparambuilder = gcxx::HostNodeParamsBuilder()
                            .setHostCallbackFn(myHostNodeCallback)
                            .setUserData(&hostFnData)
                            .build();


  auto hostNode = graph.AddHostNode(hostparambuilder, nodeDependencies);

  size_t numNodes = graph.GetNumNodes();
  printf("\nNum of nodes in the graph created manually = %zu\n", numNodes);

  auto graphExec       = graph.Instantiate();
  auto clonesGraph     = graph.Clone();
  auto clonedGraphExec = graph.Instantiate();

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    graphExec.Launch(streamForGraph);
  }
  streamForGraph.Synchronize();

  printf("Cloned Graph Output.. \n");
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    clonedGraphExec.Launch(streamForGraph);
  }
  streamForGraph.Synchronize();

  // graph.SaveDotfile("./test_manual.dot",
  //                   gcxx::flags::graphDebugDot::EventNodeParams);
}

void deviceGraphsUsingStreamCapture(float* inputVec_h, float* inputVec_d,
                                    double* outputVec_d, double* result_d,
                                    size_t inputSize, size_t numOfBlocks) {
  gcxx::Stream stream1, stream2, stream3, stream4, streamForGraph;
  gcxx::Event forkStreamEvent, memsetEvent1, memsetEvent2;
  double result_h = 0.0;

  stream1.BeginCapture(gcxx::flags::streamCaptureMode::Global);

  forkStreamEvent.RecordInStream(stream1);
  stream2.WaitOnEvent(forkStreamEvent);
  stream3.WaitOnEvent(forkStreamEvent);
  gcxx::memory::Copy(inputVec_d, inputVec_h, inputSize, stream1);

  gcxx::memory::Memset(outputVec_d, 0, numOfBlocks, stream2);
  memsetEvent1.RecordInStream(stream2);

  gcxx::memory::Memset(result_d, 0, 1, stream3);
  memsetEvent2.RecordInStream(stream3);

  stream1.WaitOnEvent(memsetEvent1);

  gcxx::launch::Kernel(stream1, numOfBlocks, THREADS_PER_BLOCK, 0, reduce,
                       inputVec_d, outputVec_d, inputSize, numOfBlocks);

  stream1.WaitOnEvent(memsetEvent2);

  gcxx::launch::Kernel(stream1, 1, THREADS_PER_BLOCK, 0, reduceFinal,
                       outputVec_d, result_d, numOfBlocks);

  gcxx::memory::Copy(&result_h, result_d, 1, stream1);


  callBackData_t hostFnData = {nullptr};
  hostFnData.data           = &result_h;
  hostFnData.fn_name        = "deviceGraphsUsingStreamCapture";

  gcxx::launch::HostFunc(stream1, myHostNodeCallback, &hostFnData);

  auto graph = stream1.EndCapture();


  size_t numNodes = graph.GetNumNodes();
  printf("\nNum of nodes in the graph created using stream capture API = %zu\n",
         numNodes);

  auto graphExec = graph.Instantiate();

  auto clonedGraph     = graph.Clone();
  auto clonedGraphExec = clonedGraph.Instantiate();

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    graphExec.Launch(streamForGraph);
  }
  streamForGraph.Synchronize();

  printf("Cloned Graph Output.. \n");
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    clonedGraphExec.Launch(streamForGraph);
  }

  streamForGraph.Synchronize();
  // graph.SaveDotfile("./test_stream_capture.dot",
  //                   gcxx::flags::graphDebugDot::EventNodeParams);
}

void deviceGraphsUsingStreamCaptureToGraph(float* inputVec_h, float* inputVec_d,
                                           double* outputVec_d,
                                           double* result_d, size_t inputSize,
                                           size_t numOfBlocks) {
  gcxx::Stream stream1, stream2, stream3, stream4, streamForGraph;
  gcxx::Event forkStreamEvent, memsetEvent1, memsetEvent2;
  double result_h = 0.0;
  gcxx::Graph graph;

  stream1.BeginCaptureToGraph(graph, gcxx::flags::streamCaptureMode::Global);

  forkStreamEvent.RecordInStream(stream1);
  stream2.WaitOnEvent(forkStreamEvent);
  stream3.WaitOnEvent(forkStreamEvent);
  gcxx::memory::Copy(inputVec_d, inputVec_h, inputSize, stream1);

  gcxx::memory::Memset(outputVec_d, 0, numOfBlocks, stream2);
  memsetEvent1.RecordInStream(stream2);

  gcxx::memory::Memset(result_d, 0, 1, stream3);
  memsetEvent2.RecordInStream(stream3);

  stream1.WaitOnEvent(memsetEvent1);

  gcxx::launch::Kernel(stream1, numOfBlocks, THREADS_PER_BLOCK, 0, reduce,
                       inputVec_d, outputVec_d, inputSize, numOfBlocks);

  stream1.WaitOnEvent(memsetEvent2);

  gcxx::launch::Kernel(stream1, 1, THREADS_PER_BLOCK, 0, reduceFinal,
                       outputVec_d, result_d, numOfBlocks);

  gcxx::memory::Copy(&result_h, result_d, 1, stream1);


  callBackData_t hostFnData = {nullptr};
  hostFnData.data           = &result_h;
  hostFnData.fn_name        = "deviceGraphsUsingStreamCaptureToGraph";

  gcxx::launch::HostFunc(stream1, myHostNodeCallback, &hostFnData);

  stream1.EndCaptureToGraph(graph);


  size_t numNodes = graph.GetNumNodes();
  printf("\nNum of nodes in the graph created using stream capture API = %zu\n",
         numNodes);

  auto graphExec = graph.Instantiate();

  auto clonedGraph     = graph.Clone();
  auto clonedGraphExec = clonedGraph.Instantiate();

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    graphExec.Launch(streamForGraph);
  }
  streamForGraph.Synchronize();

  printf("Cloned Graph Output.. \n");
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    clonedGraphExec.Launch(streamForGraph);
  }

  streamForGraph.Synchronize();
  // graph.SaveDotfile("./test_stream_capture.dot",
  //                   gcxx::flags::graphDebugDot::EventNodeParams);
}

int main(int argc, char** argv) {
  size_t size      = 1 << 24;  // number of elements to reduce
  size_t maxBlocks = 512;

  // This will pick the best possible CUDA capable device
  // int devID = 0;  // findCudaDevice(argc, (const char**)argv);

  printf("%zu elements\n", size);
  printf("threads per block  = %d\n", THREADS_PER_BLOCK);
  printf("Graph Launch iterations = %d\n", GRAPH_LAUNCH_ITERATIONS);

  auto inVec_h_raii  = gcxx::memory::make_host_pinned_unique_ptr<float>(size);
  auto inVec_d_raii  = gcxx::memory::make_device_unique_ptr<float>(size);
  auto outVec_d_raii = gcxx::memory::make_device_unique_ptr<double>(maxBlocks);
  auto result_d_raii = gcxx::memory::make_device_unique_ptr<double>(1);


  float* inputVec_h   = inVec_h_raii.get();
  float* inputVec_d   = inVec_d_raii.get();
  double* outputVec_d = outVec_d_raii.get();
  double* result_d    = result_d_raii.get();

  init_input(inputVec_h, size);

  deviceGraphsManual(inputVec_h, inputVec_d, outputVec_d, result_d, size,
                     maxBlocks);

  deviceGraphsUsingStreamCapture(inputVec_h, inputVec_d, outputVec_d, result_d,
                                 size, maxBlocks);

  deviceGraphsUsingStreamCaptureToGraph(inputVec_h, inputVec_d, outputVec_d,
                                        result_d, size, maxBlocks);

  return EXIT_SUCCESS;
}