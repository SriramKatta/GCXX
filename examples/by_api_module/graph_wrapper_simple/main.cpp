// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <cstdio>
#include <gcxx/api.hpp>

__global__ void kern_A() {
  printf("printing from %s\n", __func__);
}

__global__ void kern_B() {
  printf("printing from %s\n", __func__);
}

__global__ void kern_C() {
  printf("printing from %s\n", __func__);
}

__global__ void kern_D() {
  printf("printing from %s\n", __func__);
}

__global__ void kern_E() {
  printf("printing from %s\n", __func__);
}

__global__ void kern_F() {
  printf("printing from %s\n", __func__);
}

__global__ void kern_X() {
  printf("printing from %s\n", __func__);
}

__global__ void kern_Y() {
  printf("printing from %s\n", __func__);
}

template <bool with_graph>
void stream_capture() {
  gcxx::Stream stream1;
  gcxx::Stream stream2;
  gcxx::Stream stream3;
  gcxx::Event eve_after_A;
  gcxx::Event eve_after_B;
  gcxx::Event eve_after_D;
  gcxx::Event eve_after_E;
  gcxx::Event eve_after_Y;

  gcxx::Event start, stop;

  gcxx::Stream StreamforGraph;

  if constexpr (with_graph) {
    stream1.beginCapture(gcxx::flags::streamCaptureMode::Global);
  } else {
    start.recordInStream();
  }

  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_A);
  eve_after_A.recordInStream(stream1);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_B);
  eve_after_B.recordInStream(stream1);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_C);


  stream2.waitOnEvent(eve_after_B);
  gcxx::launch::Kernel(stream2, {1}, {1}, 0, kern_D);
  eve_after_D.recordInStream(stream2);

  stream1.waitOnEvent(eve_after_D);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_E);
  // eve_after_E.recordInStream(stream1);

  stream3.waitOnEvent(eve_after_A);
  gcxx::launch::Kernel(stream3, {1}, {1}, 0, kern_X);
  gcxx::launch::Kernel(stream3, {1}, {1}, 0, kern_Y);
  eve_after_Y.recordInStream(stream3);

  // stream1.waitOnEvent(eve_after_E);
  stream1.waitOnEvent(eve_after_Y);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_F);


  if constexpr (with_graph) {
    auto gp = stream1.endCapture();
    gp.saveDotfile("./test_stream_capture.dot",
                   gcxx::flags::graphDebugDot::Verbose);
    auto exec = gp.instantiate();
    start.recordInStream();
    exec.launch(StreamforGraph);
    stop.recordInStream();
  } else {
    stop.recordInStream();
    gcxx::Device::sync();
  }

  auto dur = stop.elapsedTimeSince(start);

  if constexpr (with_graph) {
    fmt::print("in graph mode elapsed time  : {}\n", dur);
  } else {
    fmt::print("non graph mode elapsed time : {}\n", dur);
  }
}

void stream_capture_tograph() {
  gcxx::Stream stream1;
  gcxx::Stream stream2;
  gcxx::Stream stream3;
  gcxx::Event eve_after_A;
  gcxx::Event eve_after_B;
  gcxx::Event eve_after_D;
  gcxx::Event eve_after_E;
  gcxx::Event eve_after_Y;

  gcxx::Stream StreamforGraph;

  gcxx::Graph graph;

  stream1.beginCaptureToGraph(graph, gcxx::flags::streamCaptureMode::Global);


  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_A);
  eve_after_A.recordInStream(stream1);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_B);
  eve_after_B.recordInStream(stream1);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_C);


  stream2.waitOnEvent(eve_after_B);
  gcxx::launch::Kernel(stream2, {1}, {1}, 0, kern_D);
  eve_after_D.recordInStream(stream2);

  stream1.waitOnEvent(eve_after_D);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_E);
  // eve_after_E.recordInStream(stream1);

  stream3.waitOnEvent(eve_after_A);
  gcxx::launch::Kernel(stream3, {1}, {1}, 0, kern_X);
  gcxx::launch::Kernel(stream3, {1}, {1}, 0, kern_Y);
  eve_after_Y.recordInStream(stream3);

  // stream1.waitOnEvent(eve_after_E);
  stream1.waitOnEvent(eve_after_Y);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_F);

  stream1.endCaptureToGraph(graph);
  graph.saveDotfile("./test_stream_capture_to.dot",
                    gcxx::flags::graphDebugDot::KernelNodeParams);
  auto exec = graph.instantiate();
  exec.launch(StreamforGraph);
}

void manual_graph_build() {
  gcxx::Graph graph;

  gcxx::Stream StreamforGraph;

  auto KA = gcxx::KernelParamsBuilder()
              .setKernel(kern_A)
              .setGridDim(1)
              .setBlockDim(1)
              .build();
  auto KB = gcxx::KernelParamsBuilder()
              .setKernel(kern_B)
              .setGridDim(1)
              .setBlockDim(1)
              .build();
  auto KC = gcxx::KernelParamsBuilder()
              .setKernel(kern_C)
              .setGridDim(1)
              .setBlockDim(1)
              .build();
  auto KD = gcxx::KernelParamsBuilder()
              .setKernel(kern_D)
              .setGridDim(1)
              .setBlockDim(1)
              .build();
  auto KE = gcxx::KernelParamsBuilder()
              .setKernel(kern_E)
              .setGridDim(1)
              .setBlockDim(1)
              .build();
  auto KF = gcxx::KernelParamsBuilder()
              .setKernel(kern_F)
              .setGridDim(1)
              .setBlockDim(1)
              .build();
  auto KX = gcxx::KernelParamsBuilder()
              .setKernel(kern_X)
              .setGridDim(1)
              .setBlockDim(1)
              .build();
  auto KY = gcxx::KernelParamsBuilder()
              .setKernel(kern_Y)
              .setGridDim(1)
              .setBlockDim(1)
              .build();

  auto KAnode = graph.addNode(KA);
  auto KBnode = graph.addNode(KB, {KAnode});
  auto KXnode = graph.addNode(KX, {KAnode});
  auto KCnode = graph.addNode(KC, {KBnode});
  auto KDnode = graph.addNode(KD, {KBnode});
  auto KEnode = graph.addNode(KE, {KCnode, KDnode});
  auto KYnode = graph.addNode(KY, {KXnode});
  std::ignore = graph.addNode(KF, {KEnode, KYnode});

  graph.saveDotfile("./test_manual.dot", gcxx::flags::graphDebugDot::Verbose);
  auto exec = graph.instantiate();
  exec.launch(StreamforGraph);
}

int main(int argc, char const* argv[]) {
  stream_capture_tograph();
  stream_capture<false>();
  manual_graph_build();
  stream_capture<true>();
  return 0;
}
