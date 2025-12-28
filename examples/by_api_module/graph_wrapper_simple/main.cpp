#include <stdio.h>
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

void stream_capture() {
  gcxx::Stream stream1;
  gcxx::Stream stream2;
  gcxx::Stream stream3;
  gcxx::Event eve_after_A;
  gcxx::Event eve_after_B;
  gcxx::Event eve_after_D;
  gcxx::Event eve_after_E;
  gcxx::Event eve_after_Y;

  gcxx::Stream StreamforGraph;

  stream1.BeginCapture(gcxx::flags::streamCaptureMode::Global);


  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_A);
  eve_after_A.RecordInStream(stream1);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_B);
  eve_after_B.RecordInStream(stream1);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_C);


  stream2.WaitOnEvent(eve_after_B);
  gcxx::launch::Kernel(stream2, {1}, {1}, 0, kern_D);
  eve_after_D.RecordInStream(stream2);

  stream1.WaitOnEvent(eve_after_D);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_E);
  // eve_after_E.RecordInStream(stream1);

  stream3.WaitOnEvent(eve_after_A);
  gcxx::launch::Kernel(stream3, {1}, {1}, 0, kern_X);
  gcxx::launch::Kernel(stream3, {1}, {1}, 0, kern_Y);
  eve_after_Y.RecordInStream(stream3);

  // stream1.WaitOnEvent(eve_after_E);
  stream1.WaitOnEvent(eve_after_Y);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_F);

  auto gp = stream1.EndCapture();
  gp.SaveDotfile("./test_stream_capture.dot",
                 gcxx::flags::graphDebugDot::Verbose);
  auto exec = gp.Instantiate();
  exec.Launch(StreamforGraph);
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

  stream1.BeginCaptureToGraph(graph, gcxx::flags::streamCaptureMode::Global);


  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_A);
  eve_after_A.RecordInStream(stream1);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_B);
  eve_after_B.RecordInStream(stream1);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_C);


  stream2.WaitOnEvent(eve_after_B);
  gcxx::launch::Kernel(stream2, {1}, {1}, 0, kern_D);
  eve_after_D.RecordInStream(stream2);

  stream1.WaitOnEvent(eve_after_D);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_E);
  // eve_after_E.RecordInStream(stream1);

  stream3.WaitOnEvent(eve_after_A);
  gcxx::launch::Kernel(stream3, {1}, {1}, 0, kern_X);
  gcxx::launch::Kernel(stream3, {1}, {1}, 0, kern_Y);
  eve_after_Y.RecordInStream(stream3);

  // stream1.WaitOnEvent(eve_after_E);
  stream1.WaitOnEvent(eve_after_Y);
  gcxx::launch::Kernel(stream1, {1}, {1}, 0, kern_F);

  stream1.EndCaptureToGraph(graph);
  graph.SaveDotfile("./test_stream_capture_to.dot",
                    gcxx::flags::graphDebugDot::Verbose);
  auto exec = graph.Instantiate();
  exec.Launch(StreamforGraph);
}

void manual_graph_build() {
  gcxx::Graph graph;

  gcxx::Stream StreamforGraph;

  std::vector<gcxx::deviceGraphNode_t> deps;
  deps.reserve(2);

  auto KA = gcxx::KernelParamsBuilder().setKernel(kern_A).build<0>();
  auto KB = gcxx::KernelParamsBuilder().setKernel(kern_B).build<0>();
  auto KC = gcxx::KernelParamsBuilder().setKernel(kern_C).build<0>();
  auto KD = gcxx::KernelParamsBuilder().setKernel(kern_D).build<0>();
  auto KE = gcxx::KernelParamsBuilder().setKernel(kern_E).build<0>();
  auto KF = gcxx::KernelParamsBuilder().setKernel(kern_F).build<0>();
  auto KX = gcxx::KernelParamsBuilder().setKernel(kern_X).build<0>();
  auto KY = gcxx::KernelParamsBuilder().setKernel(kern_Y).build<0>();

  auto KAnode = graph.AddKernelNode(NULL, 0, &(KA.getRawParams()));

  deps.push_back(KAnode);
  auto KBnode = graph.AddKernelNode(deps, &(KB.getRawParams()));
  auto KXnode = graph.AddKernelNode(deps, &(KX.getRawParams()));

  deps.clear();
  deps.push_back(KBnode);
  auto KCnode = graph.AddKernelNode(deps, &(KC.getRawParams()));
  auto KDnode = graph.AddKernelNode(deps, &(KD.getRawParams()));

  deps.clear();
  deps.push_back(KCnode);
  deps.push_back(KDnode);
  auto KEnode = graph.AddKernelNode(deps, &(KE.getRawParams()));

  deps.clear();
  deps.push_back(KXnode);
  auto KYnode = graph.AddKernelNode(deps, &(KY.getRawParams()));

  deps.clear();
  deps.push_back(KEnode);
  deps.push_back(KYnode);
  auto KFnode = graph.AddKernelNode(deps, &(KF.getRawParams()));

  graph.SaveDotfile("./test_manual.dot", gcxx::flags::graphDebugDot::Verbose);
  auto exec = graph.Instantiate();
  exec.Launch(StreamforGraph);
}

int main(int argc, char const* argv[]) {
  stream_capture();
  stream_capture_tograph();
  manual_graph_build();
  return 0;
}
