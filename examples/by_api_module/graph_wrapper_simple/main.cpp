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

__global__ void kern_X() {
  printf("printing from %s\n", __func__);
}

__global__ void kern_Y() {
  printf("printing from %s\n", __func__);
}

#define GRAPH 1

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
  eve_after_E.RecordInStream(stream1);

  stream3.WaitOnEvent(eve_after_A);
  gcxx::launch::Kernel(stream3, {1}, {1}, 0, kern_X);
  gcxx::launch::Kernel(stream3, {1}, {1}, 0, kern_Y);
  eve_after_Y.RecordInStream(stream3);

  stream1.WaitOnEvent(eve_after_E);
  stream1.WaitOnEvent(eve_after_Y);

  auto gp = stream1.EndCapture();
  gp.SaveDotfile("./test.dot", gcxx::flags::graphDebugDot::Verbose);
  auto exec = gp.Instantiate();
  exec.Launch(StreamforGraph);

}

int main(int argc, char const* argv[]) {
  stream_capture();
  return 0;
}
