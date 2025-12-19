#include <stdio.h>
#include <gcxx/api.hpp>

__global__ void kern_A() {
  printf("Print for %s\n", __func__);
}

__global__ void kern_B() {
  printf("Print for %s\n", __func__);
}

__global__ void kern_C() {
  printf("Print for %s\n", __func__);
}

__global__ void kern_D() {
  printf("Print for %s\n", __func__);
}

__global__ void kern_F() {
  printf("Print for %s\n", __func__);
}

int main() {

  gcxx::Stream str1, str2;
  str1.BeginCapture(gcxx::flags::streamCaptureMode::global);
  kern_A<<<1, 1, 0, str1>>>();
  auto eve_after_A = str1.RecordEvent(gcxx::flags::eventCreate::disableTiming);
  kern_B<<<1, 1, 0, str1>>>();
  str2.WaitOnEvent(eve_after_A);
  kern_C<<<1, 1, 0, str2>>>();
  auto eve_after_C = str2.RecordEvent(gcxx::flags::eventCreate::disableTiming);
  kern_D<<<1, 1, 0, str1>>>();
  str1.WaitOnEvent(eve_after_C);
  kern_F<<<1, 1, 0, str1>>>();
  gcxx::Graph gp = str1.EndCapture();

  gp.SaveDotfile("./test_comp.dot",
                 gcxx::flags::graphDebugDot::Handles |
                   gcxx::flags::graphDebugDot::HostNodeParams);

  auto exec = gp.Instantiate();
  exec.Launch(str1);
  str1.Synchronize();
  return 0;
}
