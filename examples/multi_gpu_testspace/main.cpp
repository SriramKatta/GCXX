#include <gpucxx/api.hpp>


int main(int argc, char const* argv[]) {
  int numdevices = 0;
  GPUCXX_SAFE_RUNTIME_CALL(GetDeviceCount, (&numdevices));
  
  GPUCXX_SAFE_RUNTIME_CALL(SetDevice, (0));
  auto str1 = gcxx::Stream::Create();
  GPUCXX_SAFE_RUNTIME_CALL(SetDevice, (1));
  auto str2 = gcxx::Stream::Create();
  
  return 0;
}