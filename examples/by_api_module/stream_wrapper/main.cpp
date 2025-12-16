#include <gcxx/runtime/stream.hpp>
#include <iostream>

int main() {
  namespace gflags = gcxx::flags;
  gcxx::Stream str1;
  gcxx::Stream str2(gflags::streamType::noSyncWithNull);
  auto str1default = gcxx::Stream::Create(gflags::streamType::syncWithNull);
  auto str3        = gcxx::Stream::Create(gflags::streamType::noSyncWithNull);
  auto str4        = gcxx::Stream::Create(gflags::streamType::nullStream,
                                          gflags::streamPriority::veryHigh);

  std::cout << "size of stream obj " << sizeof(str1) << std::endl;
  std::cout << "size of stream obj " << sizeof(str1.get()) << std::endl;
  return 0;
}
