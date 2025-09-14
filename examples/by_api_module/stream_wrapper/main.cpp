#include <gpucxx/runtime/stream.hpp>
#include <iostream>

namespace gflags = gcxx::flags;

void eve_ref_check(gcxx::Event& event) {
  if (event.HasOccurred()) {
    std::cout << "Event has occurred." << std::endl;
  } else {
    std::cout << "Event has not occurred." << std::endl;
  }
}

int main() {

  gcxx::Stream str1;
  auto str1default = gcxx::Stream::Create(gflags::streamType::syncWithNull);
  gcxx::Stream str2(gflags::streamType::noSyncWithNull);
  auto str3 = gcxx::Stream::Create(gflags::streamType::noSyncWithNull);
  auto str4 = gcxx::Stream::Create(gflags::streamType::nullStream,
                                   gflags::streamPriority::veryHigh);
  return 0;
}
