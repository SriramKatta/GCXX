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

int main(int argc, char const* argv[]) {

  // gcxx::Stream str1;
  // gcxx::Stream str1default = gcxx::Stream::Create(gflags::streamType::none);
  // gcxx::Stream str2(gflags::streamType::nonBlocking);
  // gcxx::Stream str3 = gcxx::Stream::Create(gflags::streamType::nonBlocking);
  gcxx::Stream str4 = gcxx::Stream::Create(gflags::streamType::null);
  return 0;
}
