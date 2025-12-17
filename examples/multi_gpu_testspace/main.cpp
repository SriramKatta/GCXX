#include <fmt/format.h>
#include <gcxx/api.hpp>
#include <vector>

int main() {
  int numdevices = gcxx::DeviceHandle::count();

  fmt::print("numdevices : {}\n", numdevices);

  std::vector<gcxx::Stream> streams;
  streams.reserve(numdevices);

  for (int i = 0; i < numdevices; ++i) {
    gcxx::DeviceHandle::set(i);
    streams.emplace_back(gcxx::flags::streamType::nullStream,
                         gcxx::flags::streamPriority::veryLow);
  }

  return 0;
}
