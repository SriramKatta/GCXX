#include <fmt/format.h>
#include <gcxx/api.hpp>
#include <vector>

int main() {
  int numdevices = gcxx::Device::count();

  fmt::print("numdevices : {}\n", numdevices);

  std::vector<gcxx::Stream> streams;
  streams.reserve(numdevices);

  for (int i = 0; i < numdevices; ++i) {
    auto devhand = gcxx::Device::set(i);
    streams.emplace_back(gcxx::flags::streamType::nullStream,
                         gcxx::flags::streamPriority::veryLow);
  }

  return 0;
}
