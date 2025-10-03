#include <fmt/format.h>
#include <gcxx/api.hpp>
#include <vector>

int main() {
  int numdevices = 0;
  GCXX_SAFE_RUNTIME_CALL(GetDeviceCount, "Failed to get GPU ID", &numdevices);

  fmt::print("numdevices : {}\n", numdevices);

  std::vector<gcxx::Stream> streams;
  streams.reserve(numdevices);

  for (int i = 0; i < numdevices; ++i) {
    GCXX_SAFE_RUNTIME_CALL(SetDevice, "Failed to set GPU ID", i);
    streams.emplace_back(gcxx::flags::streamType::nullStream,
                         gcxx::flags::streamPriority::veryLow);
  }

  return 0;
}
