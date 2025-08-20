#include <gpucxx/api.hpp>

int main(int argc, char const* argv[]) {

  auto start_event = gcxx::Event::Create();
  auto compundFlag = gcxx::flags::eventCreate::disableTiming |
                     gcxx::flags::eventCreate::interprocess;
  gcxx::Event end_event(compundFlag);

  return 0;
}
