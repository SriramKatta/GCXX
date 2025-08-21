#include <gpucxx/api.hpp>
#include <iostream>

void eve_ref_check(gcxx::event_ref event) {
  if (event.HasOccurred()) {
    std::cout << "Event has occurred." << std::endl;
  } else {
    std::cout << "Event has not occurred." << std::endl;
  }
}

int main(int argc, char const* argv[]) {

  gcxx::Event def_event;
  auto start_event = gcxx::Event::Create();
  auto compundFlag = gcxx::flags::eventCreate::disableTiming |
                     gcxx::flags::eventCreate::interprocess;
  gcxx::Event end_event(compundFlag);
  eve_ref_check(end_event);

  gcxx::event_ref end_event_ref = end_event;
  // an error because Event is a owning reference and cannot be cast to raw event
  // auto res                      = cudaEventQuery(end_event);
  auto res = cudaEventQuery(end_event_ref);

  if (res == cudaSuccess) {
    std::cout << "Event query successful." << std::endl;
  } else if (res == cudaErrorNotReady) {
    std::cout << "Event not ready." << std::endl;
  } else {
    std::cerr << "Event query failed with error code: " << res << std::endl;
  }

  return 0;
}
