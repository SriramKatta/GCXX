#include <fmt/chrono.h>  // needed to print the chrono durations
#include <fmt/format.h>
#include <array>
#include <gcxx/runtime/event.hpp>

void eve_ref_check(const gcxx::EventView& event) {
  if (event.HasOccurred()) {
    fmt::print("Event has occurred.\n");
  } else {
    fmt::print("Event has not occurred.\n");
  }
}

int main() {

  gcxx::Event def_event;
  auto start_event = gcxx::Event::Create();
  auto compundFlag = gcxx::flags::eventCreate::disableTiming |
                     gcxx::flags::eventCreate::interprocess;
  gcxx::Event end_event(compundFlag);
  eve_ref_check(end_event);

  // auto res = cudaEventQuery(end_event); // an error because Event is a owning
  // reference and cannot be cast to raw event
  gcxx::EventView end_event_ref = end_event;
  auto res                      = end_event_ref.HasOccurred();

  if (res) {
    fmt::print("Event query successful.\n");
  } else {
    fmt::print("Event not ready.\n");
  }

  gcxx::Event end_event2;

  start_event.RecordInStream();
  end_event2.RecordInStream();

  auto dur = gcxx::Event::ElapsedTimeBetween(start_event, end_event2);

  fmt::print("Elapsed time between events: {}\n", dur);

  return 0;
}
