// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_EVENT_NODE_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_EVENT_NODE_PARAMS_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/type_traits.hpp>
#include <gcxx/runtime/event/event_view.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

// Type-state builders: setEvent() is required and may be called only once;
// build() refuses to compile otherwise.

GCXX_NAMESPACE_MAIN_BEGIN()

class EventRecordNodeParamsView {
 public:
  using deviceEventRecordNodeParams_t = driver::deviceEventRecordNodeParams_t;

  GCXX_FHC auto getRawParams() const -> const deviceEventRecordNodeParams_t& {
    return m_params;
  }

  GCXX_FHC auto getEvent() const -> EventView {
    return EventView{m_params.event};
  }

 protected:
  GCXX_FH EventRecordNodeParamsView() = default;

  GCXX_FHC explicit EventRecordNodeParamsView(
    deviceEventRecordNodeParams_t params)
      : m_params{params} {}

  // Initialized once via the constructor; never mutated afterwards.
  const deviceEventRecordNodeParams_t m_params{};  // NOLINT
};

// Adds no state over the View; kept only for uniformity with the params
// kinds that do carry storage (kernel, mem-alloc, external-semaphore).
class EventRecordNodeParams : public EventRecordNodeParamsView {
 public:
  GCXX_FHC EventRecordNodeParams() = default;

  GCXX_FHC explicit EventRecordNodeParams(const EventView& event)
      : EventRecordNodeParamsView{make_raw_params(event)} {}

  // Non-copyable/non-movable so pointers into m_params cannot dangle.
  EventRecordNodeParams(const EventRecordNodeParams&) = delete;
  EventRecordNodeParams(EventRecordNodeParams&&)      = delete;

  auto operator=(const EventRecordNodeParams&) -> EventRecordNodeParams& =
                                                    delete;
  auto operator=(EventRecordNodeParams&&) -> EventRecordNodeParams& = delete;

  ~EventRecordNodeParams() = default;

 private:
  GCXX_FHC static auto make_raw_params(const EventView& event)
    -> deviceEventRecordNodeParams_t {
    deviceEventRecordNodeParams_t p{};
    p.event = event.getRawHandle();
    return p;
  }
};

class EventWaitNodeParamsView {
 public:
  using deviceEventWaitNodeParams_t = driver::deviceEventWaitNodeParams_t;

  GCXX_FHC auto getRawParams() const -> const deviceEventWaitNodeParams_t& {
    return m_params;
  }

  GCXX_FHC auto getEvent() const -> EventView {
    return EventView{m_params.event};
  }

 protected:
  GCXX_FH EventWaitNodeParamsView() = default;

  GCXX_FHC explicit EventWaitNodeParamsView(deviceEventWaitNodeParams_t params)
      : m_params{params} {}

  // Initialized once via the constructor; never mutated afterwards.
  const deviceEventWaitNodeParams_t m_params{};  // NOLINT
};

// Adds no state over the View; kept only for uniformity with the params
// kinds that do carry storage (kernel, mem-alloc, external-semaphore).
class EventWaitNodeParams : public EventWaitNodeParamsView {
 public:
  GCXX_FHC EventWaitNodeParams() = default;

  GCXX_FHC explicit EventWaitNodeParams(const EventView& event)
      : EventWaitNodeParamsView{make_raw_params(event)} {}

  // Non-copyable/non-movable so pointers into m_params cannot dangle.
  EventWaitNodeParams(const EventWaitNodeParams&) = delete;
  EventWaitNodeParams(EventWaitNodeParams&&)      = delete;

  auto operator=(const EventWaitNodeParams&) -> EventWaitNodeParams& = delete;
  auto operator=(EventWaitNodeParams&&) -> EventWaitNodeParams&      = delete;

  ~EventWaitNodeParams() = default;

 private:
  GCXX_FHC static auto make_raw_params(const EventView& event)
    -> deviceEventWaitNodeParams_t {
    deviceEventWaitNodeParams_t p{};
    p.event = event.getRawHandle();
    return p;
  }
};

GCXX_NAMESPACE_DETAILS_BEGIN()

namespace event_record_node_builder {

  struct event_tag {};

}  // namespace event_record_node_builder

namespace ernb = event_record_node_builder;

template <typename... Set>
class EventRecordParamsBuilder {
 public:
  EventRecordParamsBuilder() = default;

  GCXX_FHC auto setEvent(const EventView& event) const
    -> EventRecordParamsBuilder<Set..., ernb::event_tag> {
    static_assert(!details_::contains_v<ernb::event_tag, Set...>,
                  "setEvent() may only be called once");
    EventRecordParamsBuilder<Set..., ernb::event_tag> next = *this;
    next.m_event                                           = event;
    return next;
  }

  GCXX_FHC auto build() const -> gcxx::EventRecordNodeParams {
    static_assert(details_::contains_v<ernb::event_tag, Set...>,
                  "setEvent() required before build()");
    return gcxx::EventRecordNodeParams{m_event};
  }

 private:
  template <typename...>
  friend class EventRecordParamsBuilder;  // states construct each other

  // State hand-off between builder states.
  template <typename... Other>
  EventRecordParamsBuilder(const EventRecordParamsBuilder<Other...>& other)
      : m_event{other.m_event} {}

  EventView m_event{};
};

namespace event_wait_node_builder {

  struct event_tag {};

}  // namespace event_wait_node_builder

namespace ewnb = event_wait_node_builder;

template <typename... Set>
class EventWaitParamsBuilder {
 public:
  EventWaitParamsBuilder() = default;

  GCXX_FHC auto setEvent(const EventView& event) const
    -> EventWaitParamsBuilder<Set..., ewnb::event_tag> {
    static_assert(!details_::contains_v<ewnb::event_tag, Set...>,
                  "setEvent() may only be called once");
    EventWaitParamsBuilder<Set..., ewnb::event_tag> next = *this;
    next.m_event                                         = event;
    return next;
  }

  GCXX_FHC auto build() const -> gcxx::EventWaitNodeParams {
    static_assert(details_::contains_v<ewnb::event_tag, Set...>,
                  "setEvent() required before build()");
    return gcxx::EventWaitNodeParams{m_event};
  }

 private:
  template <typename...>
  friend class EventWaitParamsBuilder;  // states construct each other

  // State hand-off between builder states.
  template <typename... Other>
  EventWaitParamsBuilder(const EventWaitParamsBuilder<Other...>& other)
      : m_event{other.m_event} {}

  EventView m_event{};
};

GCXX_NAMESPACE_DETAILS_END()

using EventRecordParamsBuilder_t = details_::EventRecordParamsBuilder<>;

GCXX_FH auto EventRecordParamsBuilder() -> EventRecordParamsBuilder_t {
  return {};
}

using EventWaitParamsBuilder_t = details_::EventWaitParamsBuilder<>;

GCXX_FH auto EventWaitParamsBuilder() -> EventWaitParamsBuilder_t {
  return {};
}

GCXX_NAMESPACE_MAIN_END()


#endif
