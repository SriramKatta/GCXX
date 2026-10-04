// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <cstring>
#include <type_traits>
#include <vector>

#include <gcxx/runtime/event/event_view.hpp>
#include <gcxx/runtime/graph/graph_params.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>

// Params are host-side value types: constructing them, reading fields back
// and running the builders needs no GPU and makes no driver calls.

using namespace gcxx;

// ─── Compile-time contracts ────────────────────────────────────────────────

// Owning params are non-copyable/non-movable: raw pointers inside them
// (kernel args, access descs, semaphore arrays) must not dangle.
static_assert(!std::is_copy_constructible_v<EventRecordNodeParams>);
static_assert(!std::is_move_constructible_v<EventRecordNodeParams>);
static_assert(!std::is_copy_constructible_v<EventWaitNodeParams>);
static_assert(!std::is_move_constructible_v<EventWaitNodeParams>);
static_assert(!std::is_copy_constructible_v<MemFreeNodeParams>);
static_assert(!std::is_move_constructible_v<MemFreeNodeParams>);
static_assert(!std::is_copy_constructible_v<ChildGraphNodeParams>);
static_assert(!std::is_move_constructible_v<ChildGraphNodeParams>);
static_assert(!std::is_copy_constructible_v<MemAllocNodeParams>);
static_assert(!std::is_move_constructible_v<MemAllocNodeParams>);
static_assert(!std::is_copy_constructible_v<ExternalSemaphoreSignalNodeParams>);
static_assert(!std::is_move_constructible_v<ExternalSemaphoreSignalNodeParams>);
static_assert(!std::is_copy_constructible_v<ExternalSemaphoreWaitNodeParams>);
static_assert(!std::is_move_constructible_v<ExternalSemaphoreWaitNodeParams>);

// The fluent builder chain compiles and yields the owning params type.
static_assert(std::is_same_v<decltype(gcxx::EventRecordParamsBuilder()
                                        .setEvent(gcxx::EventView{})
                                        .build()),
                             gcxx::EventRecordNodeParams>);
static_assert(std::is_same_v<decltype(gcxx::EventWaitParamsBuilder()
                                        .setEvent(gcxx::EventView{})
                                        .build()),
                             gcxx::EventWaitNodeParams>);
static_assert(
  std::is_same_v<
    decltype(gcxx::MemAllocParamsBuilder()
               .setPoolProps(MemAllocNodeParamsView::deviceMemPoolProps_t{})
               .setBytesize(std::size_t{8})
               .build()),
    gcxx::MemAllocNodeParams>);
static_assert(
  std::is_same_v<
    decltype(gcxx::ExternalSemaphoreSignalParamsBuilder()
               .setSemaphores(
                 gcxx::span<const ExternalSemaphoreSignalNodeParamsView::
                              deviceExternalSemaphore_t>{})
               .setSignalParams(
                 gcxx::span<const ExternalSemaphoreSignalNodeParamsView::
                              deviceExternalSemaphoreSignalParams_t>{})
               .build()),
    gcxx::ExternalSemaphoreSignalNodeParams>);

// Public _t aliases name each builder's empty state and are the factory
// return types.
static_assert(std::is_same_v<decltype(gcxx::KernelParamsBuilder()),
                             gcxx::KernelParamsBuilder_t>);
static_assert(std::is_same_v<decltype(gcxx::EventRecordParamsBuilder()),
                             gcxx::EventRecordParamsBuilder_t>);
static_assert(std::is_same_v<decltype(gcxx::MemAllocParamsBuilder()),
                             gcxx::MemAllocParamsBuilder_t>);

// ─── Event record/wait ─────────────────────────────────────────────────────

TEST(GraphEventParams, EventRecordRoundTrip) {
  const driver::deviceEvent_t raw{driver::INVALID_EVENT};
  const EventView event{raw};

  const auto params = EventRecordParamsBuilder().setEvent(event).build();

  EXPECT_EQ(params.getRawParams().event, raw);
  EXPECT_EQ(params.getEvent().getRawHandle(), raw);
}

TEST(GraphEventParams, EventWaitRoundTrip) {
  const driver::deviceEvent_t raw{driver::INVALID_EVENT};
  const EventView event{raw};

  const auto params = EventWaitParamsBuilder().setEvent(event).build();

  EXPECT_EQ(params.getRawParams().event, raw);
  EXPECT_EQ(params.getEvent().getRawHandle(), raw);
}

// ─── Mem free ──────────────────────────────────────────────────────────────

TEST(GraphMemFreeParams, RoundTrip) {
  int dummy{};
  void* const dptr = &dummy;

  const auto params = MemFreeParamsBuilder().setDptr(dptr).build();

  EXPECT_EQ(params.getRawParams().dptr, dptr);
  EXPECT_EQ(params.getDptr(), dptr);
}

// ─── Child graph ───────────────────────────────────────────────────────────

TEST(GraphChildGraphParams, RoundTrip) {
  const GraphView graph{};  // default handle = INVALID_GRAPH, no driver call

  const auto params = ChildGraphParamsBuilder().setGraph(graph).build();

  EXPECT_EQ(params.getRawParams().graph, driver::INVALID_GRAPH);
  EXPECT_EQ(params.getGraphHandle(), driver::INVALID_GRAPH);
  EXPECT_EQ(params.getGraph().getRawHandle(), driver::INVALID_GRAPH);
}

TEST(GraphChildGraphParams, DirectConstructorMatchesBuilder) {
  const GraphView graph{};

  const ChildGraphNodeParams direct{graph};
  const auto built = ChildGraphParamsBuilder().setGraph(graph).build();

  EXPECT_EQ(direct.getRawParams().graph, built.getRawParams().graph);
}

// ─── Mem alloc ─────────────────────────────────────────────────────────────

TEST(GraphMemAllocParams, RoundTrip) {
  const MemAllocNodeParamsView::deviceMemPoolProps_t props{};
  const std::vector<MemAllocNodeParamsView::deviceMemAccessDesc_t> descs(2);

  const auto params = MemAllocParamsBuilder()
                        .setPoolProps(props)
                        .setBytesize(std::size_t{1024})
                        .setAccessDescs(descs)
                        .build();

  EXPECT_EQ(params.getBytesize(), std::size_t{1024});
  EXPECT_EQ(params.getAccessDescCount(), std::size_t{2});
  EXPECT_NE(params.getAccessDescs(), nullptr);
  EXPECT_EQ(params.getDptr(), nullptr);  // output-only, driver fills it
  EXPECT_EQ(std::memcmp(&params.getPoolProps(), &props, sizeof(props)), 0);

  // Access descs are copied into the params' own storage.
  EXPECT_EQ(std::memcmp(params.getAccessDescs(), descs.data(),
                        descs.size() * sizeof(descs[0])),
            0);
}

TEST(GraphMemAllocParams, AccessDescsOptional) {
  const MemAllocNodeParamsView::deviceMemPoolProps_t props{};

  const auto params =
    MemAllocParamsBuilder().setPoolProps(props).setBytesize(512).build();

  EXPECT_EQ(params.getAccessDescCount(), std::size_t{0});
  EXPECT_EQ(params.getAccessDescs(), nullptr);
}

// ─── External semaphore signal/wait ────────────────────────────────────────

TEST(GraphExternalSemaphoreParams, SignalRoundTrip) {
  using View = ExternalSemaphoreSignalNodeParamsView;
  const std::vector<View::deviceExternalSemaphore_t> sems(3, nullptr);
  const std::vector<View::deviceExternalSemaphoreSignalParams_t> semParams(3);

  const auto params = ExternalSemaphoreSignalParamsBuilder()
                        .setSemaphores(sems)
                        .setSignalParams(semParams)
                        .build();

  EXPECT_EQ(params.getNumExtSems(), 3U);
  EXPECT_NE(params.getSemaphores(), nullptr);
  for (unsigned int i = 0; i < params.getNumExtSems(); ++i) {
    EXPECT_EQ(params.getSemaphores()[i], sems[i]);
  }
}

TEST(GraphExternalSemaphoreParams, WaitRoundTrip) {
  using View = ExternalSemaphoreWaitNodeParamsView;
  const std::vector<View::deviceExternalSemaphore_t> sems(2, nullptr);
  const std::vector<View::deviceExternalSemaphoreWaitParams_t> semParams(2);

  const auto params = ExternalSemaphoreWaitParamsBuilder()
                        .setSemaphores(sems)
                        .setWaitParams(semParams)
                        .build();

  EXPECT_EQ(params.getNumExtSems(), 2U);
  EXPECT_NE(params.getSemaphores(), nullptr);
  for (unsigned int i = 0; i < params.getNumExtSems(); ++i) {
    EXPECT_EQ(params.getSemaphores()[i], sems[i]);
  }
}
