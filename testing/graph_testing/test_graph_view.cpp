// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include "tests_common.hpp"

#include <type_traits>

#include <gcxx/runtime/event/event_view.hpp>
#include <gcxx/runtime/graph/graph.hpp>
#include <gcxx/runtime/graph/graph_view.hpp>

// addNode dispatches on the payload type and returns the matching typed node
// view; dependencies are any GraphNodeView-derived views. All node kinds are
// created through the union-based driver::graphAddNode.

using namespace gcxx;

namespace {

  using Deps = gcxx::span<const gcxx::GraphNodeView>;

  using KPV   = const KernelNodeParamsView&;
  using MPV   = const Memcpy3DParamsView&;
  using MSPV  = const MemsetParamsView&;
  using HPV   = const HostNodeParamsView&;
  using ERPV  = const EventRecordNodeParamsView&;
  using EWPV  = const EventWaitNodeParamsView&;
  using MFPV  = const MemFreeNodeParamsView&;
  using CGPV  = const ChildGraphNodeParamsView&;
  using MAPV  = const MemAllocNodeParamsView&;
  using ESPV  = const ExternalSemaphoreSignalNodeParamsView&;
  using EWP2V = const ExternalSemaphoreWaitNodeParamsView&;

  template <typename Payload>
  using AddNodeResult = decltype(std::declval<GraphView&>().addNode(
    std::declval<Payload>(), std::declval<Deps>()));

}  // namespace

static_assert(std::is_same_v<AddNodeResult<KPV>, KernelNodeView>);
static_assert(std::is_same_v<AddNodeResult<MPV>, MemcpyNodeView>);
static_assert(std::is_same_v<AddNodeResult<MSPV>, MemsetNodeView>);
static_assert(std::is_same_v<AddNodeResult<HPV>, HostNodeView>);
static_assert(std::is_same_v<AddNodeResult<ERPV>, EventRecordNodeView>);
static_assert(std::is_same_v<AddNodeResult<EWPV>, EventWaitNodeView>);
static_assert(std::is_same_v<AddNodeResult<MFPV>, MemFreeNodeView>);
static_assert(std::is_same_v<AddNodeResult<CGPV>, ChildGraphNodeView>);
static_assert(std::is_same_v<AddNodeResult<MAPV>, MemAllocNodeView>);
static_assert(
  std::is_same_v<AddNodeResult<ESPV>, ExternalSemaphoreSignalNodeView>);
static_assert(
  std::is_same_v<AddNodeResult<EWP2V>, ExternalSemaphoreWaitNodeView>);

// Owning params bind to the params-view bases.
static_assert(
  std::is_same_v<AddNodeResult<const KernelNodeParams<2>&>, KernelNodeView>);
static_assert(
  std::is_same_v<AddNodeResult<const MemsetParams&>, MemsetNodeView>);
static_assert(
  std::is_same_v<AddNodeResult<const MemFreeNodeParams&>, MemFreeNodeView>);

// Empty node: no payload, dependencies only.
static_assert(std::is_same_v<decltype(std::declval<GraphView&>().addNode(
                               std::declval<Deps>())),
                             GraphNodeView>);

namespace {

  __global__ void incrementKernel(int* data) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      ++(*data);
    }
  }

}  // namespace

TEST(GraphAddNode, DispatchBuildsAndRunsDiamond) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No device available";
  }

  gcxx::Graph graph;

  auto dPtrRaii = gcxx::make_device_unique_ptr<int>(1);
  int* dPtr     = dPtrRaii.get();

  // Roots: an empty node and a memset node.
  auto emptyNode = graph.addNode();

  auto memsetParams = gcxx::MemsetParamsBuilder()
                        .setPtr(dPtr)
                        .setValue(0)
                        .setElementSize<int>()
                        .setWidth(1)
                        .build();
  auto memsetNode = graph.addNode(memsetParams);

  // Kernel depends on both roots (braced dependency list; the mixed view
  // types slice into GraphNodeView).
  auto kernelParams = gcxx::KernelParamsBuilder()
                        .setKernel(incrementKernel)
                        .setGridDim(1)
                        .setBlockDim(1)
                        .setArgs(dPtr)
                        .build();
  auto kernelNode = graph.addNode(kernelParams, {memsetNode, emptyNode});

  // Record a real event after the kernel, then wait on it.
  auto rawEvent = driver::eventCreateWithFlags(
    static_cast<details_::flag_t>(flags::eventCreate::None));
  gcxx::EventView event{rawEvent};

  auto eventRecordNode =
    graph.addNode(EventRecordNodeParams{event}, {kernelNode});
  auto eventWaitNode =
    graph.addNode(EventWaitNodeParams{event}, {eventRecordNode});

  EXPECT_EQ(graph.getNumNodes(), std::size_t{5});
  EXPECT_EQ(graph.getNumEdges(), std::size_t{4});
  EXPECT_EQ(emptyNode.getType(), flags::graphNodeType::Empty);
  EXPECT_EQ(memsetNode.getType(), flags::graphNodeType::Memset);
  EXPECT_EQ(kernelNode.getType(), flags::graphNodeType::Kernel);
  EXPECT_EQ(eventRecordNode.getType(), flags::graphNodeType::EventRecord);
  EXPECT_EQ(eventWaitNode.getType(), flags::graphNodeType::EventWait);

  auto graphExec = graph.instantiate();
  gcxx::Stream stream;
  graphExec.launch(stream);
  stream.sync();

  int host = 42;
  gcxx::Copy(&host, dPtr, 1);
  EXPECT_EQ(host, 1);  // memset to 0, then the kernel incremented it

  driver::eventDestroy(rawEvent);
}
