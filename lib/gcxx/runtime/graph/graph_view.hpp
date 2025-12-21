#pragma once
#ifndef GCXX_RUNTIME_GRAPH_GRAPH_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_GRAPH_VIEW_HPP_

#include <cstddef>
#include <string_view>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/flags/graph_flags.hpp>
#include <gcxx/runtime/memory/span/span.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
using deviceGraph_t     = GCXX_RUNTIME_BACKEND(Graph_t);
using deviceGraphNode_t = GCXX_RUNTIME_BACKEND(GraphNode_t);
using deviceEvent_t     = GCXX_RUNTIME_BACKEND(Event_t);
using deviceMemcpyKind  = GCXX_RUNTIME_BACKEND(MemcpyKind);

inline constexpr deviceGraph_t INVALID_GRAPH{nullptr};
inline constexpr deviceGraphNode_t INVALID_GRAPH_NODE{nullptr};

// Type aliases for CUDA graph node parameter structures
using deviceKernelNodeParams_t   = GCXX_RUNTIME_BACKEND(KernelNodeParams);
using deviceMemcpy3DParms_t      = GCXX_RUNTIME_BACKEND(Memcpy3DParms);
using deviceMemsetParams_t       = GCXX_RUNTIME_BACKEND(MemsetParams);
using deviceHostNodeParams_t     = GCXX_RUNTIME_BACKEND(HostNodeParams);
using deviceMemAllocNodeParams_t = GCXX_RUNTIME_BACKEND(MemAllocNodeParams);
using deviceGraphEdgeData_t      = GCXX_RUNTIME_BACKEND(GraphEdgeData);
using deviceGraphNodeParams_t    = GCXX_RUNTIME_BACKEND(GraphNodeParams);
using deviceExternalSemaphoreSignalNodeParams_t =
  GCXX_RUNTIME_BACKEND(ExternalSemaphoreSignalNodeParams);
using deviceExternalSemaphoreWaitNodeParams_t =
  GCXX_RUNTIME_BACKEND(ExternalSemaphoreWaitNodeParams);

GCXX_NAMESPACE_MAIN_DETAILS_END


GCXX_NAMESPACE_MAIN_BEGIN

class GraphView {
 protected:
  using deviceGraph_t     = details_::deviceGraph_t;
  using deviceGraphNode_t = details_::deviceGraphNode_t;
  using deviceEvent_t     = details_::deviceEvent_t;
  using deviceMemcpyKind  = details_::deviceMemcpyKind;

  deviceGraph_t graph_{details_::INVALID_GRAPH};

 public:
  GCXX_FHC GraphView() = default;
  GCXX_FHC GraphView(deviceGraph_t rawgraph);
  GCXX_FHC auto getRawGraph() const -> deviceGraph_t;
  GCXX_FH auto SaveDotfile(std::string_view, flags::graphDebugDot) const
    -> void;
  GCXX_FH auto GetNumNodes() const -> size_t;
  GCXX_FH auto GetNumEdges() const -> size_t;
  GCXX_FH auto Clone() const -> GraphView;

  // ════════════════════════════════════════════════════════════════════════
  // Graph Node Addition Methods
  // ════════════════════════════════════════════════════════════════════════

  /**
   * @brief Creates a child graph node and adds it to a graph.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param childGraph The child graph to embed
   * @return The created graph node
   */
  GCXX_FH auto AddChildGraphNode(const deviceGraphNode_t* pDependencies,
                                 size_t numDependencies,
                                 const GraphView& childGraph)
    -> deviceGraphNode_t;

  /**
   * @brief Adds dependency edges to a graph.
   * @param from Array of "from" nodes
   * @param to Array of "to" nodes
   * @param numDependencies Number of dependencies to add
   */
  GCXX_FH auto AddDependencies(const deviceGraphNode_t* from,
                               const deviceGraphNode_t* to,
                               size_t numDependencies) -> void;

  /**
   * @brief Creates an empty node and adds it to a graph.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @return The created graph node
   */
  GCXX_FH auto AddEmptyNode(const deviceGraphNode_t* pDependencies,
                            size_t numDependencies) -> deviceGraphNode_t;

  GCXX_FH auto AddEmptyNode(gcxx::span<deviceGraphNode_t> pDependencies)
    -> deviceGraphNode_t;

  /**
   * @brief Creates an event record node and adds it to a graph.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param event The event to record
   * @return The created graph node
   */
  GCXX_FH auto AddEventRecordNode(const deviceGraphNode_t* pDependencies,
                                  size_t numDependencies, deviceEvent_t event)
    -> deviceGraphNode_t;

  GCXX_FH auto AddEventRecordNode(gcxx::span<deviceGraphNode_t> pDependencies,
                                  deviceEvent_t event) -> deviceGraphNode_t;

  /**
   * @brief Creates an event wait node and adds it to a graph.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param event The event to wait on
   * @return The created graph node
   */
  GCXX_FH auto AddEventWaitNode(const deviceGraphNode_t* pDependencies,
                                size_t numDependencies, deviceEvent_t event)
    -> deviceGraphNode_t;

  // /**
  //  * @brief Creates an external semaphore signal node and adds it to a graph.
  //  * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
  //  * @param numDependencies Number of dependencies
  //  * @param nodeParams The external semaphore signal node parameters
  //  * @return The created graph node
  //  */
  // GCXX_FH auto AddExternalSemaphoresSignalNode(
  //   const deviceGraphNode_t* pDependencies, size_t numDependencies,
  //   const details_::deviceExternalSemaphoreSignalNodeParams_t* nodeParams)
  //   -> deviceGraphNode_t;

  // /**
  //  * @brief Creates an external semaphore wait node and adds it to a graph.
  //  * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
  //  * @param numDependencies Number of dependencies
  //  * @param nodeParams The external semaphore wait node parameters
  //  * @return The created graph node
  //  */
  // GCXX_FH auto AddExternalSemaphoresWaitNode(
  //   const deviceGraphNode_t* pDependencies, size_t numDependencies,
  //   const details_::deviceExternalSemaphoreWaitNodeParams_t* nodeParams)
  //   -> deviceGraphNode_t;
  
    GCXX_FH auto AddEventWaitNode(gcxx::span<deviceGraphNode_t> pDependencies,
                                deviceEvent_t event) -> deviceGraphNode_t;

  /**
   * @brief Creates a host execution node and adds it to a graph.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param nodeParams The host node parameters
   * @return The created graph node
   */
  GCXX_FH auto AddHostNode(const deviceGraphNode_t* pDependencies,
                           size_t numDependencies,
                           const details_::deviceHostNodeParams_t* nodeParams)
    -> deviceGraphNode_t;

  GCXX_FH auto AddHostNode(gcxx::span<deviceGraphNode_t> pDependencies,
                           const details_::deviceHostNodeParams_t* nodeParams)
    -> deviceGraphNode_t;

  /**
   * @brief Creates a kernel execution node and adds it to a graph.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param nodeParams The kernel node parameters
   * @return The created graph node
   */
  GCXX_FH auto AddKernelNode(
    const deviceGraphNode_t* pDependencies, size_t numDependencies,
    const details_::deviceKernelNodeParams_t* nodeParams) -> deviceGraphNode_t;

  GCXX_FH auto AddKernelNode(gcxx::span<deviceGraphNode_t>,
                             const details_::deviceKernelNodeParams_t*)
    -> deviceGraphNode_t;

  /**
   * @brief Creates a memory allocation node and adds it to a graph.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param nodeParams The memory allocation node parameters
   * @return The created graph node
   */
  GCXX_FH auto AddMemAllocNode(const deviceGraphNode_t* pDependencies,
                               size_t numDependencies,
                               details_::deviceMemAllocNodeParams_t* nodeParams)
    -> deviceGraphNode_t;

  GCXX_FH auto AddMemAllocNode(gcxx::span<deviceGraphNode_t> pDependencies,
                               details_::deviceMemAllocNodeParams_t* nodeParams)
    -> deviceGraphNode_t;

  /**
   * @brief Creates a memory free node and adds it to a graph.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param dptr Device pointer to free
   * @return The created graph node
   */
  GCXX_FH auto AddMemFreeNode(const deviceGraphNode_t* pDependencies,
                              size_t numDependencies, void* dptr)
    -> deviceGraphNode_t;

  GCXX_FH auto AddMemFreeNode(gcxx::span<deviceGraphNode_t> pDependencies,
                              void* dptr) -> deviceGraphNode_t;

  /**
   * @brief Creates a memcpy node and adds it to a graph.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param copyParams The 3D memcpy parameters
   * @return The created graph node
   */
  GCXX_FH auto AddMemcpyNode(const deviceGraphNode_t* pDependencies,
                             size_t numDependencies,
                             const details_::deviceMemcpy3DParms_t* copyParams)
    -> deviceGraphNode_t;

  GCXX_FH auto AddMemcpyNode(gcxx::span<deviceGraphNode_t> pDependencies,
                             const details_::deviceMemcpy3DParms_t* copyParams)
    -> deviceGraphNode_t;

  /**
   * @brief Creates a 1D memcpy node and adds it to a graph.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param dst Destination pointer
   * @param src Source pointer
   * @param count Number of bytes to copy
   * @param kind Type of transfer
   * @return The created graph node
   */
  GCXX_FH auto AddMemcpyNode1D(const deviceGraphNode_t* pDependencies,
                               size_t numDependencies, void* dst,
                               const void* src, size_t count,
                               deviceMemcpyKind kind) -> deviceGraphNode_t;

  GCXX_FH auto AddMemcpyNode1D(gcxx::span<deviceGraphNode_t> pDependencies,
                               void* dst, const void* src, size_t count,
                               deviceMemcpyKind kind) -> deviceGraphNode_t;

  /**
   * @brief Creates a memcpy node to copy from a symbol on the device.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param dst Destination pointer
   * @param symbol Device symbol address
   * @param count Number of bytes to copy
   * @param offset Offset from symbol start
   * @param kind Type of transfer
   * @return The created graph node
   */
  GCXX_FH auto AddMemcpyNodeFromSymbol(const deviceGraphNode_t* pDependencies,
                                       size_t numDependencies, void* dst,
                                       const void* symbol, size_t count,
                                       size_t offset, deviceMemcpyKind kind)
    -> deviceGraphNode_t;

  GCXX_FH auto AddMemcpyNodeFromSymbol(
    gcxx::span<deviceGraphNode_t> pDependencies, void* dst, const void* symbol,
    size_t count, size_t offset, deviceMemcpyKind kind) -> deviceGraphNode_t;

  /**
   * @brief Creates a memcpy node to copy to a symbol on the device.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param symbol Device symbol address
   * @param src Source pointer
   * @param count Number of bytes to copy
   * @param offset Offset from symbol start
   * @param kind Type of transfer
   * @return The created graph node
   */
  GCXX_FH auto AddMemcpyNodeToSymbol(const deviceGraphNode_t* pDependencies,
                                     size_t numDependencies, const void* symbol,
                                     const void* src, size_t count,
                                     size_t offset, deviceMemcpyKind kind)
    -> deviceGraphNode_t;

  GCXX_FH auto AddMemcpyNodeToSymbol(
    gcxx::span<deviceGraphNode_t> pDependencies, const void* symbol,
    const void* src, size_t count, size_t offset, deviceMemcpyKind kind)
    -> deviceGraphNode_t;

  /**
   * @brief Creates a memset node and adds it to a graph.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param memsetParams The memset parameters
   * @return The created graph node
   */
  GCXX_FH auto AddMemsetNode(const deviceGraphNode_t* pDependencies,
                             size_t numDependencies,
                             const details_::deviceMemsetParams_t* memsetParams)
    -> deviceGraphNode_t;

  GCXX_FH auto AddMemsetNode(gcxx::span<deviceGraphNode_t> pDependencies,
                             const details_::deviceMemsetParams_t* memsetParams)
    -> deviceGraphNode_t;

  /**
   * @brief Adds a node of arbitrary type to a graph.
   * @param pDependencies Pointer to array of dependent nodes (can be nullptr)
   * @param numDependencies Number of dependencies
   * @param nodeParams The generic node parameters
   * @return The created graph node
   */
  GCXX_FH auto AddNode(const deviceGraphNode_t* pDependencies,
                       size_t numDependencies,
                       details_::deviceGraphNodeParams_t* nodeParams)
    -> deviceGraphNode_t;

  GCXX_FH auto AddNode(gcxx::span<deviceGraphNode_t> pDependencies,
                       details_::deviceGraphNodeParams_t* nodeParams)
    -> deviceGraphNode_t;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/graph_view.inl>

#include <gcxx/macros/undefine_macros.hpp>
#endif
