#pragma once
#ifndef GCXX_RUNTIME_GRAPH_GRAPH_NODE_HPP_
#define GCXX_RUNTIME_GRAPH_GRAPH_NODE_HPP_

#include <new>
#include <utility>


#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>


#include <gcxx/runtime/flags/graph_flags.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
using deviceKernelNodeParams_t   = GCXX_RUNTIME_BACKEND(KernelNodeParams);
using deviceGraphNode_t          = GCXX_RUNTIME_BACKEND(GraphNode_t);
using deviceGraphExec_t          = GCXX_RUNTIME_BACKEND(GraphExec_t);
using deviceLaunchAttributeID    = GCXX_RUNTIME_BACKEND(LaunchAttributeID);
using deviceLaunchAttributeValue = GCXX_RUNTIME_BACKEND(LaunchAttributeValue);

// Alias for kernel node attribute types (CUDA 12.x uses LaunchAttribute)
using deviceKernelNodeAttrID    = deviceLaunchAttributeID;
using deviceKernelNodeAttrValue = deviceLaunchAttributeValue;

#if GCXX_CUDA_MODE
using deviceGraphDeviceNode_t     = cudaGraphDeviceNode_t;
using deviceGraphKernelNodeField  = cudaGraphKernelNodeField;
using deviceGraphKernelNodeUpdate = cudaGraphKernelNodeUpdate;
#endif

GCXX_NAMESPACE_MAIN_DETAILS_END

GCXX_NAMESPACE_MAIN_BEGIN

// ════════════════════════════════════════════════════════════════════════════
// GraphNode Base Class
// ════════════════════════════════════════════════════════════════════════════

class GraphNode {
 private:
  flags::graphNodeType type_;

 protected:
  details_::deviceGraphNode_t node_{nullptr};

  GCXX_FHC explicit GraphNode(flags::graphNodeType type) : type_(type) {}

  GCXX_FHC explicit GraphNode(flags::graphNodeType type,
                              details_::deviceGraphNode_t node)
      : type_(type), node_(node) {}

 public:
  GCXX_FHC auto getType() const -> flags::graphNodeType { return type_; }

  GCXX_FHC auto getRawNode() const -> details_::deviceGraphNode_t {
    return node_;
  }

  GCXX_FHC operator details_::deviceGraphNode_t() const GCXX_NOEXCEPT {
    return node_;
  }

  virtual ~GraphNode() = default;
};


// // ════════════════════════════════════════════════════════════════════════════
// // KernelNodeParams Wrapper
// // ════════════════════════════════════════════════════════════════════════════

// /**
//  * @brief RAII wrapper for kernel node parameters with builder pattern.
//  *
//  * Simplifies creation and management of kernel node parameters.
//  * Example usage:
//  * @code
//  *   auto params = KernelNodeParams::Create<myKernel>(gridDim, blockDim)
//  *                   .setSharedMem(1024)
//  *                   .setArgs(d_input, d_output, size);
//  * @endcode
//  */
// class KernelNodeParams {
//  private:
//   details_::deviceKernelNodeParams_t params_{};

//  public:
//   GCXX_FH KernelNodeParams() { memset(&params_, 0, sizeof(params_)); }

//   GCXX_FH ~KernelNodeParams() = default;

//   // Move semantics
//   GCXX_FH KernelNodeParams(KernelNodeParams&& other) GCXX_NOEXCEPT
//       : params_(other.params_) {
//     memset(&other.params_, 0, sizeof(details_::deviceKernelNodeParams_t));
//   }

//   GCXX_FH auto operator=(KernelNodeParams&& other)
//     GCXX_NOEXCEPT->KernelNodeParams& {
//     if (this != &other) {
//       params_ = other.params_;
//       memset(&other.params_, 0, sizeof(details_::deviceKernelNodeParams_t));
//     }
//     return *this;
//   }

//   // Delete copy operations
//   KernelNodeParams(const KernelNodeParams&)            = delete;
//   KernelNodeParams& operator=(const KernelNodeParams&) = delete;

//   // ──────────────────────────────────────────────────────────────────────────
//   // Factory Methods
//   // ──────────────────────────────────────────────────────────────────────────

//   /**
//    * @brief Create kernel node params from a kernel function pointer.
//    * @tparam KernelFunc Type of kernel function
//    * @param kernel Pointer to kernel function
//    * @param gridDim Grid dimensions
//    * @param blockDim Block dimensions
//    * @param sharedMem Shared memory size in bytes (default: 0)
//    * @return KernelNodeParams instance
//    */
//   template <typename KernelFunc>
//   GCXX_FH static auto Create(KernelFunc kernel, dim3 gridDim, dim3 blockDim,
//                              size_t sharedMem = 0) -> KernelNodeParams {
//     KernelNodeParams p;
//     p.params_.func           = reinterpret_cast<void*>(kernel);
//     p.params_.gridDim        = gridDim;
//     p.params_.blockDim       = blockDim;
//     p.params_.sharedMemBytes = static_cast<unsigned int>(sharedMem);
//     p.params_.kernelParams   = nullptr;
//     p.params_.extra          = nullptr;
//     return p;
//   }

//   // ──────────────────────────────────────────────────────────────────────────
//   // Builder Pattern Setters
//   // ──────────────────────────────────────────────────────────────────────────

//   GCXX_FH auto setFunc(void* func) & -> KernelNodeParams& {
//     params_.func = func;
//     return *this;
//   }

//   GCXX_FH auto setFunc(void* func) && -> KernelNodeParams&& {
//     params_.func = func;
//     return std::move(*this);
//   }

//   template <typename KernelFunc>
//   GCXX_FH auto setFunc(KernelFunc kernel) & -> KernelNodeParams& {
//     params_.func = reinterpret_cast<void*>(kernel);
//     return *this;
//   }

//   template <typename KernelFunc>
//   GCXX_FH auto setFunc(KernelFunc kernel) && -> KernelNodeParams&& {
//     params_.func = reinterpret_cast<void*>(kernel);
//     return std::move(*this);
//   }

//   GCXX_FH auto setGridDim(dim3 gridDim) & -> KernelNodeParams& {
//     params_.gridDim = gridDim;
//     return *this;
//   }

//   GCXX_FH auto setGridDim(dim3 gridDim) && -> KernelNodeParams&& {
//     params_.gridDim = gridDim;
//     return std::move(*this);
//   }

//   GCXX_FH auto setBlockDim(dim3 blockDim) & -> KernelNodeParams& {
//     params_.blockDim = blockDim;
//     return *this;
//   }

//   GCXX_FH auto setBlockDim(dim3 blockDim) && -> KernelNodeParams&& {
//     params_.blockDim = blockDim;
//     return std::move(*this);
//   }

//   GCXX_FH auto setSharedMem(size_t bytes) & -> KernelNodeParams& {
//     params_.sharedMemBytes = static_cast<unsigned int>(bytes);
//     return *this;
//   }

//   GCXX_FH auto setSharedMem(size_t bytes) && -> KernelNodeParams&& {
//     params_.sharedMemBytes = static_cast<unsigned int>(bytes);
//     return std::move(*this);
//   }

//   /**
//    * @brief Set kernel arguments from raw pointer array.
//    * @param args Pointer array (caller maintains ownership)
//    * @return Reference for chaining
//    */
//   GCXX_FH auto setArgsRaw(void** args) & -> KernelNodeParams& {
//     params_.kernelParams = args;
//     return *this;
//   }

//   GCXX_FH auto setArgsRaw(void** args) && -> KernelNodeParams&& {
//     params_.kernelParams = args;
//     return std::move(*this);
//   }

//  public:
//   /**
//    * @brief Set kernel arguments by reference.
//    *
//    * This method stores pointers to the provided arguments.
//    * The caller must ensure the arguments remain valid
//    * for the lifetime of the graph execution.
//    *
//    * @tparam Args Types of kernel arguments
//    * @param args Kernel arguments (by reference)
//    * @return Reference for chaining
//    *
//    * Example:
//    * @code
//    *   float* d_input;
//    *   int size;
//    *   auto params = KernelNodeParams::Create(myKernel, grid, block)
//    *                   .setArgs(d_input, size);  // pass by reference
//    * @endcode
//    */
//   template <typename... Args>
//   GCXX_FH auto setArgs(Args&... args) & -> KernelNodeParams& {
//     params_.kernelParams = {&args...};
//     return *this;
//   }

//   template <typename... Args>
//   GCXX_FH auto setArgs(Args&... args) && -> KernelNodeParams&& {
//     params_.kernelParams = {(void*)&args...};
//     return std::move(*this);
//   }

//   GCXX_FH auto setExtra(void** extra) & -> KernelNodeParams& {
//     params_.extra = extra;
//     return *this;
//   }

//   GCXX_FH auto setExtra(void** extra) && -> KernelNodeParams&& {
//     params_.extra = extra;
//     return std::move(*this);
//   }

//   // ──────────────────────────────────────────────────────────────────────────
//   // Getters
//   // ──────────────────────────────────────────────────────────────────────────

//   GCXX_FHC auto getFunc() const -> void* { return params_.func; }

//   GCXX_FHC auto getGridDim() const -> dim3 { return params_.gridDim; }

//   GCXX_FHC auto getBlockDim() const -> dim3 { return params_.blockDim; }

//   GCXX_FHC auto getSharedMem() const -> unsigned int {
//     return params_.sharedMemBytes;
//   }

//   GCXX_FHC auto getKernelParams() const -> void** {
//     return params_.kernelParams;
//   }

//   GCXX_FHC auto getRawParams() const
//     -> const details_::deviceKernelNodeParams_t* {
//     return &params_;
//   }

//   GCXX_FH auto getRawParams() -> details_::deviceKernelNodeParams_t* {
//     return &params_;
//   }
// };

// // ════════════════════════════════════════════════════════════════════════════
// // KernelNodeView - Non-owning view of a kernel node
// // ════════════════════════════════════════════════════════════════════════════

// /**
//  * @brief Non-owning view of a kernel graph node.
//  *
//  * Provides methods to query and modify kernel node parameters without
//  * managing the node's lifetime.
//  */
// class KernelNodeView : public GraphNode {
//  public:
//   using base_t = GraphNode;

//   GCXX_FHC KernelNodeView() : GraphNode(flags::graphNodeType::kernel) {}

//   GCXX_FHC explicit KernelNodeView(details_::deviceGraphNode_t node)
//       : GraphNode(flags::graphNodeType::kernel, node) {}

//   // ──────────────────────────────────────────────────────────────────────────
//   // Host API: Parameter Getters/Setters
//   // ──────────────────────────────────────────────────────────────────────────

//   /**
//    * @brief Get kernel node parameters.
//    * @param params Output parameter struct
//    * @return Reference for chaining
//    *
//    * Wraps: cudaGraphKernelNodeGetParams
//    */
//   GCXX_FH auto GetParams(details_::deviceKernelNodeParams_t* params) const
//     -> const KernelNodeView&;

//   /**
//    * @brief Get kernel node parameters as a wrapper object.
//    * @return KernelNodeParams with current node settings
//    */
//   GCXX_FH auto GetParams() const -> KernelNodeParams;

//   /**
//    * @brief Set kernel node parameters.
//    * @param params Parameter struct
//    * @return Reference for chaining
//    *
//    * Wraps: cudaGraphKernelNodeSetParams
//    */
//   GCXX_FH auto SetParams(const details_::deviceKernelNodeParams_t* params)
//     -> KernelNodeView&;

//   GCXX_FH auto SetParams(const KernelNodeParams& params) -> KernelNodeView&;

//   // ──────────────────────────────────────────────────────────────────────────
//   // Host API: Attributes
//   // ──────────────────────────────────────────────────────────────────────────

//   /**
//    * @brief Get a kernel node attribute.
//    * @param attr Attribute ID to query
//    * @param value Output value
//    * @return Reference for chaining
//    *
//    * Wraps: cudaGraphKernelNodeGetAttribute
//    */
//   GCXX_FH auto GetAttribute(details_::deviceKernelNodeAttrID attr,
//                             details_::deviceKernelNodeAttrValue* value) const
//     -> const KernelNodeView&;

//   /**
//    * @brief Set a kernel node attribute.
//    * @param attr Attribute ID to set
//    * @param value Attribute value
//    * @return Reference for chaining
//    *
//    * Wraps: cudaGraphKernelNodeSetAttribute
//    */
//   GCXX_FH auto SetAttribute(details_::deviceKernelNodeAttrID attr,
//                             const details_::deviceKernelNodeAttrValue* value)
//     -> KernelNodeView&;

//   /**
//    * @brief Copy attributes from another kernel node.
//    * @param src Source kernel node
//    * @return Reference for chaining
//    *
//    * Wraps: cudaGraphKernelNodeCopyAttributes
//    */
//   GCXX_FH auto CopyAttributesFrom(const KernelNodeView& src) -> KernelNodeView&;

//   // ──────────────────────────────────────────────────────────────────────────
//   // GraphExec Update Methods (for use with GraphExec)
//   // ──────────────────────────────────────────────────────────────────────────

//   /**
//    * @brief Update kernel node parameters in an executable graph.
//    * @param exec The executable graph
//    * @param params New parameters
//    *
//    * Wraps: cudaGraphExecKernelNodeSetParams
//    */
//   GCXX_FH auto SetParamsInExec(details_::deviceGraphExec_t exec,
//                                const details_::deviceKernelNodeParams_t* params)
//     -> KernelNodeView&;

//   GCXX_FH auto SetParamsInExec(details_::deviceGraphExec_t exec,
//                                const KernelNodeParams& params)
//     -> KernelNodeView&;

// #if GCXX_CUDA_MODE
//   // ──────────────────────────────────────────────────────────────────────────
//   // Device API (CUDA only) - For use from device code in device graphs
//   // ──────────────────────────────────────────────────────────────────────────

//   /**
//    * @brief Enable or disable a kernel node (device API).
//    * @param deviceNode Device graph node handle
//    * @param enable Whether to enable or disable
//    *
//    * Wraps: cudaGraphKernelNodeSetEnabled
//    */
//   GCXX_FD static auto SetEnabled(details_::deviceGraphDeviceNode_t deviceNode,
//                                  bool enable) -> void;

//   /**
//    * @brief Update grid dimensions of a kernel node (device API).
//    * @param deviceNode Device graph node handle
//    * @param gridDim New grid dimensions
//    *
//    * Wraps: cudaGraphKernelNodeSetGridDim
//    */
//   GCXX_FD static auto SetGridDim(details_::deviceGraphDeviceNode_t deviceNode,
//                                  dim3 gridDim) -> void;

//   /**
//    * @brief Update a single kernel parameter (device API).
//    * @tparam T Parameter type
//    * @param deviceNode Device graph node handle
//    * @param offset Offset of parameter in kernel args
//    * @param value New parameter value
//    *
//    * Wraps: cudaGraphKernelNodeSetParam (templated version)
//    */
//   template <typename T>
//   GCXX_FD static auto SetParam(details_::deviceGraphDeviceNode_t deviceNode,
//                                size_t offset, const T& value) -> void {
//     GCXX_RUNTIME_BACKEND(GraphKernelNodeSetParam)(deviceNode, offset, value);
//   }

//   /**
//    * @brief Update a single kernel parameter with raw memory (device API).
//    * @param deviceNode Device graph node handle
//    * @param offset Offset of parameter in kernel args
//    * @param value Pointer to new value
//    * @param size Size of value in bytes
//    *
//    * Wraps: cudaGraphKernelNodeSetParam (void* version)
//    */
//   GCXX_FD static auto SetParam(details_::deviceGraphDeviceNode_t deviceNode,
//                                size_t offset, const void* value, size_t size)
//     -> void;

//   /**
//    * @brief Batch apply multiple kernel node updates (device API).
//    * @param updates Array of updates to apply
//    * @param updateCount Number of updates
//    *
//    * Wraps: cudaGraphKernelNodeUpdatesApply
//    */
//   GCXX_FD static auto ApplyUpdates(
//     const details_::deviceGraphKernelNodeUpdate* updates, size_t updateCount)
//     -> void;
// #endif
// };

// // ════════════════════════════════════════════════════════════════════════════
// // KernelNode - Owning wrapper (if needed for future RAII node management)
// // ════════════════════════════════════════════════════════════════════════════

// /**
//  * @brief Owning wrapper for a kernel graph node.
//  *
//  * Currently inherits from KernelNodeView. Can be extended for RAII
//  * node lifecycle management if needed.
//  */
// class KernelNode : public KernelNodeView {
//  public:
//   using base_t = KernelNodeView;

//   GCXX_FH KernelNode() : KernelNodeView() {}

//   GCXX_FH explicit KernelNode(details_::deviceGraphNode_t node)
//       : KernelNodeView(node) {}

//   // Factory method to create from raw node
//   GCXX_FH static auto FromRaw(details_::deviceGraphNode_t node) -> KernelNode {
//     return KernelNode(node);
//   }
// };

GCXX_NAMESPACE_MAIN_END

// #include <gcxx/runtime/details/graph_node.inl>

#include <gcxx/macros/undefine_macros.hpp>

#endif