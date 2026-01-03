#pragma once
#ifndef GCXX_RUNTIME_GRAPH_GRAPH_EXEC_VIEW_HPP_
#define GCXX_RUNTIME_GRAPH_GRAPH_EXEC_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
using deviceGraphExec_t = GCXX_RUNTIME_BACKEND(GraphExec_t);
inline constexpr deviceGraphExec_t INVALID_GRAPH_EXEC{nullptr};
GCXX_NAMESPACE_MAIN_DETAILS_END


GCXX_NAMESPACE_MAIN_BEGIN

class StreamView;

class GraphExecView {
 protected:
  using deviceGraphExec_t = details_::deviceGraphExec_t;
  deviceGraphExec_t exec_{details_::INVALID_GRAPH_EXEC};  // NOLINT

 public:
  GCXX_FHC GraphExecView() = default;
  GCXX_FHC GraphExecView(deviceGraphExec_t rawExec);
  GCXX_FHC auto getRawExec() const -> deviceGraphExec_t;
  GCXX_FHC operator deviceGraphExec_t() const GCXX_NOEXCEPT;

  GCXX_FH auto Launch(const StreamView& stream) const -> void;
  GCXX_FH auto Upload(const StreamView& stream) const -> void;
};

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/graph/graph_exec_view.inl>

#include <gcxx/macros/undefine_macros.hpp>
#endif