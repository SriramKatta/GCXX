// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_HOST_NODE_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_HOST_NODE_PARAMS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_graph.hpp>

#include <gcxx/runtime/memory/memory_helpers.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class HostNodeParamsView {
 public:
  using deviceHostCallBackFn_t = driver::deviceHostCallBackFn_t;
  using deviceHostNodeParams_t = driver::deviceHostNodeParams_t;


  GCXX_FHC auto getRawParams() const -> const deviceHostNodeParams_t& {
    return m_params;
  }

  GCXX_FHC auto getHostFunc() const -> const deviceHostCallBackFn_t {
    return m_params.fn;
  }

  GCXX_FHC auto getUserData() const -> const void* { return m_params.userData; }

 protected:
  GCXX_FHC HostNodeParamsView() { std::memset(&m_params, 0, sizeof(m_params)); }
  deviceHostNodeParams_t m_params{};  // NOLINT
};

class HostNodeParams : public HostNodeParamsView {

 public:
  GCXX_FHC HostNodeParams() = default;

  GCXX_FHC HostNodeParams(deviceHostCallBackFn_t fn, void* Udata) {
    m_params.fn       = fn;
    m_params.userData = Udata;
  }

  // Disable move/copy to ensure m_params remains stable.
  HostNodeParams(const HostNodeParams&) = delete;
  HostNodeParams(HostNodeParams&&)      = delete;

  HostNodeParams operator=(const HostNodeParams&) = delete;
  HostNodeParams operator=(HostNodeParams&&)      = delete;

  ~HostNodeParams() = default;
};

GCXX_NAMESPACE_DETAILS_BEGIN()

class HostNodeParamsBuilder {
 public:
  GCXX_FH static auto create() -> HostNodeParamsBuilder { return {}; }

  GCXX_FHC
  auto setHostCallbackFn(HostNodeParamsView::deviceHostCallBackFn_t func)
    -> HostNodeParamsBuilder& {
    m_func = func;
    return *this;
  }

  GCXX_FHC auto setUserData(void* udata) -> HostNodeParamsBuilder& {
    m_Udata = udata;
    return *this;
  }

  GCXX_FHC auto build() -> gcxx::HostNodeParams { return {m_func, m_Udata}; }

 private:
  HostNodeParamsView::deviceHostCallBackFn_t m_func{};
  void* m_Udata{nullptr};
};

GCXX_NAMESPACE_DETAILS_END()

// Helper to simplify usage.
GCXX_FH auto HostNodeParamsBuilder() -> details_::HostNodeParamsBuilder {
  return details_::HostNodeParamsBuilder::create();
}

GCXX_NAMESPACE_MAIN_END()


#endif