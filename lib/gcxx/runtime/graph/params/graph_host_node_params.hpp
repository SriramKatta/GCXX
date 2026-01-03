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

#include <gcxx/runtime/memory/memory_helpers.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

using deviceCallBackFn_t     = GCXX_RUNTIME_BACKEND(HostFn_t);
using deviceHostNodeParams_t = GCXX_RUNTIME_BACKEND(HostNodeParams);

class HostNodeParamsView {
 protected:
  deviceHostNodeParams_t params_{};  // NOLINT

  GCXX_FHC HostNodeParamsView() { std::memset(&params_, 0, sizeof(params_)); }

 public:
  GCXX_FHC auto getRawParams() const -> const deviceHostNodeParams_t& {
    return params_;
  }

  GCXX_FHC auto getHostFunc() const -> const deviceCallBackFn_t {
    return params_.fn;
  }

  GCXX_FHC auto getUserData() const -> const void* { return params_.userData; }
};

class HostNodeParams : public HostNodeParamsView {

 public:
  GCXX_FHC HostNodeParams() = default;

  GCXX_FHC HostNodeParams(deviceCallBackFn_t fn, void* Udata) {
    params_.fn       = fn;
    params_.userData = Udata;
  }

  // Disable move/copy to ensure params_ remains stable
  HostNodeParams(const HostNodeParams&) = delete;
  HostNodeParams(HostNodeParams&&)      = delete;

  HostNodeParams operator=(const HostNodeParams&) = delete;
  HostNodeParams operator=(HostNodeParams&&)      = delete;

  ~HostNodeParams() = default;
};

GCXX_NAMESPACE_DETAILS_BEGIN

class HostNodeParamsBuilder {
 private:
  deviceCallBackFn_t func_{};
  void* Udata_{nullptr};


 public:
  GCXX_FH static auto create() -> HostNodeParamsBuilder { return {}; }

  GCXX_FHC auto setHostCallbackFn(deviceCallBackFn_t func)
    -> HostNodeParamsBuilder& {
    func_ = func;
    return *this;
  }

  GCXX_FHC auto setUserData(void* udata) -> HostNodeParamsBuilder& {
    Udata_ = udata;
    return *this;
  }

  GCXX_FHC gcxx::HostNodeParams build() { return {func_, Udata_}; }
};

GCXX_NAMESPACE_DETAILS_END

// helper to simplify usage
GCXX_FH auto HostNodeParamsBuilder() -> details_::HostNodeParamsBuilder {
  return details_::HostNodeParamsBuilder::create();
}

GCXX_NAMESPACE_MAIN_END


#endif