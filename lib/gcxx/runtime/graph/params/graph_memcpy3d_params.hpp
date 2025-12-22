#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEMCPY3D_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEMCPY3D_PARAMS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/memory/memory_helpers.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

using deviceMemcpy3DNodeParams_t = GCXX_RUNTIME_BACKEND(Memcpy3DParms);

class Memcpy3DNodeParamsView {
 protected:
  deviceMemcpy3DNodeParams_t params_{};

  GCXX_FHC Memcpy3DNodeParamsView() { memset(&params_, 0, sizeof(params_)); }
 public:
  GCXX_FHC auto getRawParams() const -> const deviceMemcpy3DNodeParams_t& {
    return params_;
  }

  GCXX_FHC auto getSrcPos() const -> const gcxx::memory::devicePos {
    return params_.srcPos;
  }

  GCXX_FHC auto getDstPos() const -> const gcxx::memory::devicePos {
    return params_.dstPos;
  }

  GCXX_FHC auto getSrcPtr() const -> const gcxx::memory::devicePitchedPtr {
    return params_.srcPtr;
  }

  GCXX_FHC auto getDstPtr() const -> const gcxx::memory::devicePitchedPtr {
    return params_.dstPtr;
  }

  GCXX_FHC auto getExtent() const -> const gcxx::memory::deviceExtent {
    return params_.extent;
  }
};

class Memcpy3DNodeParams : public Memcpy3DNodeParamsView {

 public:
  GCXX_FHC Memcpy3DNodeParams() = default;

  GCXX_FHC Memcpy3DNodeParams(const gcxx::memory::devicePitchedPtr& srcPtr,
                              gcxx::memory::devicePos srcPos,
                              const gcxx::memory::devicePitchedPtr& dstPtr,
                              gcxx::memory::devicePos dstPos,
                              gcxx::memory::deviceExtent extent) {
    params_.srcPtr = srcPtr;
    params_.srcPos = srcPos;
    params_.dstPtr = dstPtr;
    params_.dstPos = dstPos;
    params_.extent = extent;
    params_.kind   = GCXX_RUNTIME_BACKEND(MemcpyDefault);
  }

  // Disable move/copy to ensure params_ remains stable
  Memcpy3DNodeParams(const Memcpy3DNodeParams&) = delete;
  Memcpy3DNodeParams(Memcpy3DNodeParams&&)      = delete;
};

GCXX_NAMESPACE_DETAILS_BEGIN

class Memcpy3DParamsBuilder {
 private:
  gcxx::memory::devicePitchedPtr srcPtr_{};
  gcxx::memory::devicePos srcPos_{0,0,0};
  gcxx::memory::devicePitchedPtr dstPtr_{};
  gcxx::memory::devicePos dstPos_{0,0,0};
  gcxx::memory::deviceExtent extent_{};

 public:
  GCXX_FH static auto create() -> Memcpy3DParamsBuilder { return {}; }

  GCXX_FHC auto setSrcPtr(const gcxx::memory::devicePitchedPtr& ptr)
    -> Memcpy3DParamsBuilder& {
    srcPtr_ = ptr;
    return *this;
  }

  GCXX_FHC auto setSrcPos(gcxx::memory::devicePos pos)
    -> Memcpy3DParamsBuilder& {
    srcPos_ = pos;
    return *this;
  }

  GCXX_FHC auto setDstPtr(const gcxx::memory::devicePitchedPtr& ptr)
    -> Memcpy3DParamsBuilder& {
    dstPtr_ = ptr;
    return *this;
  }

  GCXX_FHC auto setDstPos(gcxx::memory::devicePos pos)
    -> Memcpy3DParamsBuilder& {
    dstPos_ = pos;
    return *this;
  }

  GCXX_FHC auto setExtent(gcxx::memory::deviceExtent ext)
    -> Memcpy3DParamsBuilder& {
    extent_ = ext;
    return *this;
  }

  GCXX_FHC gcxx::Memcpy3DNodeParams build() {
    return Memcpy3DNodeParams(srcPtr_, srcPos_, dstPtr_, dstPos_, extent_);
  }
};

GCXX_NAMESPACE_DETAILS_END

// helper to simplify usage
GCXX_FH auto Memcpy3DParamsBuilder() -> details_::Memcpy3DParamsBuilder {
  return details_::Memcpy3DParamsBuilder::create();
}

GCXX_NAMESPACE_MAIN_END


#endif