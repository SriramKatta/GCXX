#pragma once
#ifndef GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEMSET_PARAMS_HPP_
#define GCXX_RUNTIME_GRAPH_PARAMS_GRAPH_MEMSET_PARAMS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/memory/memory_helpers.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

using deviceMemsetParams_t = GCXX_RUNTIME_BACKEND(MemsetParams);

class MemsetParamsView {
 protected:
  deviceMemsetParams_t params_{};  // NOLINT

  GCXX_FHC MemsetParamsView() { std::memset(&params_, 0, sizeof(params_)); }

 public:
  GCXX_FHC auto getRawParams() const -> const deviceMemsetParams_t& {
    return params_;
  }

  GCXX_FHC auto getPtr() const -> const void* const { return params_.dst; }

  GCXX_FHC auto getPitch() const -> const size_t { return params_.pitch; }

  GCXX_FHC auto getValue() const -> const unsigned int { return params_.value; }

  GCXX_FHC auto getElementSize() const -> const unsigned int {
    return params_.elementSize;
  }

  GCXX_FHC auto getWidth() const -> const size_t { return params_.width; }

  GCXX_FHC auto getHeight() const -> const size_t { return params_.height; }
};

class MemsetParams : public MemsetParamsView {

 public:
  GCXX_FHC MemsetParams() = default;

  GCXX_FHC MemsetParams(void* dst, size_t pitch, unsigned int value,
                        unsigned int elementSize, size_t width, size_t height) {
    params_.dst         = dst;
    params_.pitch       = pitch;
    params_.value       = value;
    params_.elementSize = elementSize;
    params_.width       = width;
    params_.height      = height;
  }

  // Disable move/copy to ensure params_ remains stable
  MemsetParams(const MemsetParams&)           = delete;
  MemsetParams operator=(const MemsetParams&) = delete;

  MemsetParams(MemsetParams&&)           = delete;
  MemsetParams operator=(MemsetParams&&) = delete;

  ~MemsetParams() = default;
};

GCXX_NAMESPACE_DETAILS_BEGIN

class MemsetParamsBuilder {
 private:
  void* dst_{nullptr};
  size_t pitch_{0};
  unsigned int value_{0};
  unsigned int elementSize_{1};
  size_t width_{1};
  size_t height_{1};

 public:
  GCXX_FH static auto create() -> MemsetParamsBuilder { return {}; }

  GCXX_FHC auto setPtr(void* ptr) -> MemsetParamsBuilder& {
    dst_ = ptr;
    return *this;
  }

  GCXX_FHC auto setPitch(size_t pitch) -> MemsetParamsBuilder& {
    pitch_ = pitch;
    return *this;
  }

  GCXX_FHC auto setValue(unsigned int value) -> MemsetParamsBuilder& {
    value_ = value;
    return *this;
  }

  GCXX_FHC auto setElemetSize(unsigned int size) -> MemsetParamsBuilder& {
    elementSize_ = size;
    return *this;
  }

  GCXX_FHC auto setWidth(size_t width) -> MemsetParamsBuilder& {
    width_ = width;
    return *this;
  }

  GCXX_FHC auto setHeight(size_t height) -> MemsetParamsBuilder& {
    height_ = height;
    return *this;
  }

  GCXX_FHC gcxx::MemsetParams build() {
    return {dst_, pitch_, value_, elementSize_, width_, height_};
  }
};

GCXX_NAMESPACE_DETAILS_END

// helper to simplify usage
GCXX_FH auto MemsetParamsBuilder() -> details_::MemsetParamsBuilder {
  return details_::MemsetParamsBuilder::create();
}

GCXX_NAMESPACE_MAIN_END


#endif