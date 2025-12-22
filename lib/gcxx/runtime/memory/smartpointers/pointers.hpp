#pragma once
#ifndef GCXX_RUNTIME_SMARTPOINTERS_POINTERS_HPP_
#define GCXX_RUNTIME_SMARTPOINTERS_POINTERS_HPP_

#include <memory>


#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/details/memory/device_memory_helper.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

namespace memory {

  template <typename VT, typename DT>
  using gcxx_unique_ptr = std::unique_ptr<VT, DT>;

  template <typename VT>
  using device_ptr = gcxx_unique_ptr<VT, decltype(details_::device_free)>;


  template <typename VT>
  using host_pinned_ptr =
    gcxx_unique_ptr<VT[], decltype(details_::host_free)>;  // NOLINT

  template <typename VT>
  auto make_device_unique_ptr(std::size_t numElem) -> device_ptr<VT> {
    return device_ptr<VT>{
      static_cast<VT*>(details_::device_malloc(numElem * sizeof(VT))),
      details_::device_free};
  }

  template <typename VT>
  auto make_host_pinned_unique_ptr(std::size_t numElem) -> host_pinned_ptr<VT> {
    return host_pinned_ptr<VT>{
      static_cast<VT*>(details_::host_malloc(numElem * sizeof(VT))),
      details_::host_free};
  }


}  // namespace memory

GCXX_NAMESPACE_MAIN_END


#endif