#pragma once
#ifndef GCXX_RUNTIME_VECTOR_VECTOR_BASE_HPP_
#define GCXX_RUNTIME_VECTOR_VECTOR_BASE_HPP_

#include <cstddef>
#include <gpucxx/macros/define_macros.hpp>

GCXX_BEGIN_NAMESPACE

template <typename VT, typename Allocator_t>
class VectorBase {
 public:
  using allocator_type  = Alloc;
  using value_type      = T;
  using pointer         = typename alloc_traits::pointer;
  using const_pointer   = typename alloc_traits::const_pointer;
  using size_type       = typename alloc_traits::size_type;
  using difference_type = typename alloc_traits::difference_type;
  using reference       = typename alloc_traits::reference;
  using const_reference = typename alloc_traits::const_reference;

  // TODO : build an iterator
  // using iterator       = details::random_iterator<pointer>;
  // using const_iterator = details::random_iterator<const_pointer>;

  GCXX_FH auto GetNewCapacity(size_type currentSize) -> size_type;

 protected:
  pointer begin_;
  pointer end_;
  pointer capacity_;
  allocator_type allocator_;

  GCXX_FH auto get_capacity_ptr() GCXX_NOEXCEPT - > pointer& {
    return capacity_;
  }

  GCXX_FH auto get_capacity_ptr() GCXX_CONST_NOEXCEPT -> pointer const& {
    return capacity_;
  }

  GCXX_FH auto get_allocator() GCXX_NOEXCEPT -> allocator_type_reference {
    return allocator_;
  }

  GCXX_FH auto get_allocator()
    GCXX_CONST_NOEXCEPT -> const_allocator_type_reference {
    return allocator_;
  }

 public:
  GCXX_FH VectorBase();
  GCXX_FH VectorBase(const allocator_type& allocator);
  GCXX_FH VectorBase(size_type n, const allocator_type& allocator);

  ~VectorBase();

  const_allocator_type_reference get_allocator() GCXX_CONST_NOEXCEPT;
  allocator_type_reference get_allocator() GCXX_NOEXCEPT;
  void set_allocator(const allocator_type& allocator);

 protected:
  pointer DoAllocate(size_type n);
  void DoFree(VT* p, size_type n);
};

GCXX_END_NAMESPACE


#endif
