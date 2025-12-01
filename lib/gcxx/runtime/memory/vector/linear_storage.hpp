#pragma once
#ifndef GCXX_RUNTIME_VECTOR_LINEAR_STORAGE_HPP_
#define GCXX_RUNTIME_VECTOR_LINEAR_STORAGE_HPP_

#include <gcxx/macros/define_macros.hpp>


#include <memory>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

struct copy_allocator_t
{};

template <typename T, typename Alloc>
class linear_storage
{
private:
  using alloc_traits = std::allocator_traits<Alloc>;

public:
  using allocator_type  = Alloc;
  using value_type      = T;
  using pointer         = typename alloc_traits::pointer;
  using const_pointer   = typename alloc_traits::const_pointer;
  using size_type       = typename alloc_traits::size_type;
  using difference_type = typename alloc_traits::difference_type;
  using reference       = typename alloc_traits::reference;
  using const_reference = typename alloc_traits::const_reference;

  using iterator       = T*; // TODO implement an device vector
  using const_iterator = const T*;

  
  GCXX_HD explicit linear_storage(const allocator_type& alloc = allocator_type());

  
  GCXX_HD explicit linear_storage(size_type n, const allocator_type& alloc = allocator_type());

  
  GCXX_HD explicit linear_storage(copy_allocator_t, const linear_storage& other);

  
  GCXX_HD explicit linear_storage(copy_allocator_t, const linear_storage& other, size_type n);

  linear_storage& operator=(const linear_storage& x) = delete;

  
  GCXX_HD ~linear_storage();

  GCXX_HD size_type size() const;

  GCXX_HD size_type max_size() const;

  GCXX_HD pointer data();

  GCXX_HD const_pointer data() const;

  GCXX_HD iterator begin();

  GCXX_HD const_iterator begin() const;

  GCXX_HD iterator end();

  GCXX_HD const_iterator end() const;

  GCXX_HD reference operator[](size_type n);

  GCXX_HD const_reference operator[](size_type n) const;

  GCXX_HD allocator_type get_allocator() const;

  // note that allocate does *not* automatically call deallocate
  GCXX_HD void allocate(size_type n);

  GCXX_HD void deallocate() noexcept;

  
  GCXX_HD void swap(linear_storage& other)
  {
    using std::swap;
    swap(m_begin, other.m_begin);
    swap(m_size, other.m_size);

    // From C++ standard [container.reqmts]
    //   If allocator_traits<allocator_type>::propagate_on_container_swap::value is true, then allocator_type
    //   shall meet the Cpp17Swappable requirements and the allocators of a and b shall also be exchanged by calling
    //   swap as described in [swappable.requirements]. Otherwise, the allocators shall not be swapped, and the behavior
    //   is undefined unless a.get_allocator() == b.get_allocator().
    if constexpr (alloc_traits::propagate_on_container_swap::value)
    {
      swap(m_allocator, other.m_allocator);
    }
    else if constexpr (!alloc_traits::is_always_equal::value)
    {
    //   NV_IF_TARGET(NV_IS_DEVICE, (assert(m_allocator == other.m_allocator);), (if (m_allocator != other.m_allocator) {
    //                  throw allocator_mismatch_on_swap();
    //                }));
    }
  }

  GCXX_HD void value_initialize_n(iterator first, size_type n);

  GCXX_HD void uninitialized_fill_n(iterator first, size_type n, const value_type& value);

  template <typename InputIterator>
  GCXX_HD iterator uninitialized_copy(InputIterator first, InputIterator last, iterator result);

//   template <typename System, typename InputIterator>
//   GCXX_HD iterator uninitialized_copy(
//     thrust::execution_policy<System>& from_system, InputIterator first, InputIterator last, iterator result);

//   template <typename InputIterator, typename Size>
//   GCXX_HD iterator uninitialized_copy_n(InputIterator first, Size n, iterator result);

//   template <typename System, typename InputIterator, typename Size>
//   GCXX_HD iterator
//   uninitialized_copy_n(thrust::execution_policy<System>& from_system, InputIterator first, Size n, iterator result);

  GCXX_HD void destroy(iterator first, iterator last) noexcept;

  
  GCXX_HD void deallocate_on_allocator_mismatch(const linear_storage& other) noexcept
  {
    if constexpr (alloc_traits::propagate_on_container_copy_assignment::value)
    {
      if (m_allocator != other.m_allocator)
      {
        deallocate();
      }
    }
  }

  
  GCXX_HD void destroy_on_allocator_mismatch(
    const linear_storage& other, [[maybe_unused]] iterator first, [[maybe_unused]] iterator last) noexcept
  {
    if constexpr (alloc_traits::propagate_on_container_copy_assignment::value)
    {
      if (m_allocator != other.m_allocator)
      {
        destroy(first, last);
      }
    }
  }

  GCXX_HD void set_allocator(const allocator_type& alloc);

  
  GCXX_HD void propagate_allocator(const linear_storage& other)
  {
    if constexpr (alloc_traits::propagate_on_container_copy_assignment::value)
    {
      m_allocator = other.m_allocator;
    }
  }

  
  GCXX_HD void propagate_allocator(linear_storage& other)
  {
    if constexpr (alloc_traits::propagate_on_container_move_assignment::value)
    {
      m_allocator = std::move(other.m_allocator);
    }
  }

  // allow move assignment for a sane implementation of allocator propagation
  GCXX_HD linear_storage& operator=(linear_storage&& other);

//   _CCCL_SYNTHESIZE_SEQUENCE_ACCESS(linear_storage, const_iterator)

private:
  // XXX we could inherit from this to take advantage of empty base class optimization
  allocator_type m_allocator;

  iterator m_begin;

  size_type m_size;

  friend GCXX_HD void swap(linear_storage& lhs, linear_storage& rhs) noexcept(noexcept(lhs.swap(rhs)))
  {
    lhs.swap(rhs);
  }
}; // end linear_storage


GCXX_NAMESPACE_MAIN_DETAILS_END


#include <gcxx/macros/undefine_macros.hpp>

#endif