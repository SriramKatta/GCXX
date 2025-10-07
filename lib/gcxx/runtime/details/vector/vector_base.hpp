#pragma once
#ifndef GCXX_RUNTIME_DETAILS_VECTOR_VECTOR_BASE_HPP
#define GCXX_RUNTIME_DETAILS_VECTOR_VECTOR_BASE_HPP

#include <cstddef>
#include <gcxx/macros/define_macros.hpp>

#include <vector>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

/* VectorBase
The reason we have a VectorBase class is that it makes exception handling
simpler to implement because memory allocation is implemented entirely
in this class. If a user creates a vector which needs to allocate
memory in the constructor, VectorBase handles it. If an exception is thrown
by the allocator then the exception throw jumps back to the user code and
no try/catch code need be written in the vector or VectorBase constructor.
If an exception is thrown in the vector (not VectorBase) constructor, the
destructor for VectorBase will be called automatically (and free the
allocated memory) before the execution jumps back to the user code. However,
if the vector class were to handle both allocation and initialization then it
would have no choice but to implement an explicit try/catch statement for all
pathways that allocate memory. This increases code size and decreases
performance and makes the code a little harder read and maintain.

The C++ standard (15.2 paragraph 2) states:
  "An object that is partially constructed or partially destroyed will
  have destructors executed for all its fully constructed subobjects,
  that is, for subobjects for which the constructor has been completed
  execution and the destructor has not yet begun execution."

The C++ standard (15.3 paragraph 11) states:
  "The fully constructed base classes and members of an object shall
  be destroyed before entering the handler of a function-try-block
  of a constructor or destructor for that block."
*/
template <typename VT, typename Allocator_t>
class vector_base {
 public:
  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                      Member types                      █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  using value_type      = VT;
  using allocator_type  = Allocator_t;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference       = value_type&;
  using const_reference = const value_type&;
  using pointer = typename std::allocator_traits<allocator_type>::pointer;
  using const_pointer =
    typename std::allocator_traits<allocator_type>::const_pointer;
  // TODO : build an DEVICE iterator
  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                    Member Function                     █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

  // ==========================================================
  //                        Constructors
  // ==========================================================
  GCXX_FH vector_base();
  GCXX_FH vector_base(const allocator_type& allocator);
  GCXX_FH vector_base(size_type n, const allocator_type& allocator);

  // ==========================================================
  //                         destructor
  // ==========================================================

  virtual ~vector_base();
  // ==========================================================
  //                         operator=
  // ==========================================================


  // ==========================================================
  //                           assign
  // ==========================================================


  // ==========================================================
  //                       get_allocator
  // ==========================================================

  auto get_allocator() GCXX_CONST_NOEXCEPT -> allocator_type;

  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                     Element access                     █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  auto at(size_type pos) -> reference;
  auto at(size_type pos) const -> const_reference;

  auto operator[](size_type pos) -> reference;
  auto operator[](size_type pos) const -> const_reference;

  auto front() -> reference;
  auto front() const -> const_reference;

  auto back() -> reference;
  auto back() const -> const_reference;

  auto data() -> pointer;
  auto data() const -> const_pointer;

  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                       Iterators                        █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

  auto begin() -> iterator;
  auto begin() const -> const_iterator;
  auto cbegin() const noexcept -> const_iterator;


  auto end() -> iterator;
  auto end() const -> const_iterator;
  auto cend() const noexcept -> const_iterator;

  auto rbegin() -> iterator;
  auto rbegin() const -> const_iterator;
  auto crbegin() const noexcept -> const_iterator;


  auto rend() -> iterator;
  auto rend() const -> const_iterator;
  auto crend() const noexcept -> const_iterator;

  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                        Capacity                        █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  auto empty() const -> bool;
  auto size() const -> size_type;
  auto max_size() const -> size_type;
  auto reserve(size_type new_cap) const -> void;
  auto capacity() const -> size_type;
  auto shrink_to_fit() -> void;

  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                       Modifiers                        █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  auto clear() -> void;
  template <typename T>
  auto insert(const_iterator pos, const T& value) -> iterator;
  template <typename T>
  auto insert(const_iterator pos, T&& value) -> iterator;
  template <typename T>
  auto insert(const_iterator pos, size_type count, const T& value) -> iterator;

  template <class InputIt>
  auto insert(const_iterator pos, InputIt first, InputIt last) -> iterator;
  template <typename T>
  auto insert(const_iterator pos, std::initializer_list<T> ilist) -> iterator;

  template <typename R>
  auto insert_range(const_iterator pos, R&& rg) -> iterator;

  template <class... Args>
  auto emplace(const_iterator pos, Args&&... args) -> iterator;


  auto erase(const_iterator pos) -> iterator;
  auto erase(const_iterator first, const_iterator last) -> iterator;

  template <typename T>
  auto push_back(const T& value) -> void;
  template <typename T>
  auto push_back(T&& value) -> void;

  template <class... Args>
  auto emplace_back(Args&&... args) -> reference;

  auto pop_back() -> void;

  auto resize(size_type count) -> void;
  auto resize(size_type count, const value_type& value) -> void;
  auto swap(vector_base& other) noexcept -> void;

 protected:
  pointer begin_;
  pointer end_;
  pointer capacity_;
  allocator_type allocator_;
};

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif
