// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "arrow/util/launder.h"
#include "arrow/util/macros.h"
#include "arrow/util/make_unique.h"
#include "arrow/util/span.h"

namespace arrow {
namespace internal {

template <typename Derived, bool TriviallyDestructible>
class ConditionallyTriviallyDestructible {
 protected:
  ~ConditionallyTriviallyDestructible() { static_cast<Derived*>(this)->destroy(); }
};
template <typename Derived>
class ConditionallyTriviallyDestructible<Derived, true> {};

template <typename T, size_t N>
struct StaticVectorStorage
    : ConditionallyTriviallyDestructible<StaticVectorStorage<T, N>,
                                         std::is_trivially_destructible<T>::value> {
  std::array<AlignedStorage<T>, N> static_data_;
  size_t size_ = 0;

  StaticVectorStorage() noexcept = default;

  AlignedStorage<T>* data_ptr() { return static_data_.data(); }
  constexpr const AlignedStorage<T>* const_data_ptr() const {
    return static_data_.data();
  }

  util::span<AlignedStorage<T>> data_span() { return {data_ptr(), size_}; }

  constexpr size_t capacity() const { return N; }

  constexpr size_t max_size() const { return N; }

  struct ResizeInProgress {
    util::span<AlignedStorage<T>> from, to;

    util::span<AlignedStorage<T>> uninitialized() const {
      return to.subspan(from.size());
    }
    static constexpr bool to_new_allocation() { return false; }
  };

  ResizeInProgress increase_size(size_t new_size, ...) {
    assert(new_size <= N);

    ResizeInProgress out;
    out.from = {data_ptr(), size_};
    out.to = {data_ptr(), new_size};

    size_ = new_size;
    return out;
  }

  void destroy() {
    for (size_t i = 0; i < size_; ++i) {
      static_data_[i].Destroy();
    }

    size_ = 0;
  }

  // Move objects from another storage, but don't destroy any objects currently
  // stored in *this. You need to call destroy() first if necessary (e.g. in a
  // move assignment operator).
  void move_from(StaticVectorStorage* other) noexcept {
    for (size_t i = 0; i < other->size_; ++i) {
      static_data_[i].MoveFrom(&other->static_data_[i]);
    }

    size_ = other->size_;
    other->size_ = 0;
  }
};

template <typename T, size_t N>
struct SmallVectorStorage {
  std::array<AlignedStorage<T>, N> static_data_;
  size_t size_ = 0;
  AlignedStorage<T>* data_ = static_data_.data();
  size_t dynamic_capacity_ = 0;

  SmallVectorStorage() noexcept = default;

  ~SmallVectorStorage() { destroy(); }

  AlignedStorage<T>* data_ptr() { return data_; }
  constexpr const AlignedStorage<T>* const_data_ptr() const { return data_; }

  util::span<AlignedStorage<T>> data_span() { return {data_ptr(), size_}; }

  constexpr size_t capacity() const { return dynamic_capacity_ ? dynamic_capacity_ : N; }

  constexpr size_t max_size() const { return std::numeric_limits<size_t>::max(); }

  void destroy() {
    for (size_t i = 0; i < size_; ++i) {
      data_[i].Destroy();
    }

    if (dynamic_capacity_) {
      delete[] data_;
    }

    size_ = 0;
    dynamic_capacity_ = 0;
  }

  // Move objects from another storage, but don't destroy any objects currently
  // stored in *this. You need to call destroy() first if necessary (e.g. in a
  // move assignment operator).
  void move_from(SmallVectorStorage* other) noexcept {
    if (other->dynamic_capacity_) {
      data_ = other->data_;
      other->data_ = NULLPTR;

      dynamic_capacity_ = other->dynamic_capacity_;
      other->dynamic_capacity_ = 0;
    } else {
      for (size_t i = 0; i < other->size_; ++i) {
        data_[i].MoveFrom(&other->data_[i]);
      }
    }

    size_ = other->size_;
    other->size_ = 0;
  }

  struct ResizeInProgress {
    util::span<AlignedStorage<T>> from, to;

    util::span<AlignedStorage<T>> uninitialized() const {
      return to.subspan(from.size());
    }
    bool to_new_allocation() const { return from.data() != to.data(); }

    // ensure old data_ is deleted after the resize is complete
    std::unique_ptr<AlignedStorage<T>[]>
        scoped_delete;  // NOLINT modernize-avoid-c-arrays
  };

  ResizeInProgress increase_size(size_t new_size, bool move_active = true) {
    assert(new_size > size_);

    ResizeInProgress out;
    out.from = {data_, size_};

    if (new_size > N && new_size > dynamic_capacity_) {
      // need to allocate more storage
      if (dynamic_capacity_) {
        out.scoped_delete.reset(data_);
      }

      dynamic_capacity_ = std::max(dynamic_capacity_ * 2, new_size);
      data_ = new AlignedStorage<T>[dynamic_capacity_];

      if (move_active) {
        size_t i = 0;
        for (auto& from : out.from) {
          data_[i++].MoveFrom(&from);
        }
      }
    }

    size_ = new_size;
    out.to = {data_, size_};
    return out;
  }

 private:
  void set_dynamic_capacity(size_t new_capacity) {
    assert(new_capacity >= size_);

    // NOLINTNEXTLINE modernize-avoid-c-arrays
    std::unique_ptr<AlignedStorage<T>[]> scoped_delete;
    if (dynamic_capacity_) {
      // delete old data_ at the close of this scope
      scoped_delete.reset(data_);
    }

    // NOLINTNEXTLINE modernize-avoid-c-arrays
    auto new_data = internal::make_unique<AlignedStorage<T>[]>(new_capacity);
    for (size_t i = 0; i < size_; ++i) {
      new_data[i].MoveFrom(&data_[i]);
    }

    data_ = new_data.release();
    dynamic_capacity_ = new_capacity;
  }
};

template <typename T, size_t N, typename Storage>
class StaticVectorImpl {
 public:
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;

  constexpr StaticVectorImpl() noexcept = default;

  // Move and copy constructors
  StaticVectorImpl(StaticVectorImpl&& other) noexcept {
    storage_.move_from(&other.storage_);
  }

  StaticVectorImpl& operator=(StaticVectorImpl&& other) noexcept {
    storage_.destroy();
    storage_.move_from(&other.storage_);
    return *this;
  }

  StaticVectorImpl(const StaticVectorImpl& other) { assign(other.begin(), other.end()); }

  StaticVectorImpl& operator=(const StaticVectorImpl& other) noexcept {
    if (&other == this) {
      return *this;
    }
    assign(other.begin(), other.end());
    return *this;
  }

  // Automatic conversion from std::vector<T>, for convenience
  StaticVectorImpl(const std::vector<T>& other) {  // NOLINT: explicit
    assign(other.begin(), other.end());
  }

  StaticVectorImpl& operator=(const std::vector<T>& other) {
    assign(other.begin(), other.end());
    return *this;
  }

  StaticVectorImpl(std::vector<T>&& other) noexcept {  // NOLINT: explicit
    assign(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
  }

  StaticVectorImpl& operator=(std::vector<T>&& other) noexcept {
    assign(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
    return *this;
  }

  // Constructing from count and optional initialization value
  explicit StaticVectorImpl(size_t count) { resize(count); }

  StaticVectorImpl(size_t count, const T& value) { resize(count, value); }

  StaticVectorImpl(std::initializer_list<T> values) {
    assign(values.begin(), values.end());
  }

  constexpr bool empty() const { return storage_.size_ == 0; }

  constexpr size_t size() const { return storage_.size_; }

  constexpr size_t capacity() const { return storage_.capacity(); }

  constexpr size_t max_size() const { return storage_.max_size(); }

  T& operator[](size_t i) { return data()[i]; }
  constexpr const T& operator[](size_t i) const { return data()[i]; }

  T& front() { return data()[0]; }
  constexpr const T& front() const { return data()[0]; }

  T& back() { return data()[storage_.size_ - 1]; }
  constexpr const T& back() const { return data()[storage_.size_ - 1]; }

  T* data() { return storage_.data_ptr()->get(); }
  constexpr const T* data() const { return storage_.const_data_ptr()->get(); }

  iterator begin() { return iterator(data()); }
  constexpr const_iterator begin() const { return data(); }

  iterator end() { return iterator(data() + storage_.size_); }
  constexpr const_iterator end() const { return data() + storage_.size_; }

  void reserve(size_t n) {
    if (n <= capacity()) return;
    size_t old_size = size();
    storage_.increase_size(n);
    storage_.size_ = old_size;
  }

  void clear() { storage_.destroy(); }

  void push_back(const T& value) { emplace_back(value); }

  void push_back(T&& value) { emplace_back(std::move(value)); }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    auto& back = storage_.increase_size(size() + 1).uninitialized()[0];
    back.Construct(std::forward<Args>(args)...);
  }

  template <typename ForwardIt>
  iterator insert(const_iterator insert_at, ForwardIt first, ForwardIt last) {
    const auto pos = insert_at - data();
    assert(pos >= 0 && static_cast<size_t>(pos) <= size());

    if (first == last) {
      return data() + pos;
    }

    auto num_displaced = size() - pos;
    auto num_inserted = std::distance(first, last);

    auto resized = storage_.increase_size(size() + num_inserted, /*move_active=*/false);

    if (resized.to_new_allocation()) {
      // even elements before pos must be moved
      size_t i = 0;
      for (auto& from : resized.from.first(pos)) {
        resized.to[i++].MoveFrom(&from);
      }
    }

    // move displaced elements in reverse to make a gap for insertion
    auto from_reversed = resized.from.rbegin();
    auto to_reversed = resized.to.rbegin();
    for (size_t i = 0; i < num_displaced; ++i) {
      to_reversed[i].MoveFrom(&from_reversed[i]);
    }

    // insert into the gap from the provided iterator
    size_t i = pos;
    while (first != last) {
      resized.to[i++].Construct(*first++);
    }

    return data() + pos;
  }

  template <typename ForwardIt>
  void assign(ForwardIt begin, ForwardIt end) {
    auto n = static_cast<size_t>(std::distance(begin, end));

    if (n <= size()) {
      decrease_size(n);
      for (auto& slot : storage_.data_span()) {
        *slot = *begin++;
      }
      return;
    }

    auto resized = storage_.increase_size(n);

    for (auto& slot : resized.to.first(resized.from.size())) {
      *slot = *begin++;
    }
    for (auto& slot : resized.uninitialized()) {
      slot.Construct(*begin++);
    }
  }

  void resize(size_t n) { do_resize(n); }

  void resize(size_t n, const T& value) { do_resize(n, value); }

 private:
  template <typename... Args>
  void do_resize(size_t n, const Args&... args) {
    if (n <= size()) {
      return decrease_size(n);
    }

    auto resized = storage_.increase_size(n);

    for (auto& slot : resized.uninitialized()) {
      slot.Construct(args...);
    }
  }

  void decrease_size(size_t n) {
    assert(n <= size());

    for (auto& slot : storage_.data_span().subspan(n)) {
      slot.Destroy();
    }

    storage_.size_ = n;
  }

  Storage storage_;
};

template <typename T, size_t N>
using StaticVector = StaticVectorImpl<T, N, StaticVectorStorage<T, N>>;

template <typename T, size_t N>
using SmallVector = StaticVectorImpl<T, N, SmallVectorStorage<T, N>>;

}  // namespace internal
}  // namespace arrow
