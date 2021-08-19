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

#include <new>
#include <type_traits>

namespace arrow {
namespace internal {

#if __cplusplus >= 201703L
using std::launder;
#else
template <class T>
constexpr T* launder(T* p) noexcept {
  return p;
}
#endif

template <typename T>
class AlignedStorage {
 public:
  T* get() { return launder(reinterpret_cast<T*>(&data_)); }
  constexpr const T* get() const { return launder(reinterpret_cast<const T*>(&data_)); }

  T& operator*() { return *get(); }
  constexpr const T& operator*() const { return *get(); }

  T* operator->() { return get(); }
  constexpr const T* operator->() const { return get(); }

  void Destroy() noexcept {
    if (!std::is_trivially_destructible<T>::value) {
      get()->~T();
    }
  }

  template <typename... A>
  void Construct(A&&... args) {
    new (&data_) T(std::forward<A>(args)...);
  }

  void MoveFrom(AlignedStorage* other) noexcept {
    static_assert(std::is_nothrow_move_constructible<T>::value, "");
    Construct(std::move(*other->get()));
    other->Destroy();
  }

 private:
  typename std::aligned_storage<sizeof(T), alignof(T)>::type data_;
};

}  // namespace internal
}  // namespace arrow
