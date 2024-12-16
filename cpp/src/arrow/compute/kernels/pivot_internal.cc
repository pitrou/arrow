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

#include "arrow/compute/kernels/pivot_internal.h"

#include <cstdint>

#include "arrow/compute/exec.h"
#include "arrow/compute/kernels/codegen_internal.h"
#include "arrow/scalar.h"
#include "arrow/type_traits.h"
#include "arrow/util/checked_cast.h"
#include "arrow/visit_type_inline.h"

namespace arrow::compute::internal {

using ::arrow::util::span;

struct BasePivotKeyMapper : public PivotKeyMapper {
  Status Init(const PivotOptions* options) override {
    if (options->key_names.size() > static_cast<size_t>(kMaxPivotKey) + 1) {
      return Status::NotImplemented("Pivoting to more than ",
                                    static_cast<size_t>(kMaxPivotKey) + 1,
                                    " columns: got ", options->key_names.size());
    }
    key_name_map_.reserve(options->key_names.size());
    PivotKeyIndex index = 0;
    for (const auto& key_name : options->key_names) {
      bool inserted =
          key_name_map_.try_emplace(std::string_view(key_name), index++).second;
      if (!inserted) {
        return Status::KeyError("Duplicate key name '", key_name, "' in PivotOptions");
      }
    }
    unexpected_key_behavior_ = options->unexpected_key_behavior;
    return Status::OK();
  }

 protected:
  Result<PivotKeyIndex> KeyNotFound(std::string_view key_name) {
    if (unexpected_key_behavior_ == PivotOptions::kIgnore) {
      return kNullPivotKey;
    }
    return Status::KeyError("Unexpected pivot key: ", key_name);
  }

  Result<PivotKeyIndex> LookupKey(std::string_view key_name) {
    const auto it = this->key_name_map_.find(key_name);
    if (ARROW_PREDICT_FALSE(it == this->key_name_map_.end())) {
      return KeyNotFound(key_name);
    } else {
      return it->second;
    }
  }

  static constexpr int kBatchLength = 512;
  // The strings backing the string_views should be kept alive by PivotOptions.
  std::unordered_map<std::string_view, PivotKeyIndex> key_name_map_;
  PivotOptions::UnexpectedKeyBehavior unexpected_key_behavior_;
  TypedBufferBuilder<PivotKeyIndex> key_indices_buffer_;
};

template <typename KeyType>
struct TypedPivotKeyMapper : public BasePivotKeyMapper {
  Result<span<const PivotKeyIndex>> MapKeys(const ArraySpan& array) override {
    RETURN_NOT_OK(this->key_indices_buffer_.Reserve(array.length));
    PivotKeyIndex* key_indices = this->key_indices_buffer_.mutable_data();
    int64_t i = 0;
    RETURN_NOT_OK(VisitArrayValuesInline<KeyType>(
        array,
        [&](std::string_view key_name) {
          ARROW_ASSIGN_OR_RAISE(key_indices[i], LookupKey(key_name));
          ++i;
          return Status::OK();
        },
        [&]() { return Status::KeyError("key name cannot be null"); }));
    return span(key_indices, array.length);
  }

  Result<PivotKeyIndex> MapKey(const Scalar& scalar) override {
    const auto& binary_scalar = checked_cast<const BaseBinaryScalar&>(scalar);
    return LookupKey(binary_scalar.view());
  }
};

struct PivotKeyMapperFactory {
  template <typename T>
  Status Visit(const T& key_type) {
    if constexpr (is_base_binary_like(T::type_id)) {
      instance = std::make_unique<TypedPivotKeyMapper<T>>();
      return instance->Init(options);
    }
    return Status::NotImplemented("Pivot key type: ", key_type);
  }

  const PivotOptions* options;
  std::unique_ptr<PivotKeyMapper> instance{};
};

Result<std::unique_ptr<PivotKeyMapper>> PivotKeyMapper::Make(
    const DataType& key_type, const PivotOptions* options) {
  PivotKeyMapperFactory factory{options};
  RETURN_NOT_OK(VisitTypeInline(key_type, &factory));
  return std::move(factory).instance;
}

/*
TODO
would probably like to write:

Result<std::unique_ptr<PivotKeyMapper>> PivotKeyMapper::Make(const DataType& key_type,
                                                             const PivotOptions* options)
{ std::unique_ptr<PivotKeyMapper> instance; RETURN_NOT_OK(VisitTypeInline(key_type,
[&](auto key_type) { using T = std::decay_t<decltype(key_type)>; if constexpr
(is_base_binary_like(T::type_id)) { instance = std::make_unique<TypedPivotKeyMapper<T>>();
      return instance->Init(options);
    }
    return Status::NotImplemented("Pivot key type: ", key_type);
  }));
  return instance;
}

or even:

Result<std::unique_ptr<PivotKeyMapper>> PivotKeyMapper::Make(const DataType& key_type,
                                                             const PivotOptions* options)
{ return VisitTypeInline(key_type, [&](auto key_type) ->
Result<std::unique_ptr<PivotKeyMapper>> { using T = std::decay_t<decltype(key_type)>; if
constexpr (is_base_binary_like(T::type_id)) { auto instance =
std::make_unique<TypedPivotKeyMapper<T>>(); RETURN_NOT_OK(instance->Init(options)); return
instance;
    }
    return Status::NotImplemented("Pivot key type: ", key_type);
  });
}
*/

}  // namespace arrow::compute::internal
