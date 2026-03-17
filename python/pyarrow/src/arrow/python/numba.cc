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

#include "arrow/python/numba.h"

#include "arrow/array/data.h"
#include "arrow/array/util.h"
#include "arrow/buffer.h"
#include "arrow/c/bridge.h"
#include "arrow/c/helpers.h"
#include "arrow/status.h"
#include "arrow/util/logging.h"

#include "arrow/python/common.h"
#include "arrow/python/pyarrow.h"

namespace arrow::py::internal {
namespace {

static_assert(sizeof(std::unique_ptr<ResizableBuffer>) == sizeof(ResizableBuffer*));

class DummyBuffer : public ResizableBuffer {
 public:
  DummyBuffer() : ResizableBuffer(nullptr, 0) {}

  Status Resize(const int64_t new_size, bool shrink_to_fit) override {
    return Status::OK();
  }

  Status Reserve(const int64_t new_capacity) override { return Status::OK(); }

  PyArrow_PoolBufferMeta GetInstanceMeta() const {
    // HACK because we are not able to call offsetof on a protected member field.
#define CUSTOM_OFFSETOF(_member)                     \
  reinterpret_cast<const uint8_t*>(&this->_member) - \
      reinterpret_cast<const uint8_t*>(this)

    return PyArrow_PoolBufferMeta{
        .offset_of_data = CUSTOM_OFFSETOF(data_),
        .offset_of_size = CUSTOM_OFFSETOF(size_),
        .offset_of_capacity = CUSTOM_OFFSETOF(capacity_),
    };

#undef CUSTOM_OFFSETOF
  }

  static PyArrow_PoolBufferMeta GetMeta() {
    static const auto meta = DummyBuffer().GetInstanceMeta();
    return meta;
  }
};

}  // namespace
}  // namespace arrow::py::internal

void PyArrow_GetPoolBufferMeta(PyArrow_PoolBufferMeta* out) {
  using namespace arrow;
  *out = py::internal::DummyBuffer::GetMeta();
}

PyArrow_PoolBufferPtr PyArrow_AllocatePoolBuffer(int64_t size) {
  using namespace arrow;
  // Will raise an exception and returning a null unique_ptr on error
  auto buffer = py::GetResultValue(AllocateResizableBuffer(size));
  return reinterpret_cast<PyArrow_PoolBufferPtr>(buffer.release());
}

int32_t PyArrow_PoolBufferReserve(PyArrow_PoolBufferPtr buf_ptr, int64_t capacity) {
  using namespace arrow;
  auto buf = reinterpret_cast<ResizableBuffer*>(buf_ptr);
  return py::internal::check_status(buf->Reserve(capacity));
}

int32_t PyArrow_PoolBufferResizeExactly(PyArrow_PoolBufferPtr buf_ptr, int64_t size) {
  using namespace arrow;
  auto buf = reinterpret_cast<ResizableBuffer*>(buf_ptr);
  return py::internal::check_status(buf->Resize(size, /*shrink_to_fit=*/true));
}

void PyArrow_PoolBufferDestroy(PyArrow_PoolBufferPtr buf_ptr) {
  using namespace arrow;
  auto to_destroy =
      std::unique_ptr<ResizableBuffer>(reinterpret_cast<ResizableBuffer*>(buf_ptr));
  ARROW_UNUSED(to_destroy);
}

int32_t PyArrow_MakeAndExportArray(int64_t length, int64_t null_count,
                                   int32_t num_buffers,
                                   PyArrow_PoolBufferPtr* buffer_ptrs, ArrowArray* out) {
  using namespace arrow;
  BufferVector buffers(num_buffers);
  for (int32_t i = 0; i < num_buffers; ++i) {
    if (buffer_ptrs[i] != nullptr) {
      auto buf = reinterpret_cast<ResizableBuffer*>(buffer_ptrs[i]);
      buffers[i] = std::shared_ptr(std::unique_ptr<ResizableBuffer>(buf));
    }
  }
  // FIXME need a way to pass the proper type
  auto type = int64();
  auto data = ArrayData::Make(type, length, std::move(buffers), null_count);
  return py::internal::check_status(ExportArray(std::move(data), out));
}

void PyArrow_ReleaseArray(ArrowArray* array) { ArrowArrayRelease(array); }

const auto PyArrow_gg = []() {
  PyArrow_PoolBufferMeta meta;
  PyArrow_GetPoolBufferMeta(&meta);
  ARROW_LOG(INFO) << "offset_of_data = " << meta.offset_of_data;
  ARROW_LOG(INFO) << "offset_of_size = " << meta.offset_of_size;
  ARROW_LOG(INFO) << "offset_of_capacity = " << meta.offset_of_capacity;
  return true;
}();

// FIXME: need to do this because this is currently loaded independently using ctypes
const bool PyArrowLoaded = []() {
  ARROW_CHECK_EQ(arrow::py::import_pyarrow(), 0);
  return true;
}();
