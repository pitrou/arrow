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

#include <cstdint>

#include "arrow/c/abi.h"
#include "arrow/python/platform.h"
#include "arrow/python/visibility.h"

extern "C" {

struct PyArrow_PoolBuffer;
typedef PyArrow_PoolBuffer* PyArrow_PoolBufferPtr;

struct PyArrow_PoolBufferMeta {
  int32_t offset_of_data;
  int32_t offset_of_size;
  int32_t offset_of_capacity;
};

ARROW_PYTHON_EXPORT
void PyArrow_GetPoolBufferMeta(PyArrow_PoolBufferMeta* out);

// FIXME: these functions raise a Python exception on error
ARROW_PYTHON_EXPORT
PyArrow_PoolBufferPtr PyArrow_AllocatePoolBuffer(int64_t size);

ARROW_PYTHON_EXPORT
int32_t PyArrow_PoolBufferReserve(PyArrow_PoolBufferPtr buf, int64_t capacity);

ARROW_PYTHON_EXPORT
int32_t PyArrow_PoolBufferResizeExactly(PyArrow_PoolBufferPtr buf, int64_t size);

ARROW_PYTHON_EXPORT
void PyArrow_PoolBufferDestroy(PyArrow_PoolBufferPtr buf);

ARROW_PYTHON_EXPORT
int32_t PyArrow_MakeAndExportArray(int64_t length, int64_t null_count,
                                   int32_t num_buffers, PyArrow_PoolBufferPtr* buffers,
                                   ArrowArray* out);

ARROW_PYTHON_EXPORT
void PyArrow_ReleaseArray(ArrowArray* array);

}  // extern "C"
