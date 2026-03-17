# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import functools
import ctypes
import os

import numba as nb
from numba import types as nbt
from numba import extending as nbext
from numba.core import cgutils
from numba.core.typing.templates import (
    AttributeTemplate, AttributeTemplate, signature)
from numba.core.datamodel.models import StructModel

import llvmlite.binding
from llvmlite import ir

import pyarrow as pa


def _arrow_type_to_nb_type(ty):
    if pa.types.is_signed_integer(ty):
        return nbt.Integer(str(ty), bitwidth=ty.bit_width, signed=True)
    if pa.types.is_boolean(ty):
        return nbt.boolean
    raise TypeError(f'Unsupported PyArrow type {ty}')

# PyArrow_PoolBufferPtr PyArrow_AllocatePoolBuffer(int64_t size);
# int32_t PyArrow_PoolBufferReserve(PyArrow_PoolBufferPtr buf, int64_t capacity);
# int32_t PyArrow_PoolBufferResizeExactly(PyArrow_PoolBufferPtr buf, int64_t size);
# void PyArrow_PoolBufferDestroy(PyArrow_PoolBufferPtr buf);
# int32_t PyArrow_MakeAndExportArray(int64_t length, int64_t null_count,
#                                    int32_t num_buffers,
#                                    PyArrow_PoolBufferPtr* buffer_ptrs, ArrowArray* out)
# void PyArrow_ReleaseArray(ArrowArray* array);


int64_t = ir.IntType(64)

pyarrow_cpp_funcs = {
    'PyArrow_AllocatePoolBuffer': ir.FunctionType(
        cgutils.voidptr_t,
        [int64_t],
    ),
    'PyArrow_PoolBufferReserve': ir.FunctionType(
        cgutils.int32_t, [cgutils.voidptr_t, int64_t]),
    'PyArrow_PoolBufferResizeExactly': ir.FunctionType(
        cgutils.int32_t, [cgutils.voidptr_t, int64_t]),
    'PyArrow_PoolBufferDestroy': ir.FunctionType(
        ir.VoidType(), [cgutils.voidptr_t]),
    'PyArrow_MakeAndExportArray': ir.FunctionType(
        cgutils.int32_t, [int64_t, int64_t, cgutils.int32_t,
                          cgutils.voidptr_t.as_pointer(), cgutils.voidptr_t]),
    'PyArrow_ReleaseArray': ir.FunctionType(ir.VoidType(), [cgutils.voidptr_t]),
}


@functools.cache
def get_pyarrow_dll():
    exc = None
    dll = None
    for path in pa.get_library_dirs():
        try:
            dll = ctypes.CDLL(os.path.join(path, 'libarrow_python.so'))
            break
        except OSError as e:
            exc = e
    if dll is None:
        raise exc
    # TODO use same idiom as Numba (see `_load_global_helpers`)
    for func_name in pyarrow_cpp_funcs:
        func = dll[func_name]
        func_addr = ctypes.cast(func, ctypes.c_void_p).value
        llvmlite.binding.add_symbol(func_name, func_addr)
    return dll


dll = get_pyarrow_dll()


def get_pyarrow_cpp_func(builder, func_name):
    fnty = pyarrow_cpp_funcs[func_name]
    fn = cgutils.get_or_insert_function(builder.module, fnty,
                                        func_name)
    return fn
