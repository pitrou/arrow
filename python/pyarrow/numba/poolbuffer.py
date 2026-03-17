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

from contextlib import ExitStack
import functools
import operator
import os

import numba as nb
from numba import types as nbt
from numba import extending as nbext
from numba.core import cgutils
from numba.core.typing.templates import (
    AttributeTemplate, AttributeTemplate, infer_getattr, signature, bound_function)
from numba.core.datamodel.models import StructModel

import llvmlite.binding
from llvmlite import ir

import pyarrow as pa
from ._utils import _arrow_type_to_nb_type, get_pyarrow_cpp_func, int64_t

#
# 0. Pure Python
#


class PoolBuffer:
    pass


# XXX rename to BufferBuilder for conciseness?
class TypedBufferBuilder:
    def __init__(self, value_type):
        if not pa.types.is_integer(value_type) and not pa.types.is_boolean(value_type):
            raise TypeError(f'Cannot create TypedBufferBuilder for {value_type}')
        self.value_type = value_type


class Int64BufferBuilder:
    def __init__(self):
        super().__init__(pa.int64())


class BooleanBufferBuilder:
    def __init__(self):
        super().__init__(pa.bool_())


#
# 1. Typing
#


class PoolBufferType(nbt.Type):

    def __init__(self):
        super().__init__(name=f'PoolBuffer')


# XXX only needed for pure Python
@nbext.typeof_impl.register(PoolBuffer)
def typeof_buffer(val, c):
    assert isinstance(val, PoolBuffer)
    return PoolBufferType()


class TypedBufferBuilderType(nbt.Type):

    def __init__(self, value_type):
        self._value_type = value_type
        self._nb_type = _arrow_type_to_nb_type(self._value_type)
        super().__init__(name=f'TypedBufferBuilder({self._value_type})')

    @property
    def arrow_type(self):
        return self._value_type

    @property
    def numba_type(self):
        return self._nb_type


# XXX only needed for pure Python
@nbext.typeof_impl.register(TypedBufferBuilder)
def typeof_buffer(val, c):
    assert isinstance(val, TypedBufferBuilder)
    return TypedBufferBuilderType(val.value_type)


# TODO rework this once PyArrow types such as Int64Type are Numba-typed
@nbext.type_callable(Int64BufferBuilder)
def type_int64_buffer_builder(context):
    def typer():
        return TypedBufferBuilderType(pa.int64())
    return typer


@nbext.type_callable(BooleanBufferBuilder)
def type_boolean_buffer_builder(context):
    def typer():
        return TypedBufferBuilderType(pa.bool_())
    return typer


@infer_getattr
class TypedBufferBuilderAttribute(AttributeTemplate):
    key = TypedBufferBuilderType

    @bound_function("TypedBufferBuilder.reserve")
    def resolve_reserve(self, buffer_builder, args, kws):
        if kws or len(args) != 1:
            return
        return signature(nbt.none, nbt.int64)

    @bound_function("TypedBufferBuilder.append")
    def resolve_append(self, buffer_builder, args, kws):
        if kws or len(args) != 1:
            return
        assert not kws
        return signature(nbt.none, buffer_builder.numba_type)

    @bound_function("TypedBufferBuilder.finish")
    def resolve_finish(self, buffer_builder, args, kws):
        if args or kws:
            return
        return signature(PoolBufferType())


#
# 2. Data model
#


@nbext.register_model(PoolBufferType)
class PoolBufferModel(nbext.models.StructModel):
    cpp_ptr_type = nbt.RawPointer('PyArrow_PoolBufferPtr')

    def __init__(self, dmm, fe_type):
        members = [
            ('cpp_ptr', self.cpp_ptr_type),
        ]
        super().__init__(dmm, fe_type, members)


@nbext.register_model(TypedBufferBuilderType)
class TypedBufferBuilderModel(nbext.models.StructModel):
    data_ptr_type = nbt.RawPointer('uint8_t*')
    # XXX: for booleans, perhaps the bit mask and byte index can be stored separately

    def __init__(self, dmm, fe_type):
        members = [
            ('pool_buffer', PoolBufferType()),
            ('index_ptr', nbt.EphemeralPointer(nbt.int64)),
            ('capacity_ptr', nbt.EphemeralPointer(nbt.int64)),
            ('data_ptr_ptr', nbt.EphemeralPointer(self.data_ptr_type)),
        ]
        super().__init__(dmm, fe_type, members)


#
# 3. Code generation
#

# FIXME get this dynamically
_pool_buffer_offsetof_data = 16
_pool_buffer_offsetof_size = 24
_pool_buffer_offsetof_capacity = 32


def pool_buffer_allocate_ll(context, builder, result_type):
    pool_buffer = context.make_helper(builder, result_type)
    fn = get_pyarrow_cpp_func(builder, 'PyArrow_AllocatePoolBuffer')
    hardcoded_size = 0  # FIXME
    pool_buffer.cpp_ptr = builder.call(fn, [int64_t(hardcoded_size)])
    # TODO handle errors
    return pool_buffer._getvalue()


def _pool_buffer_reserve_or_resize_ll(context, builder, func_name, pool_buffer_ty,
                                      pool_buffer, size_or_capacity):
    pool_buffer = context.make_helper(builder, pool_buffer_ty, value=pool_buffer)

    fn = get_pyarrow_cpp_func(builder, func_name)
    # TODO handle errors
    builder.call(fn, [pool_buffer.cpp_ptr, size_or_capacity])


def pool_buffer_reserve_ll(context, builder, pool_buffer_ty, pool_buffer, capacity):
    return _pool_buffer_reserve_or_resize_ll(context, builder, 'PyArrow_PoolBufferReserve',
                                             pool_buffer_ty, pool_buffer, capacity)


def pool_buffer_resize_exactly_ll(context, builder, pool_buffer_ty, pool_buffer, capacity):
    return _pool_buffer_reserve_or_resize_ll(context, builder, 'PyArrow_PoolBufferResizeExactly',
                                             pool_buffer_ty, pool_buffer, capacity)


@nbext.intrinsic
def pool_buffer_ctor_impl(typingctx):
    result_type = PoolBufferType()
    sig = result_type()

    def codegen(context, builder, signature, args):
        return pool_buffer_allocate_ll(context, builder, result_type)

    return sig, codegen


@nbext.intrinsic
def pool_buffer_reserve_impl(typingctx, pool_buffer_type, capacity_type):
    assert isinstance(pool_buffer_type, PoolBufferType)
    sig = nbt.none(pool_buffer_type, nbt.int64)

    def codegen(context, builder, signature, args):
        # TODO handle errors
        return pool_buffer_reserve_ll(context, builder, pool_buffer_type, args[0], *args[1:])

    return sig, codegen


def pool_buffer_get_properties(context, builder, pool_buffer_ty, pool_buffer):
    pool_buffer = context.make_helper(builder, pool_buffer_ty, value=pool_buffer)

    size_ptr = builder.inttoptr(
        builder.add(
            builder.ptrtoint(pool_buffer.cpp_ptr, int64_t),
            int64_t(_pool_buffer_offsetof_size)), int64_t.as_pointer())
    capa_ptr = builder.inttoptr(
        builder.add(
            builder.ptrtoint(pool_buffer.cpp_ptr, int64_t),
            int64_t(_pool_buffer_offsetof_capacity)), int64_t.as_pointer())
    data_ptr = builder.inttoptr(
        builder.add(
            builder.ptrtoint(pool_buffer.cpp_ptr, int64_t),
            int64_t(_pool_buffer_offsetof_data)), cgutils.voidptr_t.as_pointer())

    return builder.load(size_ptr), builder.load(capa_ptr), builder.load(data_ptr)


@nbext.intrinsic
def pool_buffer_get_size(typingctx, pool_buffer_ty):
    result_type = nbt.int64
    sig = result_type(pool_buffer_ty)

    def codegen(context, builder, signature, args):
        size, capa, data = pool_buffer_get_properties(
            context, builder, pool_buffer_ty, args[0])
        return size

    return sig, codegen


@nbext.intrinsic
def pool_buffer_get_capa(typingctx, pool_buffer_ty):
    result_type = nbt.int64
    sig = result_type(pool_buffer_ty)

    def codegen(context, builder, signature, args):
        size, capa, data = pool_buffer_get_properties(
            context, builder, pool_buffer_ty, args[0])
        return capa

    return sig, codegen


@nbext.intrinsic
def pool_buffer_get_data(typingctx, pool_buffer_ty):
    result_type = nbt.RawPointer('uint8_t*')
    sig = result_type(pool_buffer_ty)

    def codegen(context, builder, signature, args):
        size, capa, data = pool_buffer_get_properties(
            context, builder, pool_buffer_ty, args[0])
        return data

    return sig, codegen


@nbext.overload(PoolBuffer)
def pool_buffer_ctor():
    def impl():
        return pool_buffer_ctor_impl()
    return impl


@nbext.overload_attribute(PoolBufferType, 'size')
def pool_buffer_size(pool_buffer):
    def get(pool_buffer):
        return pool_buffer_get_size(pool_buffer)
    return get


@nbext.overload_attribute(PoolBufferType, 'capacity')
def pool_buffer_size(pool_buffer):
    def get(pool_buffer):
        return pool_buffer_get_capa(pool_buffer)
    return get


@nbext.overload_attribute(PoolBufferType, 'data')
def pool_buffer_data(pool_buffer):
    def get(pool_buffer):
        return pool_buffer_get_data(pool_buffer)
    return get


@nbext.overload_method(PoolBufferType, 'reserve')
def pool_buffer_reserve(pool_buffer, capacity):
    if capacity in nbt.integer_domain:
        def impl(pool_buffer, capacity):
            return pool_buffer_reserve_impl(pool_buffer, nbt.int64(capacity))
        return impl


def get_item_capacity(context, builder, item_type, byte_capacity):
    if pa.types.is_boolean(item_type):
        return builder.mul(byte_capacity, byte_capacity.type(8))
    else:
        return builder.udiv(byte_capacity, byte_capacity.type(item_type.byte_width))


def get_byte_capacity(context, builder, item_type, item_capacity):
    if pa.types.is_boolean(item_type):
        # `(item_capacity + 7) / 8`
        return builder.udiv(builder.add(item_capacity, item_capacity.type(7)),
                            item_capacity.type(8))
    else:
        return builder.mul(item_capacity, item_capacity.type(item_type.byte_width))


def typed_buffer_builder_ctor_ll(context, builder, buffer_builder_type):
    value_type = buffer_builder_type.arrow_type

    # TODO: for booleans, should ensure the buffer is zeroed
    buffer_builder = context.make_helper(builder, buffer_builder_type)
    buffer_builder.pool_buffer = pool_buffer_allocate_ll(
        context, builder, PoolBufferType())
    zero = context.get_constant(nbt.int64, 0)
    buffer_builder.index_ptr = cgutils.alloca_once_value(builder, zero)
    _, capa, data = pool_buffer_get_properties(
        context, builder, PoolBufferType(), buffer_builder.pool_buffer)

    buffer_builder.capacity_ptr = cgutils.alloca_once_value(
        builder, get_item_capacity(context, builder, value_type, capa))
    buffer_builder.data_ptr_ptr = cgutils.alloca_once_value(builder, data)
    return buffer_builder._getvalue()


@nbext.lower_builtin(Int64BufferBuilder)
@nbext.lower_builtin(BooleanBufferBuilder)
# XXX high-level extending API + @intrinsic doesn't work for this
# because of EphemeralPointer, report Numba bug?
def typed_buffer_builder_ctor(context, builder, sig, args):
    return typed_buffer_builder_ctor_ll(context, builder, sig.return_type)


def typed_buffer_builder_finish_ll(context, builder, buffer_builder_type, buffer_builder):
    value_type = buffer_builder_type.arrow_type
    buffer_builder = context.make_helper(
        builder, buffer_builder_type, value=buffer_builder)
    final_size = builder.load(buffer_builder.index_ptr)
    final_byte_size = get_byte_capacity(context, builder, value_type, final_size)
    pool_buffer_resize_exactly_ll(
        context, builder, PoolBufferType(), buffer_builder.pool_buffer, final_byte_size)
    return buffer_builder.pool_buffer


@nbext.lower_builtin('TypedBufferBuilder.finish', TypedBufferBuilderType)
def typed_buffer_builder_finish(context, builder, sig, args):
    return typed_buffer_builder_finish_ll(context, builder, sig.args[0], args[0])


@nbext.lower_builtin('TypedBufferBuilder.reserve', TypedBufferBuilderType, nbt.int64)
def typed_buffer_builder_reserve(context, builder, sig, args):
    buffer_builder_type = sig.args[0]
    value_type = buffer_builder_type.arrow_type

    buffer_builder = context.make_helper(builder, sig.args[0], value=args[0])
    current_size = builder.load(buffer_builder.index_ptr)
    additional_capa = args[1]
    new_capa = builder.add(current_size, additional_capa)

    # XXX should factor this out (see append below)
    new_byte_capacity = get_byte_capacity(context, builder, value_type, new_capa)
    pool_buffer_reserve_ll(
        context, builder, PoolBufferType(), buffer_builder.pool_buffer, new_byte_capacity)
    _, new_capa, new_data = pool_buffer_get_properties(
        context, builder, PoolBufferType(), buffer_builder.pool_buffer)
    builder.store(get_item_capacity(context, builder, value_type,
                  new_capa), buffer_builder.capacity_ptr)
    builder.store(new_data, buffer_builder.data_ptr_ptr)


@nbext.lower_builtin('TypedBufferBuilder.append', TypedBufferBuilderType, nbt.Any)
def typed_buffer_builder_append(context, builder, sig, args):
    buffer_builder_type = sig.args[0]
    value_type = buffer_builder_type.arrow_type

    buffer_builder = context.make_helper(builder, sig.args[0], value=args[0])
    data_value = args[1]
    sizeof_value = context.get_abi_sizeof(data_value.type)

    index = builder.load(buffer_builder.index_ptr)
    index_plus_one = builder.add(index, index.type(1))
    capa = builder.load(buffer_builder.capacity_ptr)

    needed_capa = index_plus_one
    not_enough_capa = builder.icmp_signed('<', capa, needed_capa)
    with builder.if_then(not_enough_capa, likely=False) as then_block:
        # XXX check growth heuristic
        new_capa = builder.mul(needed_capa, capa.type(2))
        new_byte_capacity = get_byte_capacity(context, builder, value_type, new_capa)
        # TODO: for booleans, should ensure the additional buffer bytes are zeroed
        pool_buffer_reserve_ll(
            context, builder, PoolBufferType(), buffer_builder.pool_buffer, new_byte_capacity)
        _, new_capa, new_data = pool_buffer_get_properties(
            context, builder, PoolBufferType(), buffer_builder.pool_buffer)
        builder.store(get_item_capacity(context, builder, value_type,
                      new_capa), buffer_builder.capacity_ptr)
        builder.store(new_data, buffer_builder.data_ptr_ptr)

    data = builder.load(buffer_builder.data_ptr_ptr)
    # TODO make data_ptr_ptr the right pointer type?
    if pa.types.is_boolean(value_type):
        byte_index = builder.udiv(index, index.type(8))
        bit_index = builder.urem(index, index.type(8))
        mask = builder.shl(cgutils.int8_t(1), builder.trunc(bit_index, cgutils.int8_t))
        flipped_mask = builder.not_(mask)
        byte_ptr = builder.gep(data, [byte_index])
        byte = builder.load(byte_ptr)
        byte = builder.select(data_value, builder.or_(byte, mask),
                              builder.and_(byte, flipped_mask))
        builder.store(byte, byte_ptr)
    else:
        data = builder.bitcast(data, data_value.type.as_pointer())
        append_ptr = builder.gep(data, [index])
        builder.store(data_value, append_ptr)
    builder.store(index_plus_one, buffer_builder.index_ptr)


# @functools.cache
# def typed_buffer_builder_ctor_impl(value_type):
#     result_type = TypedBufferBuilderType(value_type)
#     sig = result_type()
#
#     @nbext.intrinsic
#     def impl(typingctx):
#         def codegen(context, builder, signature, args):
#             buffer_builder = context.make_helper(builder, result_type)
#             buffer_builder.pool_buffer = pool_buffer_allocate_ll(
#                 context, builder, PoolBufferType())
#             zero = context.get_constant(nbt.int64, 0)
#             buffer_builder.index_ptr = cgutils.alloca_once_value(builder, zero)
#             return buffer_builder._getvalue()
#         return sig, codegen
#
#     return impl
#
#
# @nbext.overload(Int64BufferBuilder)
# def int64_buffer_builder_ctor():
#     wrapped_impl = typed_buffer_builder_ctor_impl(pa.int64())
#
#     def impl():
#         return wrapped_impl()
#     return impl
