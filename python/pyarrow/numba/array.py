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
import operator

import numba as nb
from numba import types as nbt
from numba import extending as nbext
from numba.core import cgutils
from numba.core.typing.templates import (
    AttributeTemplate, AttributeTemplate, signature)
from numba.core.datamodel.models import StructModel

from llvmlite import ir

import pyarrow as pa
from ._utils import _arrow_type_to_nb_type, get_pyarrow_cpp_func

#
# 1. Typing
#


class PyArrowArrayType(nbt.Type):

    def __init__(self, type):
        self._type = type
        self._nb_type = _arrow_type_to_nb_type(type)
        super().__init__(name=f'PyArrowArray({type})')

    @property
    def arrow_type(self):
        return self._type

    @property
    def numba_type(self):
        return self._nb_type

    @property
    def value_type(self):
        return nb.optional(self._nb_type)


@nbext.typeof_impl.register(pa.Array)
def typeof_array(val, c):
    assert isinstance(val, pa.Array)
    return PyArrowArrayType(val.type)


# XXX useful?
# nbext.as_numba_type.register(pa.Int64Array, PyArrowArrayType(pa.int64()))

@nbext.infer_getattr
class PyArrowArrayAttribute(AttributeTemplate):
    key = PyArrowArrayType

    def resolve_length(self, typ):
        return nbt.int64

    def resolve_null_count(self, typ):
        return nbt.int64

    def resolve_offset(self, typ):
        return nbt.int64


#
# 2. Data model
#

@nbext.register_model(PyArrowArrayType)
class ArrayModel(nbext.models.StructModel):
    def __init__(self, dmm, fe_type):
        # XXX Does this respect the C ABI? Probably here, since everything is word-sized
        # XXX Should this instead carry a pointer to a heap-allocated struct?
        # Something else?
        members = [
            ('length', nbt.int64),
            ('null_count', nbt.int64),
            ('offset', nbt.int64),
            ('n_buffers', nbt.int64),
            ('n_children', nbt.int64),
            ('buffers', nbt.CPointer(nbt.CPointer(nbt.uint8))),
            ('children', nbt.CPointer(nbt.CPointer(nbt.uint8))),
            ('dictionary', nbt.CPointer(nbt.uint8)),
            ('release_cb', nbt.CPointer(nbt.uint8)),
            ('private_data', nbt.CPointer(nbt.uint8)),
        ]
        super().__init__(dmm, fe_type, members)


#
# 3. Code generation
#

@nbext.unbox(PyArrowArrayType)
def unbox_array(typ, obj, c):
    """
    Unbox a pa.Array object as a ArrowArray struct wrapper.
    """
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    c_array = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    with ExitStack() as stack:
        # NOTE: could use PyCapsule interface instead
        # TODO this method call is slow, should implement it as
        # a PyArrow native function
        array_ptr_obj = c.pyapi.long_from_unsigned_int(
            c.builder.ptrtoint(c_array._getpointer(), cgutils.intp_t))
        ret_obj = c.pyapi.call_method(obj, "_export_to_c", (array_ptr_obj,))
        c.pyapi.decref(array_ptr_obj)
        with cgutils.early_exit_if_null(c.builder, stack, ret_obj):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        c.pyapi.decref(ret_obj)

    def cleanup():
        c_array_ptr = c.builder.bitcast(c_array._getpointer(), cgutils.voidptr_t)
        fn = get_pyarrow_cpp_func(c.builder, 'PyArrow_ReleaseArray')
        c.builder.call(fn, [c_array_ptr])

    return nbext.NativeValue(c_array._getvalue(),
                             cleanup=cleanup,
                             is_error=c.builder.load(is_error_ptr))


@nbext.box(PyArrowArrayType)
def box_array(typ, val, c):
    """
    Box a ArrowArray struct as a pa.Array object.
    """
    c_array = c.context.make_helper(c.builder, typ, value=val)

    with ExitStack() as stack:
        # TODO this sequence of calls through Python is slow, should implement
        # it as a PyArrow native function
        pa_name = c.context.insert_const_string(c.builder.module, 'pyarrow')
        pa_mod = c.pyapi.import_module(pa_name)
        array_cls = c.pyapi.object_getattr_string(pa_mod, 'Array')
        # FIXME use proper type
        datatype_obj = c.pyapi.call_method(pa_mod, 'int64')
        c.pyapi.decref(pa_mod)
        array_ptr_obj = c.pyapi.long_from_unsigned_int(
            c.builder.ptrtoint(c_array._getpointer(), cgutils.intp_t))
        ret_obj = c.pyapi.call_method(
            array_cls, "_import_from_c", (array_ptr_obj, datatype_obj))
        c.pyapi.decref(array_cls)
        c.pyapi.decref(array_ptr_obj)
        c.pyapi.decref(datatype_obj)
        return ret_obj


def make_array(typ):
    """
    Return the Structure representation of the given array type.
    (an instance of types.PyArrowArrayType).
    """
    # NOTE see numba/np/arrayobj.py to perhaps add annotations for speed
    assert isinstance(typ, PyArrowArrayType)
    return cgutils.create_struct_proxy(typ)


# NOTE This + infer_getattr is the same as
#   nbext.make_attribute_wrapper(PyArrowArrayType, 'null_count', 'null_count')


@nbext.lower_getattr(PyArrowArrayType, 'null_count')
def array_null_count(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    return array.null_count


@nbext.lower_getattr(PyArrowArrayType, 'length')
def array_length(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    return array.length


@nbext.lower_getattr(PyArrowArrayType, 'offset')
def array_offset(context, builder, typ, value):
    arrayty = make_array(typ)
    array = arrayty(context, builder, value)
    return array.offset


@nbext.intrinsic
def ll_array_is_null(typingctx, arr_type, index_type):
    assert isinstance(arr_type, PyArrowArrayType)
    result_type = nbt.bool_
    # Force index to int64
    sig = result_type(arr_type, nbt.int64)

    def codegen(context, builder, signature, args):
        (arr_val, index) = args
        arrayty = make_array(arr_type)
        array = arrayty(context, builder, arr_val)

        null_bitmap = builder.load(cgutils.gep(
            builder, array.buffers, 0, inbounds=True))
        has_null_bitmap = builder.icmp_unsigned(
            '!=', array.null_count, array.null_count.type(0))

        with builder.if_else(has_null_bitmap) as (then_block, else_block):
            with then_block:
                # FIXME add offset
                byte_offset = builder.lshr(index, index.type(3))
                bit_offset = builder.and_(index, index.type(7))
                bitmap_byte = builder.load(cgutils.gep(
                    builder, null_bitmap, byte_offset))
                mask = builder.shl(cgutils.int8_t(
                    1), builder.trunc(bit_offset, cgutils.int8_t))
                masked_byte = builder.and_(bitmap_byte, mask)
                is_null_with_bitmap = builder.icmp_unsigned(
                    '==', masked_byte, masked_byte.type(0))
                bb_then = builder.basic_block
            with else_block:
                is_null_without_bitmap = cgutils.false_bit
                bb_else = builder.basic_block

        is_null = builder.phi(is_null_with_bitmap.type)
        is_null.add_incoming(is_null_with_bitmap, bb_then)
        is_null.add_incoming(is_null_without_bitmap, bb_else)
        return is_null

    return sig, codegen


def nb_type_to_ll_type(ty):
    if ty in nbt.integer_domain:
        return ir.IntType(ty.bitwidth)
    raise TypeError(f'Unsupported type {ty}')


@nbext.intrinsic
def ll_array_get_value(typingctx, arr_type, index_type):
    assert isinstance(arr_type, PyArrowArrayType)
    result_type = arr_type.numba_type
    if result_type != nbt.int64:
        # TODO other types
        return
    # Force index to int64
    sig = result_type(arr_type, nbt.int64)

    def codegen(context, builder, signature, args):
        (arr_val, index) = args
        arrayty = make_array(arr_type)
        array = arrayty(context, builder, arr_val)

        data_buffer = builder.load(cgutils.gep(
            builder, array.buffers, 1, inbounds=True))
        data_type = result_type
        data_buffer = builder.bitcast(
            data_buffer, nb_type_to_ll_type(data_type).as_pointer())
        # FIXME add offset
        data_ptr = cgutils.gep(builder, data_buffer, index)
        return builder.load(data_ptr)

    return sig, codegen


@nbext.overload(operator.getitem)
def array_getitem(arr, index):
    if not isinstance(arr, PyArrowArrayType):
        return
    if not index in nbt.integer_domain:
        raise nbt.TypingError("array indices must be integers")

    def getitem_impl(arr, index):
        if ll_array_is_null(arr, index):
            return None
        else:
            return ll_array_get_value(arr, index)

    return getitem_impl


@nbext.overload(len)
def array_len(arr):
    if not isinstance(arr, PyArrowArrayType):
        return

    def len_impl(arr):
        return arr.length

    return len_impl
