# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from pylibcudf.io.datasource cimport Datasource
from pylibcudf.libcudf.io.datasource cimport datasource

from cucascade.libcucascade.cudf.advising_datasource cimport advising_datasource

__all__ = ["AdvisingDatasource"]


cdef class AdvisingDatasource(Datasource):
    """
    An object-store object read through cuCascade's prefetching cache.

    Every sizeable read of it is advised to the cache, under the cache's column
    admission policy, before it runs. It is a :class:`pylibcudf.io.datasource.Datasource`,
    so any pylibcudf reader can read it.

    Not constructible directly: get them from :meth:`ScanCache.open`. A datasource
    keeps the cache that opened it alive.
    """
    def __init__(self):
        raise ValueError("Get an AdvisingDatasource from ScanCache.open.")

    @staticmethod
    cdef AdvisingDatasource from_libcucascade(
        unique_ptr[datasource] source, object owner
    ):
        cdef AdvisingDatasource result = AdvisingDatasource.__new__(AdvisingDatasource)
        result.c_obj = move(source)
        result.owner = owner
        return result

    cdef datasource* get_datasource(self) except * nogil:
        return self.c_obj.get()

    cpdef size_t nbytes(self):
        """
        Size of the object.

        Returns
        -------
        int
            Size in bytes.
        """
        return (<advising_datasource*> self.c_obj.get()).size()
