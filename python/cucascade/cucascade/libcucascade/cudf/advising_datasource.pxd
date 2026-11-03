# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf.io.datasource cimport datasource

from cucascade.exception_handler cimport cucascade_exception_handler


cdef extern from "cucascade/cudf/advising_datasource.hpp" namespace "cucascade::io" nogil:

    cdef cppclass advising_datasource(datasource):
        size_t size() except +cucascade_exception_handler const
