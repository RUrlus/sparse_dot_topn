/* Copyright (c) 2023 ING Analytics Wholesale Banking
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <Eigen/SparseCore>

#include <limits>
#include <utility>
#include <vector>

#include <sparse_dot_topn/common.hpp>
#include <sparse_dot_topn/eigen/common.hpp>
#include <sparse_dot_topn/maxheap.hpp>

namespace sdtn::eigen {

namespace nb = nanobind;

namespace core {}  // namespace core

namespace api {

template <typename eT, typename idxT>
nb::tuple sp_matmul_topn_scalar(
    nb::handle& A_src, nb::handle& B_src, const int top_n
) {
    SpMapMat<eT, idxT> A = to_eigen<eT, idxT>(A_src);
    SpColMapMat<eT, idxT> B = to_eigen<eT, idxT, Eigen::ColMajor>(B_src);

    const idxT nrows = A.rows();
    const idxT ncols = B.cols();
    const idxT result_size = static_cast<idxT>(top_n * nrows);

    std::vector<eT> C_data;
    C_data.reserve(result_size);
    std::vector<idxT> C_indices;
    C_indices.reserve(result_size);
    std::vector<idxT> C_indptr(nrows + 1);

    auto max_heap
        = sdtn::core::MaxHeap<eT, idxT>(top_n, std::numeric_limits<eT>::min());
    idxT nnz = 0;
    for (idxT i = 0; i < nrows; ++i) {
        eT min = max_heap.reset();
        for (idxT j = 0; j < B.cols(); ++j) {
            eT val = A.row(i).dot(B.col(j));
            if (val > min) {
                min = max_heap.push_pop(j, val);
            }
        }
        max_heap.insertion_sort();
        int n_set = max_heap.get_n_set();
        for (int ii = 0; ii < n_set; ++ii) {
            C_indices.push_back(max_heap.heap[ii].idx);
            C_data.push_back(max_heap.heap[ii].val);
        }
        nnz += n_set;
        C_indptr[i + 1] = nnz;
    }
    return nb::make_tuple(
        nb::make_tuple(
            to_nbvec<eT>(std::move(C_data)),
            to_nbvec<idxT>(std::move(C_indices)),
            to_nbvec<idxT>(std::move(C_indptr))
        ),
        nb::make_tuple(nrows, ncols)
    );
}

template <typename eT, typename idxT>
nb::tuple sp_matmul_topn_blocks(
    nb::handle& A_src, nb::handle& B_src, const int top_n, const int blocksize
) {
    SpMapMat<eT, idxT> A = to_eigen<eT, idxT>(A_src);
    SpColMapMat<eT, idxT> B = to_eigen<eT, idxT, Eigen::ColMajor>(B_src);
    using spVec = Eigen::SparseVector<eT>;

    const idxT nrows = A.rows();
    const idxT ncols = B.cols();
    const idxT result_size = static_cast<idxT>(top_n * nrows);

    std::vector<eT> C_data;
    C_data.reserve(result_size);
    std::vector<idxT> C_indices;
    C_indices.reserve(result_size);
    std::vector<idxT> C_indptr(nrows + 1);

    const int n_blocks = ncols / blocksize;
    const idxT trailing_cols = ncols - (n_blocks * blocksize);
    const bool trailing = trailing_cols != 0;

    auto max_heap
        = sdtn::core::MaxHeap<eT, idxT>(top_n, std::numeric_limits<eT>::min());
    idxT nnz = 0;
    for (idxT i = 0; i < nrows; ++i) {
        eT min = max_heap.reset();
        int lb = 0;
        for (int j = 0; j < n_blocks; ++j) {
            spVec block = A.row(i) * B.innerVectors(lb, blocksize);
            for (typename spVec::InnerIterator it(block); it; ++it) {
                eT val = it.value();
                if (val > min) {
                    min = max_heap.push_pop(it.index(), val);
                }
            }
            lb += blocksize;
        }
        if (trailing) {
            spVec block = A.row(i) * B.rightCols(trailing_cols);
            for (typename spVec::InnerIterator it(block); it; ++it) {
                eT val = it.value();
                if (val > min) {
                    min = max_heap.push_pop(it.index(), val);
                }
            }
        }
        max_heap.insertion_sort();
        int n_set = max_heap.get_n_set();
        for (int ii = 0; ii < n_set; ++ii) {
            C_indices.push_back(max_heap.heap[ii].idx);
            C_data.push_back(max_heap.heap[ii].val);
        }
        nnz += n_set;
        C_indptr[i + 1] = nnz;
    }
    return nb::make_tuple(
        nb::make_tuple(
            to_nbvec<eT>(std::move(C_data)),
            to_nbvec<idxT>(std::move(C_indices)),
            to_nbvec<idxT>(std::move(C_indptr))
        ),
        nb::make_tuple(nrows, ncols)
    );
}

}  // namespace api

namespace bindings {
void bind_sp_matmul_topn_scalar(nb::module_& m);
void bind_sp_matmul_topn_blocks(nb::module_& m);
}  // namespace bindings
//
}  // namespace sdtn::eigen
