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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <utility>
#include <vector>

#include <sparse_dot_topn/sp_matmul_topn.hpp>
#include <sparse_dot_topn/sp_matmul_topn_bindings.hpp>

namespace sdtn {

namespace nb = nanobind;

namespace api {

template <typename eT>
using nb_vec
    = nb::ndarray<nb::numpy, eT, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename eT>
inline nb_vec<eT> to_nbvec(std::vector<eT>&& seq) {
    std::vector<eT>* seq_ptr = new std::vector<eT>(std::move(seq));
    eT* data = seq_ptr->data();
    auto capsule = nb::capsule(seq_ptr, [](void* p) noexcept {
        delete reinterpret_cast<std::vector<eT>*>(p);
    });
    return nb_vec<eT>(data, {seq_ptr->size()}, capsule);
}

template <typename eT>
inline nb_vec<eT> to_nbvec(eT* data, size_t size) {
    auto capsule = nb::capsule(data, [](void* p) noexcept {
        delete[] reinterpret_cast<eT*>(p);
    });
    return nb_vec<eT>(data, {size}, capsule);
}

template <typename eT, typename idxT, core::iffInt<idxT> = true>
inline nb::tuple sp_matmul_topn(
    const idxT top_n,
    const idxT nrows,
    const idxT ncols,
    const eT threshold,
    const double density,
    const nb_vec<eT>& A_data,
    const nb_vec<idxT>& A_indptr,
    const nb_vec<idxT>& A_indices,
    const nb_vec<eT>& B_data,
    const nb_vec<idxT>& B_indptr,
    const nb_vec<idxT>& B_indices
) {
    auto pre_alloc_size = static_cast<int64_t>(ceil(density * top_n * nrows));
    std::vector<eT> C_data;
    C_data.reserve(pre_alloc_size);
    std::vector<idxT> C_indices;
    C_indices.reserve(pre_alloc_size);
    std::vector<idxT> C_indptr(nrows + 1);
    core::sp_matmul_topn<eT, idxT>(
        top_n,
        nrows,
        ncols,
        threshold,
        A_data.data(),
        A_indptr.data(),
        A_indices.data(),
        B_data.data(),
        B_indptr.data(),
        B_indices.data(),
        C_data,
        C_indptr,
        C_indices
    );
    return nb::make_tuple(
        to_nbvec<eT>(std::move(C_data)),
        to_nbvec<idxT>(std::move(C_indices)),
        to_nbvec<idxT>(std::move(C_indptr))
    );
}

#ifdef SDTN_OMP_ENABLED
template <typename eT, typename idxT, core::iffInt<idxT> = true>
inline nb::tuple sp_matmul_topn_mt(
    const idxT top_n,
    const idxT nrows,
    const idxT ncols,
    const eT threshold,
    const int n_threads,
    const nb_vec<eT>& A_data,
    const nb_vec<idxT>& A_indptr,
    const nb_vec<idxT>& A_indices,
    const nb_vec<eT>& B_data,
    const nb_vec<idxT>& B_indptr,
    const nb_vec<idxT>& B_indices
) {
    auto [total_nonzero, C_data, C_indices, C_indptr]
        = core::sp_matmul_topn_mt<eT, idxT>(
            top_n,
            nrows,
            ncols,
            threshold,
            n_threads,
            A_data.data(),
            A_indptr.data(),
            A_indices.data(),
            B_data.data(),
            B_indptr.data(),
            B_indices.data()
        );
    return nb::make_tuple(
        to_nbvec<eT>(C_data, total_nonzero),
        to_nbvec<idxT>(C_indices, total_nonzero),
        to_nbvec<idxT>(C_indptr, nrows + 1)
    );
}
#endif  // SDTN_OMP_ENABLED

}  // namespace api

namespace bindings {

void bind_sp_matmul_topn(nb::module_& m);
#ifdef SDTN_OMP_ENABLED
void bind_sp_matmul_topn_mt(nb::module_& m);
#endif  // SDTN_OMP_ENABLED
}  // namespace bindings
}  // namespace sdtn
