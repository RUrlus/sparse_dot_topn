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

#include <sparse_dot_topn/common.hpp>

namespace sdtn::eigen {

namespace nb = nanobind;

template <typename eT, typename idxT>
using SpMat = Eigen::SparseMatrix<eT, Eigen::RowMajor>;

template <typename eT, typename idxT, int Order = Eigen::RowMajor>
using SpMapMat = Eigen::MappedSparseMatrix<eT, Order, idxT>;

template <typename eT, typename idxT>
using SpColMat = Eigen::SparseMatrix<eT, Eigen::ColMajor>;

template <typename eT, typename idxT>
using SpColMapMat = Eigen::MappedSparseMatrix<eT, Eigen::ColMajor, idxT>;

namespace api {

using namespace sdtn::api;

template <
    typename eT,
    typename idxT,
    int Order = Eigen::RowMajor,
    typename ReturnT = Eigen::MappedSparseMatrix<eT, Order, idxT>>
ReturnT to_eigen(nb::handle& src) {
    using ScalarCaster = nanobind::detail::make_caster<nb_vec<eT>>;
    using StorageIndexCaster = nanobind::detail::make_caster<nb_vec<idxT>>;

    ScalarCaster data_caster;
    StorageIndexCaster indices_caster;
    StorageIndexCaster indptr_caster;

    nb::object obj = borrow(src);
    nb::object data_o = obj.attr("data");
    data_caster.from_python(data_o, 0, nullptr);
    sdtn::api::nb_vec<eT>& values = data_caster.value;
    nb::object indices_o = obj.attr("indices");
    indices_caster.from_python(indices_o, 0, nullptr);
    sdtn::api::nb_vec<idxT>& inner_indices = indices_caster.value;
    nb::object indptr_o = obj.attr("indptr");
    indptr_caster.from_python(indptr_o, 0, nullptr);
    sdtn::api::nb_vec<idxT>& outer_indices = indptr_caster.value;
    nb::object shape_o = obj.attr("shape");

    eT* data_ptr = values.data();
    idxT* outer_indices_ptr = outer_indices.data();
    idxT* inner_indices_ptr = inner_indices.data();

    return ReturnT(
        nb::cast<idxT>(shape_o[0]),       // rows
        nb::cast<idxT>(shape_o[1]),       // cols
        nb::cast<idxT>(obj.attr("nnz")),  // nnz
        outer_indices_ptr,                // outerIndex
        inner_indices_ptr,                // innerIndex
        data_ptr                          // values
    );
}  // to_eigen

}  // namespace api

}  // namespace sdtn::eigen
