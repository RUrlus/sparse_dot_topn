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
#include <sparse_dot_topn/eigen/common.hpp>

namespace sdtn::eigen {

namespace nb = nanobind;

namespace core {}  // namespace core

namespace api {

template <typename eT, typename idxT, int Order = Eigen::RowMajor>
SpMat<eT, idxT> sp_matmul(nb::handle& A_src, nb::handle& B_src) {
    SpMapMat<eT, idxT> A = to_eigen<eT, idxT>(A_src);
    SpMapMat<eT, idxT, Order> B = to_eigen<eT, idxT, Order>(B_src);
    return A * B;
}

}  // namespace api

namespace bindings {

void bind_sp_matmul_csr_csr(nb::module_& m);
void bind_sp_matmul_csr_csc(nb::module_& m);

}  // namespace bindings
}  // namespace sdtn::eigen
