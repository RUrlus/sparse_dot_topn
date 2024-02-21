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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <sparse_dot_topn/eigen/sp_matmul_topn.hpp>

namespace sdtn::eigen::bindings {
namespace nb = nanobind;

using namespace nb::literals;

void bind_sp_matmul_topn_scalar(nb::module_& m) {
    m.def(
        "eigen_sp_matmul_topn_scalar",
        &api::sp_matmul_topn_scalar<double, int>,
        "A"_a.noconvert(),
        "B"_a.noconvert(),
        "top_n"_a,
        nb::raw_doc("Compute sparse dot product and keep top n.\n"
                    "\n"
                    "Args:\n"
                    "    A (scipy.sparse.csr_matrix): LHS of product\n"
                    "    B (scipy.sparse.csr_matrix): RHS of product\n"
                    "    top_n (int): the number of results to retain\n"
                    "\n"
                    "Returns:\n"
                    "    C (scipy.sparse.csr_matrix): the result of A.dot(B)\n"
                    "\n")
    );
    m.def(
        "eigen_sp_matmul_topn_scalar",
        &api::sp_matmul_topn_scalar<float, int>,
        "A"_a.noconvert(),
        "B"_a.noconvert(),
        "top_n"_a
    );
    m.def(
        "eigen_sp_matmul_topn_scalar",
        &api::sp_matmul_topn_scalar<double, int64_t>,
        "A"_a.noconvert(),
        "B"_a.noconvert(),
        "top_n"_a
    );
    m.def(
        "eigen_sp_matmul_topn_scalar",
        &api::sp_matmul_topn_scalar<float, int64_t>,
        "A"_a.noconvert(),
        "B"_a.noconvert(),
        "top_n"_a
    );
    m.def(
        "eigen_sp_matmul_topn_scalar",
        &api::sp_matmul_topn_scalar<int64_t, int>,
        "A"_a.noconvert(),
        "B"_a.noconvert(),
        "top_n"_a
    );
    m.def(
        "eigen_sp_matmul_topn_scalar",
        &api::sp_matmul_topn_scalar<int, int>,
        "A"_a.noconvert(),
        "B"_a.noconvert(),
        "top_n"_a
    );
    m.def(
        "eigen_sp_matmul_topn_scalar",
        &api::sp_matmul_topn_scalar<int64_t, int64_t>,
        "A"_a.noconvert(),
        "B"_a.noconvert(),
        "top_n"_a
    );
    m.def(
        "eigen_sp_matmul_topn_scalar",
        &api::sp_matmul_topn_scalar<int, int64_t>,
        "A"_a.noconvert(),
        "B"_a.noconvert(),
        "top_n"_a
    );
}  // bind_sp_matmul_scalar

}  // namespace sdtn::eigen::bindings
