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
#include <sparse_dot_topn/eigen/sp_matmul.hpp>

namespace sdtn::eigen::bindings {
namespace nb = nanobind;

using namespace nb::literals;

void bind_sp_matmul_csr_csr(nb::module_& m) {
    m.def(
        "eigen_sp_matmul_csr_csr",
        &api::sp_matmul<double, int>,
        "A"_a.noconvert(),
        "B"_a.noconvert(),
        nb::raw_doc("Compute sparse A.dot(B).\n"
                    "\n"
                    "Args:\n"
                    "    A (scipy.sparse.csr_matrix): LHS of product\n"
                    "    B (scipy.sparse.csr_matrix): RHS of product\n"
                    "\n"
                    "Returns:\n"
                    "    C (scipy.sparse.csr_matrix): the result of A.dot(B)\n"
                    "\n")
    );
    m.def(
        "eigen_sp_matmul_csr_csr",
        &api::sp_matmul<float, int>,
        "A"_a.noconvert(),
        "B"_a.noconvert(),
        nb::raw_doc("Compute sparse A.dot(B).\n"
                    "\n"
                    "Args:\n"
                    "    A (scipy.sparse.csr_matrix): LHS of product\n"
                    "    B (scipy.sparse.csr_matrix): RHS of product\n"
                    "\n"
                    "Returns:\n"
                    "    C (scipy.sparse.csr_matrix): the result of A.dot(B)\n"
                    "\n")
    );
    m.def(
        "eigen_sp_matmul_csr_csr",
        &api::sp_matmul<double, int64_t>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
    m.def(
        "eigen_sp_matmul_csr_csr",
        &api::sp_matmul<float, int64_t>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
    m.def(
        "eigen_sp_matmul_csr_csr",
        &api::sp_matmul<int64_t, int>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
    m.def(
        "eigen_sp_matmul_csr_csr",
        &api::sp_matmul<int, int>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
    m.def(
        "eigen_sp_matmul_csr_csr",
        &api::sp_matmul<int64_t, int64_t>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
    m.def(
        "eigen_sp_matmul_csr_csr",
        &api::sp_matmul<int, int64_t>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
}

void bind_sp_matmul_csr_csc(nb::module_& m) {
    m.def(
        "eigen_sp_matmul_csr_csc",
        &api::sp_matmul<double, int, Eigen::ColMajor>,
        "A"_a.noconvert(),
        "B"_a.noconvert(),
        nb::raw_doc("Compute sparse A.dot(B).\n"
                    "\n"
                    "Args:\n"
                    "    A (scipy.sparse.csr_matrix): LHS of product\n"
                    "    B (scipy.sparse.csc_matrix): RHS of product\n"
                    "\n"
                    "Returns:\n"
                    "    C (scipy.sparse.csr_matrix): the result of A.dot(B)\n"
                    "\n")
    );
    m.def(
        "eigen_sp_matmul_csr_csc",
        &api::sp_matmul<float, int, Eigen::ColMajor>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
    m.def(
        "eigen_sp_matmul_csr_csc",
        &api::sp_matmul<double, int64_t, Eigen::ColMajor>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
    m.def(
        "eigen_sp_matmul_csr_csc",
        &api::sp_matmul<float, int64_t, Eigen::ColMajor>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
    m.def(
        "eigen_sp_matmul_csr_csc",
        &api::sp_matmul<int64_t, int, Eigen::ColMajor>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
    m.def(
        "eigen_sp_matmul_csr_csc",
        &api::sp_matmul<int, int, Eigen::ColMajor>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
    m.def(
        "eigen_sp_matmul_csr_csc",
        &api::sp_matmul<int64_t, int64_t, Eigen::ColMajor>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
    m.def(
        "eigen_sp_matmul_csr_csc",
        &api::sp_matmul<int, int64_t, Eigen::ColMajor>,
        "A"_a.noconvert(),
        "B"_a.noconvert()
    );
}

}  // namespace sdtn::eigen::bindings
