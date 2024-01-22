# sparse\_dot\_topn

-------------------------------**WARNING**-------------------------------

Version 1.0 introduces major and potentially breaking changes to the API.

Please see the Migrating section below.

-------------------------------**WARNING**-------------------------------

**sparse\_dot\_topn** provides a fast way to performing a sparse matrix multiplication followed by top-n multiplication result selection.

Comparing very large feature vectors and picking the best matches, in practice often results in performing a sparse matrix multiplication followed by selecting the top-n multiplication results.

**sparse\_dot\_topn** provides a (parallelised) sparse matrix multiplication implementation that integrates selecting the top-n values, resulting in a significantly lower memory footprint and improved performance.

## Usage

```python
import scipy.sparse as sparse
from sparse_dot_topn import sp_matmul_topn

A = sparse.random(1000, 100, density=0.1, format="csr")
B = sparse.random(100, 2000, density=0.1, format="csr")

C = sp_matmul_topn(A, B, top_n=10)
```

`sp_matmul_topn` supports `{CSR, CSC, COO}` matrices with `{32, 64}bit {int, float}` data.
Note that `COO` and `CSC` inputs are converted to the `CSR` format and are therefore slower.
Two options to further reduce memory requirements are `threshold` and `density`.
Optionally, the values can be sorted such that the first column for a given row contains the largest value.
Note that `sp_matmul_topn(A, B, top_n=B.shape[1])` is equal to `A.dot(B)`.

## Installation

**sparse\_dot\_topn** provides wheels for CPython 3.8 to 3.12 for:

* Windows (64bit)
* Linux (64bit)
* MacOS (x86 and ARM)

```shell
pip install sparse_dot_topn
```

**sparse\_dot\_topn** relies on a C++ extension for the computationally intensive multiplication routine.
Note that the wheels vendor/ships OpenMP with the extension to provide parallelisation out-of-the-box.
If you run into issues with OpenMP see INSTALLATION.md for help.

Installing from source requires a C++17 compatible compiler.
If you have a compiler available it is advised to install without the wheel as this enables architecture specific optimisations.

You can install from source using:

```shell
pip install sparse_dot_topn --no-binary sparse_dot_topn
```

### Build configuration

**sparse\_dot\_topn** provides some configuration options when building from source.
Building from source can enable architecture specific optimisations and is recommended for those that have a C++ compiler installed.
See INSTALLATION.md for details.

## Migrating to v1.

**sparse\_dot\_topn** v1 is a significant change from `v0.*` with a new bindings and API.

**`awesome_cossim_topn` has been deprecated and will be removed in a future version.**

Users should switch to `sp_matmul_topn` which is largely compatible:

For example:

```python
C = awesome_cossim_topn(A, B, ntop=10)
```

can be replicated using:

```python
C = sp_matmul_topn(A, B, top_n=10, threshold=0.0, sort=True)
```

### API changes

1. `ntop` has been renamed to `topn`
2. `lower_bound` has been renamed to `threshold`
3. `use_threads` and `n_jobs` have been combined into `n_threads`
4. `return_best_ntop` option has been removed
5. `test_nnz_max` option has been removed
6. `B` is auto-transposed when its shape is not compatible but its transpose is.

The output of `return_best_ntop` can be replicated with:

```python
C = sp_matmul_topn(A, B, top_n=10)
best_ntop = np.diff(C.indptr).max()
```

### Default changes

1. `threshold` no longer `0.0` but disabled by default

This enables proper functioning for matrices that contain negative values.
Additionally a different data-structure is used internally when collecting non-zero results that has a much lower memory-footprint than previously.
This means that the effect of the `threshold` parameter on performance and memory requirements is negligible. 

2. `sort = False`, the result matrix is no longer sorted by default

The matrix is returned with the same column order as if not filtering of the top-n results has taken place.
This means that when you set `top_n` equal to the number of columns of `B` you obtain the same result as normal multiplication,
i.e. `sp_matmul_topn(A, B, top_n=B.shape[1])` is equal to `A.dot(B)`.

## Contributing

Contributions are very welcome, please see CONTRIBUTING for details.

### Contributors

This package was developed and is maintained by authors (previously) affiliated with ING Analytics Wholesale Banking Advanced Analytics.
The original implementation was based on modified version of Scipy's CSR multiplication implementation.
You can read about it in a [blog](https://medium.com/@ingwbaa/https-medium-com-ingwbaa-boosting-selection-of-the-most-similar-entities-in-large-scale-datasets-450b3242e618) [(mirror)](https://www.sun-analytics.nl/posts/2017-07-26-boosting-selection-of-most-similar-entities-in-large-scale-datasets/) written by Zhe Sun.

* [Zhe Sun](https://github.com/ymwdalex/)
* [Ahmet Erdem](https://github.com/aerdem4)
* [Stephane Collet](https://github.com/stephanecollot)
* [Particular Miner](https://github.com/ParticularMiner) (no ING affiliation)
* [Ralph Urlus](https://github.com/RUrlus)

