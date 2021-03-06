# CFASTPT
C version of FAST-PT
-- Xiao Fang

The official python version of FAST-PT is hosted at [https://github.com/JoeMcEwen/FAST-PT](https://github.com/JoeMcEwen/FAST-PT).

Our papers ([JCAP 2016, 9, 15](https://iopscience.iop.org/article/10.1088/1475-7516/2016/09/015); [arXiv:1603.04826](https://arxiv.org/abs/1603.04826)) and ([JCAP 2017, 2, 30](https://iopscience.iop.org/article/10.1088/1475-7516/2017/02/030); [arXiv:1609.05978](https://arxiv.org/abs/1609.05978)) describe the FAST-PT algorithm and implementation. Please cite these papers when using FAST-PT in your research.

For the nonlinear galaxy bias implementation, cite [JCAP 2016, 9, 15](https://iopscience.iop.org/article/10.1088/1475-7516/2016/09/015) ([arXiv:1603.04826](https://arxiv.org/abs/1603.04826)).

For the intrinsic alignment implementation, cite [JCAP 2017, 2, 30](https://iopscience.iop.org/article/10.1088/1475-7516/2017/02/030) ([arXiv:1609.05978](https://arxiv.org/abs/1609.05978)) and [PRD, 100, 103506](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.103506) ([arXiv:1708.09247](https://arxiv.org/abs/1708.09247)).

Available Modules:
* nonlinear galaxy bias terms: `Pd1d2`, `Pd2d2`, `Pd1s2`, `Pd2s2`, `Ps2s2`, `Pd1d3nl`
* nonlinear (tidal alignment & tidal torquing) intrinsic galaxy alignment terms: `IA_tt`, `IA_ta`, `IA_mix`

Make sure you have FFTW Library installed.
Test Run:
```shell
make home
./a.out
```
or
```shell
gcc cfastpt.c utils.c utils_complex.c -lgsl -lgslcblas -lm -lfftw3 -O3 -ffast-math
./a.out
```
