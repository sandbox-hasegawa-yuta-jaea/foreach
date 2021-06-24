// foreach.h 使用方法サンプル（JUPITER-AMRの CPU/GPU 共通化手法）
// コメント内のファイル名は https://github.com/NaoyukiOnodera/JupiterAMR を参照してください。

#include <iostream>
#include "foreach.h"

#ifdef USE_NVCC
#include <cuda.h>
#else 
#include <cstdlib>
#endif

using real = double;

// function_srcs/funcAllocate.h 相当
void allocate(real** f, int n) {
#ifdef USE_NVCC
    cudaMallocManaged(f, n * sizeof(real));
#else
    *f = reinterpret_cast<real*>(std::aligned_alloc(32, n * sizeof(real)));
#endif
}

// defines/defineCal.h 相当
#ifdef USE_NVCC
  #define __HOST__    __host__
  #define __DEVICE__  __device__
#else
  #define __HOST__ 
  #define __DEVICE__
#endif

#define __HD__ __HOST__ __DEVICE__

// これが共通カーネル (function_srcs/FuncNS.hなど）
__HD__  
void sample_copy(
const int nx,
const int ny,
real* fn,
const real* f
) {
    FOR_EACH2D(i, j, nx, ny) {
        const int ij = i + j*nx;
        fn[ij] = f[ij];
    }
}


// ここからmain
int main() {
    real* f;
    real* fn;
    constexpr int nx = 1024;
    constexpr int ny = 1024;
    allocate(&f , nx*ny);
    allocate(&fn, nx*ny);

    // 共通カーネルをforeachで呼ぶ（include/NavierStokes.h:260など）
    foreach::exec2d<foreach::opti>(
        // dim3
        nx, ny, 

        // lambda
        [=] __HD__ () {
            sample_copy(nx, ny, fn, f);
        }
    );
}
