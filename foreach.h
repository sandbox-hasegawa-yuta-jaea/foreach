#ifndef FOREACH_H_
#define FOREACH_H_

#include <cstdint>
#include <cassert>
#include <type_traits>
#include <string>

#if defined(USE_INTEL)
  #define PRAGMA_FOR_SIMD  _Pragma("ivdep")

  #define ASSUME_ALIGNED64(VAL) __assume_aligned(VAL, 64)
  #define ASSUME64(VAL) __assume(VAL%64 == 0)

#elif defined(USE_A64FX)
  #define PRAGMA_FOR_SIMD  _Pragma("loop prefetch_sequential soft")

  #define ASSUME_ALIGNED64(VAL)  
  #define ASSUME64(VAL)  
#else
  #define PRAGMA_FOR_SIMD  

  #define ASSUME_ALIGNED64(VAL)  
  #define ASSUME64(VAL)  
#endif


// macro
#ifdef __CUDA_ARCH__
    #define FOR_EACH2D(I, J, NX, NY) \
        const auto I = threadIdx.x; \
        const auto J = threadIdx.y;

    #define FOR_EACH1D(IJK, NN) \
        const auto IJK = threadIdx.x + NX_LEAF*threadIdx.y + NX_LEAF*NX_LEAF*threadIdx.z;
     
    #define SKIP_FOR() return
#else
    #define FOR_EACH2D(I, J, NX, NY) \
        PRAGMA_FOR_SIMD \
        _Pragma("omp parallel for") \
        for(int J=0; J<NY; J++) \
        PRAGMA_FOR_SIMD \
        for(int I=0; I<NX; I++) 

    #define FOR_EACH1D(IJK, NN) \
        PRAGMA_FOR_SIMD \
        for(int IJK=0; IJK<NN; IJK++) 

    #define SKIP_FOR() continue
#endif


namespace foreach {

struct backend {};

struct openmp    : backend {};
struct cuda      : backend {};


#ifdef USE_NVCC
using opti = cuda;
#else
using opti = openmp;
#endif

#ifdef USE_NVCC
template<class Func, class... Args> 
__global__ void exec2d_gpu(
    Func    func,
    Args... args
    ) 
{
    func(args...);
} 
#endif


template<class Func, class... Args> 
void exec2d_cpu(
    Func    func,
    Args... args
    )
{
    func(args...);
}


template<class ExecutionPolicy, class Func, class... Args>
void exec2d(
    const int nx,
    const int ny,
    Func    func,
    Args... args
    )
{
    if (std::is_same<ExecutionPolicy, cuda>::value) {
#ifdef USE_NVCC
        constexpr int nth_x = 16, nth_y = 16;
        const int bx = (nx + nth_x - 1) / nth_x;
        const int by = (ny + nth_y - 1) / nth_y;
        exec2d_gpu<Func, Args...> <<<
                dim3(bx, by), 
                dim3(nth_x, nth_y)
            >>> (func, args...);
        cudaDeviceSynchronize();
#endif
    }
    else if (std::is_same<ExecutionPolicy, openmp>::value) {
        exec2d_cpu<Func, Args...>(func, args...);
    }
    else {
        static_assert(std::is_base_of<backend, ExecutionPolicy>::value, "unexpected Execution Policy");
    }
}

} // namespace foreach


#endif
