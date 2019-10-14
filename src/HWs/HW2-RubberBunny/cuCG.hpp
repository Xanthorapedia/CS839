#pragma once

#include <vector>
#include <Eigen/Core>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_functions.h>

#include "cgConfig.hpp"

#ifdef __NVCC__
#define _KCALL(ker, nBlock, tPerBlock) /*printf(#ker"<<<%d, %d>>>\n", nBlock, tPerBlock);*/ if (nBlock > 0) ker<<<nBlock, tPerBlock>>>
#else
#define _KCALL(ker, nBlock, tPerBlock) ker
#endif

#define _min(x, y) ((x) < (y) ? (x) : (y))

// kernel call
#define KCALL(func, N_ELEMENTS) _KCALL(func, (((N_ELEMENTS) + N_THREAD - 1) / N_THREAD), _min(N_ELEMENTS, N_THREAD))

// error handling
#define HANDLE_ERROR(err) (cuCG::HandleError(err, __FILE__, __LINE__))

namespace cuCG
{

void HandleError(cudaError_t err, const char *file, int line);

void progress(SimCfg *scfg, float eps);

// p_i = x_i * y_i
__global__ void dot(float *p, Vec3 *x, Vec3 *y, size_t N);

// z_i = c * x_i + y_i
__global__ void saxpy(Vec3 *z, float c, Vec3 *x, Vec3 *y, size_t N);

// p_i = max(x_i)
__global__ void max(float *p, Vec3 *x, size_t N);

// x_i = 0
__global__ void zero(Vec3 *x, unsigned *idx, size_t N);

__global__ void applyA(Vec3 *Adx, SimCfg *scfg, Vec3 *dx);

__global__ void computeb(Vec3 *b, SimCfg *scfg, Vec3 *xpred, Vec3 *vpred);

__device__ __host__ void printAllVec(Vec3* v, size_t N);

} // namespace cuCG
