#include "SVD.h"

#include "ThrustUtils.h"

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <Eigen/SVD>

namespace FEM
{
namespace SVD
{

void handleFailure(cusolverStatus_t err);
void handleFailure(cudaError_t err);

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
static cusolverDnHandle_t solver_handle = NULL;

int *devInfo;

/* devie workspace for gesvdj */
static int work_size = 0;
static float *d_work = NULL;
#endif

void destroyCUSolverHandle()
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
	if (solver_handle)
		handleFailure(cusolverDnDestroy(solver_handle));
	if (devInfo)
		handleFailure(cudaFree(devInfo));
	if (d_work)
	{
		handleFailure(cudaFree(d_work));
		d_work = NULL;
		work_size = 0;
	}
#endif
}

struct SVD_op : public thrust::unary_function<Tuple<const float *, float *, float *, float *> &, void>
{
	// argument indices
	enum
	{
		arg_A,
		arg_U,
		arg_vS,
		arg_V,
	};

	SVD_op() {}

	__host__ __device__ void operator()(const Tuple<const float *, float *, float *, float *> &args) const
	{
		ConstMat3View A(thrust::get<arg_A>(args));
		Mat3View U(thrust::get<arg_U>(args));
		Vec3View vS(thrust::get<arg_vS>(args));
		Mat3View V(thrust::get<arg_V>(args));

		Eigen::JacobiSVD<Mat3> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
		U = svd.matrixU();
		vS = svd.singularValues();
		V = svd.matrixV();
	}
};

void svd(collection<float> &A, collection<float> &U, collection<float> &S, collection<float> &V,
		 const float tol, const int maxSweeps)
{
	const unsigned M = 3;
	const unsigned N = 3;
	const unsigned numMatrices = A.size() / (M * N);

// #ifdef HOST_EXEC
// 	for (unsigned i = 0; i < numMatrices; i++)
// 	{
// 		Eigen::JacobiSVD<Mat3> svd(Eigen::Map<Mat3>(thrust::raw_pointer_cast(A.data() + i * 9)),
// 								   Eigen::ComputeFullU | Eigen::ComputeFullV);
// 		Eigen::Map<Mat3>(thrust::raw_pointer_cast(U.data() + i * 9)) = svd.matrixU();
// 		Eigen::Map<Vec3>(thrust::raw_pointer_cast(S.data() + i * 3)) = svd.singularValues();
// 		Eigen::Map<Mat3>(thrust::raw_pointer_cast(V.data() + i * 9)) = svd.matrixV();
// 	}
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CPP || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
	IncrItr idx(0);

	auto svd_itr_UvSV = make_zipped_itr(make_strided_itr(A, idx, 9),
										make_strided_itr(U, idx, 9),
										make_strided_itr(S, idx, 3),
										make_strided_itr(V, idx, 9));
	thrust::for_each_n(EXEC_POL, svd_itr_UvSV, numMatrices, SVD_op());

#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
	IncrItr idx(0);

	thrust::host_vector<float> Ah = A;
	thrust::host_vector<float> Uh = U;
	thrust::host_vector<float> Sh = S;
	thrust::host_vector<float> Vh = V;

	auto svd_itr_UvSV = make_zipped_itr(make_strided_itrh(Ah, idx, 9),
										make_strided_itrh(Uh, idx, 9),
										make_strided_itrh(Sh, idx, 3),
										make_strided_itrh(Vh, idx, 9));
	thrust::for_each_n(thrust::host, svd_itr_UvSV, numMatrices, SVD_op());
	thrust::copy(Uh.begin(), Uh.end(), U.begin());
	thrust::copy(Sh.begin(), Sh.end(), S.begin());
	thrust::copy(Vh.begin(), Vh.end(), V.begin());

#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA && SVD_IMPL == CUDA
	const int lda = M;

	// --- Setting the device matrix and moving the host matrix to the device
	float *A_arr = thrust::raw_pointer_cast(A.data());
	float *U_arr = thrust::raw_pointer_cast(U.data());
	float *S_arr = thrust::raw_pointer_cast(S.data());
	float *V_arr = thrust::raw_pointer_cast(V.data());

	// --- device side SVD workspace and matrices
	int new_work_size = 0;

	if (!devInfo)
		handleFailure(cudaMalloc(&devInfo, sizeof(int)));

	// int devInfo_h = 0;	/* host copy of error devInfo_h */

	// --- CUSOLVER_EIG_MODE_VECTOR - Compute eigenvectors
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

	// --- CUDA solver initialization
	if (!solver_handle)
		handleFailure(cusolverDnCreate(&solver_handle));

	// --- Configuration of gesvdj
	gesvdjInfo_t gesvdj_params = NULL;
	handleFailure(cusolverDnCreateGesvdjInfo(&gesvdj_params));

	// --- Set the maximum number of sweeps, since the default value of max. sweeps is 100
	handleFailure(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
	handleFailure(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, maxSweeps));

	// --- Query the SVD workspace
	handleFailure(cusolverDnSgesvdjBatched_bufferSize(
		solver_handle,
		jobz,  // --- Compute the singular vectors or not
		M,	 // --- Nubmer of rows of A, 0 <= M
		N,	 // --- Number of columns of A, 0 <= N
		A_arr, // --- M x N
		lda,   // --- Leading dimension of A
		S_arr, // --- Square matrix of size min(M, N) x min(M, N)
		U_arr, // --- M x M
		lda,   // --- Leading dimension of U, ldu >= max(1, M)
		V_arr, // --- N x N
		lda,   // --- Leading dimension of V, ldv >= max(1, N)
		&new_work_size,
		gesvdj_params,
		numMatrices));

	// resize if necessary
	if (!d_work || work_size < new_work_size)
	{
		work_size = new_work_size;
		handleFailure(cudaFree(d_work));
		handleFailure(cudaMalloc(&d_work, sizeof(float) * work_size));
	}

	// --- Compute SVD
	handleFailure(cusolverDnSgesvdjBatched(
		solver_handle,
		jobz,  // --- Compute the singular vectors or not
		M,	 // --- Number of rows of A, 0 <= M
		N,	 // --- Number of columns of A, 0 <= N
		A_arr, // --- M x N
		lda,   // --- Leading dimension of A
		S_arr, // --- Square matrix of size min(M, N) x min(M, N)
		U_arr, // --- M x M
		lda,   // --- Leading dimension of U, ldu >= max(1, M)
		V_arr, // --- N x N
		lda,   // --- Leading dimension of V, ldv >= max(1, N)
		d_work,
		work_size,
		devInfo,
		gesvdj_params,
		numMatrices));

	// handleFailure(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

	// if (0 < devInfo_h)
	// {
	// 	printf("%d-th parameter is wrong \n", -devInfo_h);
	// 	exit(1);
	// }
	// else if (0 > devInfo_h)
	// {
	// 	printf("WARNING: svd(): devInfo_h = %d : gesvdj does not converge \n", devInfo_h);
	// }

	if (gesvdj_params)
		handleFailure(cusolverDnDestroyGesvdjInfo(gesvdj_params));
#endif
}

// https://devtalk.nvidia.com/default/topic/821174/cuda-7-0-many-small-parallel-svds-in-matlab/
inline void __handleSolverError(cusolverStatus_t err, const char *file, const int line)
{
	if (CUSOLVER_STATUS_SUCCESS != err)
	{
		std::string errCode;
		switch (err)
		{
		case CUSOLVER_STATUS_SUCCESS:
			errCode = "CUSOLVER_SUCCESS";

		case CUSOLVER_STATUS_NOT_INITIALIZED:
			errCode = "CUSOLVER_STATUS_NOT_INITIALIZED";

		case CUSOLVER_STATUS_ALLOC_FAILED:
			errCode = "CUSOLVER_STATUS_ALLOC_FAILED";

		case CUSOLVER_STATUS_INVALID_VALUE:
			errCode = "CUSOLVER_STATUS_INVALID_VALUE";

		case CUSOLVER_STATUS_ARCH_MISMATCH:
			errCode = "CUSOLVER_STATUS_ARCH_MISMATCH";

		case CUSOLVER_STATUS_EXECUTION_FAILED:
			errCode = "CUSOLVER_STATUS_EXECUTION_FAILED";

		case CUSOLVER_STATUS_INTERNAL_ERROR:
			errCode = "CUSOLVER_STATUS_INTERNAL_ERROR";

		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			errCode = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
		}
		std::cout << errCode << " in " << file << " at line " << line << std::endl;
		cudaDeviceReset();
		assert(0);
	}
}

inline void __handleGeneralError(cudaError_t err, const char *file, const int line)
{
	// CUDA error handeling from the "CUDA by example" book
	if (cudaSuccess != err)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

void handleFailure(cusolverStatus_t err) { __handleSolverError(err, __FILE__, __LINE__); }
void handleFailure(cudaError_t err) { __handleGeneralError(err, __FILE__, __LINE__); }

} // namespace SVD
} // namespace FEM
