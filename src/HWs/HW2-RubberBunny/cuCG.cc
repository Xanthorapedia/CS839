#include "cuCG.hpp"

namespace cuCG
{

static float dot_prod(Vec3 *x, Vec3 *y, float *arr, int N);
static float oneNorm(Vec3 *x, float *arr, int N);
static void *getDevPtr(void *hostPtr);

void HandleError(cudaError_t err, const char *file, int line)
{
	// CUDA error handeling from the "CUDA by example" book
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

// void populateRestTetr(SimCfg &scfg)
// {
// 	KCALL(calcDmV, scfg->N_TETR)
// 	(scfg);
// }

void solveCG(SimCfg *scfg, float delta)
{
	SimCfg *const devScfg = (SimCfg *)getDevPtr(scfg);
	const unsigned N_PART = scfg->N_PART;
	const unsigned N_CELL = scfg->N_CELL;

	Vec3 *b = scfg->b,
		 *p = scfg->p,
		 *q = scfg->q,
		 *r = scfg->r,
		 *dx = scfg->dx;

	float *scratch_arr = scfg->scratch_arr,
		  rho_old = 0;

	// b, assume v_pred = v, x_pred (r) = x + v * dt
	KCALL(zero, N_PART)
	(b, NULL, N_PART);
	KCALL(saxpy, N_PART)
	(r, scfg->dt, scfg->v, scfg->x, N_PART);
	KCALL(computeb, N_CELL)
	(b, devScfg, r, scfg->v);
	HANDLE_ERROR(cudaDeviceSynchronize());

	// A * dx0
	KCALL(zero, N_PART)
	(q, NULL, N_PART);
	KCALL(applyA, N_CELL)
	(q, devScfg, dx);
	HANDLE_ERROR(cudaDeviceSynchronize());

	// r = - A * dx0 + b
	KCALL(saxpy, N_PART)
	(r, -1, q, b, N_PART);
	HANDLE_ERROR(cudaDeviceSynchronize());

	// Project r
	KCALL(zero, scfg->N_CONSTR)
	(r, scfg->constrained, scfg->N_CONSTR);

	size_t i;
	for (i = 0; i < CG_MAX_ITERATION; i++)
	{
		// rho = r * r
		float rho = dot_prod(r, r, scratch_arr, N_PART);

		if (oneNorm(r, scratch_arr, N_PART) < delta)
			break;

		// p = r + rho / rho_old * p
		KCALL(saxpy, N_PART)
		(p, (i == 0 ? 0 : rho / rho_old), p, r, N_PART);
		//REMOVE
		HANDLE_ERROR(cudaDeviceSynchronize());

		// q = A * p
		KCALL(zero, N_PART)
		(q, NULL, N_PART);
		KCALL(applyA, N_CELL)
		(q, devScfg, p);
		//REMOVE
		HANDLE_ERROR(cudaDeviceSynchronize());

		// Project q
		KCALL(zero, scfg->N_CONSTR)
		(q, scfg->constrained, scfg->N_CONSTR);

		float alpha = rho / dot_prod(p, q, scratch_arr, N_PART);

		// x = x + alpha * p
		KCALL(saxpy, N_PART)
		(dx, +alpha, p, dx, N_PART);
		//REMOVE
		HANDLE_ERROR(cudaDeviceSynchronize());

		// r = r - alpha * q
		KCALL(saxpy, N_PART)
		(r, -alpha, q, r, N_PART);
		//REMOVE
		HANDLE_ERROR(cudaDeviceSynchronize());

		rho_old = rho;
	}

	HANDLE_ERROR(cudaDeviceSynchronize());

	if (i == CG_MAX_ITERATION)
		printf("CG exploded with rho = %f after %d iterations\n", rho_old, CG_MAX_ITERATION);
}

void progress(SimCfg *scfg, float eps)
{
	const int N_PART = scfg->N_PART;

	// x += v * dt
	KCALL(saxpy, N_PART)
	(scfg->x, scfg->dt, scfg->v, scfg->x, N_PART);

	solveCG(scfg, eps);

	// v += dx / dt
	KCALL(saxpy, N_PART)
	(scfg->v, 1 / scfg->dt, scfg->dx, scfg->v, N_PART);

	// x += dx
	KCALL(saxpy, N_PART)
	(scfg->x, 1, scfg->dx, scfg->x, N_PART);
}

static float dot_prod(Vec3 *x, Vec3 *y, float *arr, int N)
{
	KCALL(dot, N)
	(arr, x, y, N);

	// block until gpu finishes before reading data
	HANDLE_ERROR(cudaDeviceSynchronize());

	float sum = 0;
	for (size_t i = 0; i < N; i++)
		sum += arr[i];

	return sum;
}

static float oneNorm(Vec3 *x, float *arr, int N)
{
	KCALL(max, N)
	(arr, x, N);

	// block until gpu finishes before reading data
	HANDLE_ERROR(cudaDeviceSynchronize());

	float max = std::abs(arr[0]);
	for (size_t i = 0; i < N; i++)
		max = std::max(max, std::abs(arr[i]));

	return max;
}

static void *getDevPtr(void *hostPtr)
{
	/*return hostPtr;*/
	void *devPtr = NULL;
	HANDLE_ERROR(cudaHostGetDevicePointer(&devPtr, hostPtr, 0));
	if (!devPtr)
		std::logic_error("Device pointer NULL.");
	return devPtr;
}

} // namespace cuCG
