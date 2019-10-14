#include "cuCG.hpp"

#ifndef __NVCC__
// make linter happy
dim3 gridDim, blockIdx, blockDim, threadIdx;
__device__ unsigned int atomicOr(unsigned int *address, unsigned int val);
__device__ unsigned int atomicAnd(unsigned int *address, unsigned int val);
#endif

#define DEBUG_THREAD -1

namespace cuCG
{

using Vec3 = Eigen::Vector3f;
using Mat3 = Eigen::Matrix3f;
// strain functions
using strain_func = Mat3 (*)(const Mat3 &, const Mat3 &, const Mat3 &, float, float);

// REMOVE
__device__ __host__ static void printMat(const Mat3 &m)
{
	printf("Mat3@%p:\n", &m);
	for (int i = 0; i < 3; i++)
	{
		printf("%f %f %f\n", m(i, 0), m(i, 1), m(i, 2));
	}
}

// REMOVE
__device__ __host__ static void printVec(const Vec3 &v)
{
	printf("Vec3@%p:\n", &v);
	for (int i = 0; i < 3; i++)
	{
		printf("%f\n", v(i));
	}
}

__device__ __host__ void printAllVec(Vec3 *v, size_t N)
{
	if (!v)
		return;
	for (int i = 0; i < N; i++)
	{
		printf("v[%d]:\n", i);
		printVec(v[i]);
	}
	printf("\n");
}

__device__ static void matMul(const Mat3 &a, const Mat3 &b, Mat3 &c)
{
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{
			if (blockIdx.x * blockDim.x + threadIdx.x == DEBUG_THREAD)
				printf("c(%d, %d) = (%f, %f, %f) dot (%f, %f, %f)\n", i, j, a(i, 0), a(i, 1), a(i, 2), b(0, j), b(1, j), b(2, j));
			c(i, j) = a.row(i).dot(b.col(j));
		}
	// REMOVE
	if (blockIdx.x * blockDim.x + threadIdx.x == DEBUG_THREAD)
	{
		printf("a: \n");
		printMat(a);
		printf("b: \n");
		printMat(b);
	}
}

// __device__ static Mat3 Kirchhoff_x(const Mat3 &F)
// {
// 	return .5 * (F + F.transpose()) - Mat3::Identity();
// }

// __device__ static Mat3 Kirchhoff_v(const Mat3 &F)
// {
// 	return .5 * (F + F.transpose());
// }
__device__ static Mat3 Kirchhoff_x(const Mat3 &F, const Mat3 &dF, const Mat3 &E, float lambda, float mu)
{
	// P = 2 * mu * E + lambda + E.trace * I
	Mat3 P;
	matMul(F, ((2. * mu) * E + lambda * E.trace() * Mat3::Identity()), P);
	return P;
}

__device__ static Mat3 Kirchhoff_v(const Mat3 &F, const Mat3 &dF, const Mat3 &E, float lambda, float mu)
{
	// dE = .5 * ( dF' * F + F' * dF);
	Mat3 dFtF, FtdF, dummy;
	matMul(dF.transpose(), F, dFtF);
	matMul(F.transpose(), dF, FtdF);
	Mat3 dE = 0.5 * (dFtF + FtdF);

	// dP =
	// dF * (2. * mu *  E + lambda *  E.trace() * I) +
	//  F * (2. * mu * dE + lambda * dE.trace() * I);
	return Kirchhoff_x(dF, dummy, E, lambda, mu) + Kirchhoff_x(F, dummy, dE, lambda, mu);
}



// Linear elasticity or damping. K_i is defined by scfg[idx]
__device__ static void mul_K4(Vec3 f[4], SimCfg *scfg, unsigned tetrIdx, Vec3 q[4], strain_func pfunc, float scale)
{
	const float mu = scfg->mu, lambda = scfg->lambda;
	const Vec3 *x = scfg->x;
	const unsigned *vIdx = &(scfg->tetrIdcs[tetrIdx << 2]);
	const unsigned tetrKind = scfg->tetrKind[tetrIdx];
	const Mat3 &dmInv = scfg->dmInvs[tetrKind];
	const float restV = scfg->restVs[tetrKind];

	Mat3 Ds, F;
	for (int i = 0; i < 3; i++)
		Ds.col(i) = x[vIdx[i + 1]] - x[vIdx[0]];
	matMul(Ds, dmInv, F);

	Mat3 Dq, DF;
	for (int i = 0; i < 3; i++)
		Dq.col(i) = q[i + 1] - q[0];
	matMul(Dq, dmInv, DF);

	// E = 0.5 * (F'F - I)
	Mat3 E;
	matMul(F.transpose(), F, E);
	E = .5 * (E - Mat3::Identity());

	// // REMOVE
	// if (blockIdx.x * blockDim.x + threadIdx.x == DEBUG_THREAD)
	// 	for (int i = 0; i < 4; i++)
	// 	{
	// 		printf("x[%d]:", i);
	// 		printVec(x[i]);
	// 	}

	// Mat3 strain = pfunc(F);
	// Mat3 P = scale * (2. * mu * strain + lambda * strain.trace() * Mat3::Identity());
	Mat3 P = scale * pfunc(F, DF, E, lambda, mu);

	Mat3 H;
	matMul(-restV * P, dmInv.transpose(), H);

	//REMOVE
	if (blockIdx.x * blockDim.x + threadIdx.x == DEBUG_THREAD)
	{
		printf("Ds: \n");
		printMat(Ds);
		printf("dmInv: \n");
		printMat(dmInv);
		printf("F: \n");
		printMat(F);
		printf("E: \n");
		printMat(E);
		printf("P: \n");
		printMat(P);
		printf("dmInv.T: \n");
		printMat(dmInv.transpose());
		printf("H: \n");
		printMat(H);
	}

	for (int i = 0; i < 3; i++)
	{
		f[i + 1] += H.col(i);
		f[0] -= H.col(i);
	}
}

__device__ static void applyA4(Vec3 Adx[4], SimCfg *scfg, unsigned tetrIdx, Vec3 dx[4])
{
	// 1 + g / h
	const float _1_gamma_h = 1 + scfg->gamma / scfg->dt;
	// 1 / h ^ 2
	const float _1_h_sq = 1 / (scfg->dt * scfg->dt);
	// M_i
	const float m = scfg->m[tetrIdx];

	Vec3 *f = Adx;

	// M_i / h ^ 2 * dx_i
	for (size_t i = 0; i < 4; i++)
		f[i] = (m * _1_h_sq) * dx[i];

	// + K_i * dx_i
	mul_K4(f, scfg, tetrIdx, dx, Kirchhoff_v, _1_gamma_h);

	// * (1 + g / h)
	for (size_t i = 0; i < 4; i++)
		f[i] = _1_gamma_h * f[i];

	if (blockIdx.x * blockDim.x + threadIdx.x == DEBUG_THREAD)
	{
		printf("tetrKind @ applyA4: %d\n", scfg->tetrKind[tetrIdx]);
		printf("Adx @ applyA4:\n");
		printAllVec(Adx, 4);
		printf("dx @ applyA4:\n");
		printAllVec(dx, 4);
	}
}

__device__ static void computeb4(Vec3 b[4], SimCfg *scfg, unsigned tetrIdx, Vec3 xpred[4], Vec3 vold[4], Vec3 vpred[4])
{
	const float m = scfg->m[tetrIdx];
	const float h = scfg->dt;

	Vec3 *f = b;

	// M_i / h * (v_i - v_pred_i)
	for (size_t i = 0; i < 4; i++)
		f[i] = m / h * (vold[i] - vpred[i]);

	// + f_el(x_pred_i)
	mul_K4(f, scfg, tetrIdx, xpred, Kirchhoff_x, 1);

	// + f_fr(v_pred_i)
	mul_K4(f, scfg, tetrIdx, vpred, Kirchhoff_v, -scfg->gamma);
}

// __device__ static void calcDmV4(Vec3 verts[4], Mat3 &Dm, float &restV)
// {
// 	for (size_t i = 0; i < 4; i++)
// 		Dm.col(i) = verts[i] - verts[0];

// 	Dm = Dm.inverse();

// 	restV = Dm.determinant();
// 	assert(restV > 0);
// }

__device__ static void copy4(Vec3 dest[4], Vec3 *src, unsigned idx[4])
{
	for (size_t i = 0; i < 4; i++)
		dest[i] = src[idx[i]];
}

__device__ static void atomic_paste1(Vec3 *dest, Vec3 *src, int idx, unsigned *mutex)
{
	unsigned mutexSecIdx = idx / sizeof(unsigned);
	// shift to the bit of index idx in mutex bitmap
	unsigned mutexFlagBit = 1 << (idx - mutexSecIdx * sizeof(unsigned));
	bool isSet = false;
	do
	{
		// atomically fetch the old section and see if the old bit is not set
		if (isSet = ((atomicOr(mutex + mutexSecIdx, mutexFlagBit) & mutexFlagBit) == 0))
		{
			// critical section
			src[idx] += *dest;

			// clear flag
			atomicAnd(mutex + mutexSecIdx, ~mutexFlagBit);
		}
	} while (!isSet);
}

__device__ static void paste4(Vec3 dest[4], Vec3 *src, unsigned idx[4], unsigned *mutex)
{
	for (size_t i = 0; i < 4; i++)
		atomic_paste1(&dest[i], src, idx[i], mutex);
	//REMOVE
	if (blockIdx.x * blockDim.x + threadIdx.x == DEBUG_THREAD)
	{
		printf("dest @ paste4:\n");
		printAllVec(dest, 4);
	}
}

__global__ void applyA(Vec3 *Adx, SimCfg *scfg, Vec3 *dx)
{
	const size_t N_CELL = scfg->N_CELL;

	int tetrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tetrIdx >= N_CELL)
		return;

	unsigned *vIdx = &(scfg->tetrIdcs[tetrIdx << 2]);

	Vec3 Adx4[4], dx4[4];
	copy4(Adx4, Adx, vIdx), copy4(dx4, dx, vIdx);

	applyA4(Adx4, scfg, tetrIdx, dx4);

	paste4(Adx4, Adx, vIdx, scfg->mutex);

	//REMOVE
	if (blockIdx.x * blockDim.x + threadIdx.x == DEBUG_THREAD)
	{
		printf("Adx4 @ applyA:\n");
		printAllVec(Adx4, 4);
		printf("Adx @ applyA:\n");
		printAllVec(Adx, scfg->N_PART);
		printf("dx @ applyA:\n");
		printAllVec(dx, scfg->N_PART);
	}
}

__global__ void computeb(Vec3 *b, SimCfg *scfg, Vec3 *xpred, Vec3 *vpred)
{
	const size_t N_CELL = scfg->N_CELL;

	int tetrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tetrIdx >= N_CELL)
		return;

	unsigned *vIdx = &(scfg->tetrIdcs[tetrIdx << 2]);

	Vec3 b4[4], xpred4[4], vold4[4], vpred4[4];
	copy4(b4, b, vIdx), copy4(xpred4, xpred, vIdx), copy4(vold4, scfg->v, vIdx), copy4(vpred4, vpred, vIdx);

	computeb4(b4, scfg, tetrIdx, xpred4, vold4, vpred4);

	paste4(b4, b, vIdx, scfg->mutex);

	//REMOVE
	if (blockIdx.x * blockDim.x + threadIdx.x == DEBUG_THREAD)
	{
		printf("b @ computeb:\n");
		printAllVec(b, scfg->N_PART);
		printf("xpred @ computeb:\n");
		printAllVec(xpred, scfg->N_PART);
		printf("vpred @ computeb:\n");
		printAllVec(vpred, scfg->N_PART);
		printf("tetrKind @ applyA4: %d", scfg->tetrKind[tetrIdx]);
	}
}

// __global__ void calcDmV(SimCfg *scfg)
// {
// 	const size_t N_CELL = scfg->N_CELL;

// 	int tetrIdx = blockIdx.x * blockDim.x + threadIdx.x;

// 	if (tetrIdx >= N_CELL)
// 		return;

// 	unsigned *vIdx = *scfg->tetrIdcs[tetrIdx << 2];

// 	Vec3 v4[4];
// 	copy4(v4, scfg->x, vIdx);

// 	calcDmV4(v4, scfg->dmInvs[tetrIdx], scfg->restVs[tetrIdx]);

// 	paste4(v4, scfg->x, vIdx);
// }

// micro kernel for {p_i} = {<x_i, y_i>}
__global__ void dot(float *p, Vec3 *x, Vec3 *y, size_t N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
		p[i] = x[i].dot(y[i]);

	//REMOVE
	if (i == DEBUG_THREAD)
	{
		printf("x @ dot:\n");
		printAllVec(x, N);
		printf("y @ dot:\n");
		printAllVec(y, N);
	}
}

// micro kernel for {z_i} = {x_i * c + y_i}
__global__ void saxpy(Vec3 *z, float c, Vec3 *x, Vec3 *y, size_t N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
		z[i] = x[i] * c + ((y == NULL) ? Vec3::Zero() : y[i]);

	//REMOVE
	if (i == DEBUG_THREAD)
	{
		printf("x @ saxpy:\n");
		printAllVec(x, N);
		printf("y @ saxpy:\n");
		printAllVec(y, N);
		printf("z @ saxpy:\n");
		printAllVec(z, N);
	}
}

__global__ void max(float *p, Vec3 *x, size_t N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
		p[i] = x[i].maxCoeff();
}

// zero out all if idx == NULL
__global__ void zero(Vec3 *x, unsigned *idx, size_t N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
		x[idx == NULL ? i : idx[i]] = Vec3::Zero();
}

} // namespace cuCG
