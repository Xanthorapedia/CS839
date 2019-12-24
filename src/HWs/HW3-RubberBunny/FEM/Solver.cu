#include "Solver.h"
#include "CG.h"
#include "SVD.h"
#include "ThrustUtils.h"

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/iterator/discard_iterator.h>

#include <Eigen/Dense>

// #define PRINT_DBG

namespace FEM
{
namespace Solver
{

template <typename Mat3T0, typename Mat3T1, typename Mat3T2>
__device__ __host__ static void matMul(const Mat3T0 &A, const Mat3T1 &B, Mat3T2 &AB)
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
	const int R = A.rows();
	const int C = B.cols();
	for (int i = 0; i < R; i++)
		for (int j = 0; j < C; j++)
			AB(i, j) = A.row(i).dot(B.col(j));
#else
	AB = A * B;
#endif
}

template <typename Mat3T>
__device__ __host__ static float determinant(const Mat3T &A)
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
	return A.row(0).dot(A.row(1).cross(A.row(2)));
#else
	return A.determinant();
#endif
}

template <typename T>
__device__ __host__ static void printMat(const T &m)
{
	for (int i = 0; i < m.rows(); i++)
	{
		for (int j = 0; j < m.cols(); j++)
			printf("%f ", m(i, j));
		printf("\n");
	}
	printf("\n");
}

template <typename T>
__device__ __host__ static void printMat(const collection<float> &arr)
{
	const float *mat = thrust::raw_pointer_cast(arr.data());
	T t;
	int N = arr.size() / t.size();
	for (int i = 0; i < N; i++)
	{
		printf("Mat %d:\n", i);
		printMat(Eigen::Map<const T>(mat + t.size() * i));
	}
}

// takes in {(x0, x1, x2, x3, &out)}
// Mat(out) = Mat(x1 - x0, x2 - x0, x3 - x0)
struct find_F_dF_op : public thrust::unary_function<tetrIdces<Tuple<
														const Vec3 &, const Vec3 &,
														const Vec3 &, const Vec3 &,
														float *>>,
													void>
{
	// argument indices
	enum
	{
		arg_x_dx_0,
		arg_x_dx_1,
		arg_x_dx_2,
		arg_x_dx_3,
		arg_out,
	};

	__host__ __device__ void operator()(const tetrIdces<Tuple<const Vec3 &, const Vec3 &,
															  const Vec3 &, const Vec3 &,
															  float *>> &ti) const
	{
		const Mat3 &dmInv = thrust::get<T_D_dmInv>(thrust::get<T_D>(ti));
		auto args = thrust::get<T_E>(ti);

		Mat3 Ds_dDs;
		const Vec3 &x0 = thrust::get<arg_x_dx_0>(args);
		Ds_dDs.col(0) = thrust::get<arg_x_dx_1>(args) - x0;
		Ds_dDs.col(1) = thrust::get<arg_x_dx_2>(args) - x0;
		Ds_dDs.col(2) = thrust::get<arg_x_dx_3>(args) - x0;

		matMul(Ds_dDs, dmInv, Eigen::Map<Mat3>(thrust::get<arg_out>(args)));
#ifdef PRINT_DBG
		printf("in find_F_dF_op\n");
		printMat(Ds_dDs);
		printMat(dmInv);
		printMat(Eigen::Map<Mat3>(thrust::get<arg_out>(args)));
#endif
	}
};

struct rectify_SVD_op : public thrust::unary_function<Tuple<float *, float *, float *> &, void>
{
	// argument indices
	enum
	{
		arg_U,
		arg_vS,
		arg_V,
	};

	// singular value threshold
	const float sThresh;

	rectify_SVD_op(const float sThresh) : sThresh(sThresh) {}

	__host__ __device__ void operator()(const Tuple<float *, float *, float *> &svd) const
	{
		// array repr of SVD-ed F or dF
		Mat3View U(thrust::get<arg_U>(svd));
		Vec3View vS(thrust::get<arg_vS>(svd));
		Mat3View V(thrust::get<arg_V>(svd));

		if (determinant(U) < 0)
		{
			if (determinant(V) < 0)
			{
				// Both determinants negative, just negate 2nd column on both
				U.col(1) *= -1.f;
				V.col(1) *= -1.f;
			}
			else
			{
				// Only U has negative determinant, negate 2nd column and second singular value
				U.col(1) *= -1.f;
				vS[1] = -vS[1];
			}
		}
		else if (determinant(V) < 0)
		{
			// Only V has negative determinant, negate 2nd column and second singular value
			V.col(1) *= -1.f;
			vS[1] = -vS[1];
		}

		// Apply thresholding of singular values
		vS = vS.cwiseMin(sThresh);
	}
};

using prod_K_tuple = Tuple<const float *, const float *,
						   const float *, const float *,
						   const float, float *>;
/* Multiplies tuple[0] (list of tetrahedra) by K specified by:
 * - mu, lambda from scfg
 * - singular value threshold from scfg
 * - 
 * and save results to tuple[2] (list of forces)
 */
struct add_prod_K_op : public thrust::unary_function<tetrIdces<prod_K_tuple>, void>
{
	// argument indices
	enum
	{
		arg_U,
		arg_vS,
		arg_V,
		arg_dF,
		arg_scale,
		arg_out,
	};

	const bool calc_dP;

	add_prod_K_op(const bool calc_dP) : calc_dP(calc_dP) {}

	using Vec2 = Eigen::Vector2f;
	using Mat2 = Eigen::Matrix2f;

	__host__ __device__ void operator()(const tetrIdces<prod_K_tuple> &ti) const
	{
		// consts
		auto dmInfo = thrust::get<T_D>(ti);
		const Mat3 &dmInv = thrust::get<T_D_dmInv>(dmInfo);
		const float restV = thrust::get<T_D_restV>(dmInfo);

		auto consts = thrust::get<T_C>(ti);
		const float lambda = thrust::get<T_C_lambda>(consts);
		const float mu = thrust::get<T_C_mu>(consts);

		// array repr of SVD-ed dF
		auto args = thrust::get<T_E>(ti);
		const ConstMat3View U(thrust::get<arg_U>(args));
		const ConstVec3View vS(thrust::get<arg_vS>(args));
		const ConstMat3View V(thrust::get<arg_V>(args));
		const ConstMat3View dF(thrust::get<arg_dF>(args));
		const float scale = thrust::get<arg_scale>(args);
		Mat34View out(thrust::get<arg_out>(args));

#ifdef PRINT_DBG
		printf("in add_prod_K_op:\n");
		printf("U:\n");
		printMat(U);
		printf("vS:\n");
		printMat(vS);
		printf("V:\n");
		printMat(V);
		if (calc_dP)
		{
			printf("dF:\n");
			printMat(dF);
		}
#endif
		Mat3 P_dP = calc_dP ? find_dP(U, vS, V, dF, mu, lambda)
							: find__P(U, vS, V, mu, lambda);
#ifdef PRINT_DBG
		printf("P_dP:\n");
		printMat(P_dP);
#endif
		Mat3 H_dH;
		matMul(-restV * P_dP * scale, dmInv.transpose(), H_dH);
#ifdef PRINT_DBG
		printf("dmInv':\n");
		printMat(dmInv.transpose());
		printf("H_dH:\n");
		printMat(H_dH);
#endif
		// expands H_dH:
		// out(0) += -H_dH(0) - H_dH(1) - H_dH(2)
		// out(1) += H_dH(0)
		// out(2) += H_dH(1)
		// out(3) += H_dH(2)
		Mat34 H_dH4;
		matMul(H_dH, H_dH_exp_mat, H_dH4);
		out += H_dH4;
	}

private:
	// -1 1 0 0
	// -1 0 1 0
	// -1 0 0 1
	const Mat34 H_dH_exp_mat = (Mat34() << -1, 1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 1).finished();

	// 1 1 0
	// 1 0 1
	// 0 1 1
	const Mat3 sumSigma_ij_mat = (Mat3() << 1, 1, 0, 1, 0, 1, 0, 1, 1).finished();

	// (a01, a10), (a02, a20) (a12 a21)
	const unsigned matEntryIdx[3][4] = {{0, 1, 1, 0},
										{0, 2, 2, 0},
										{1, 2, 2, 1}};

	__host__ __device__ Mat3 find__P(const Mat3 &U, const Vec3 &vS, const Mat3 &V, const float mu, const float lambda) const
	{
		Vec3 vStrain = vS - Vec3::Ones();
		Vec3 vP = (2 * mu) * vStrain + lambda * vStrain.sum() * Vec3::Ones();
#ifdef PRINT_DBG
		printf("in find__P:\n");
		printf("vP:\n");
		printMat(vP);
#endif
		Mat3 P, vPDiag, UvP;
		vPDiag = Mat3::Zero();
		vPDiag(0, 0) = vP(0);
		vPDiag(1, 1) = vP(1);
		vPDiag(2, 2) = vP(2);
		matMul(U, vPDiag, UvP);
		matMul(UvP, V.transpose(), P);
		// matMul(U * vP.asDiagonal(), V.transpose(), P);
		return P;
	}

	__host__ __device__ Mat3 find_dP(const Mat3 &U, const Vec3 &vS, const Mat3 &V, const Mat3 &dF, const float mu, const float lambda) const
	{

		Mat3 dF_hat, dP_hat;
		matMul(U.transpose(), dF, dP_hat); // use dP_hat as a temp
		matMul(dP_hat, V, dF_hat);
#ifdef PRINT_DBG
		printf("in find_dP:\n");
		printf("dF_hat:\n");
		printMat(dF_hat);
#endif

		float vStrain_tr = vS.sum() - 3;
		vStrain_tr = vStrain_tr < 0 ? 0 : vStrain_tr;

		Mat3 A = lambda * Mat3::Ones() + (2 * mu) * Mat3::Identity();

		// beta_ij = mu + (lambda * tr(Sigma - I) - 2 * mu) / (sigma_i + sigma_j)
		Vec3 sumSigma_ij;
		matMul(sumSigma_ij_mat, vS, sumSigma_ij);
		Vec3 beta_ij = mu * Vec3::Ones() + (lambda * vStrain_tr - 2 * mu) * (sumSigma_ij + 1e-4f * Vec3::Ones()).cwiseInverse();
#ifdef PRINT_DBG
		printf("vStrain_tr: %f\n\n", vStrain_tr);

		// printf("A:\n");
		// printMat(A);

		printf("sumSigma_ij:\n");
		printMat(sumSigma_ij);

		printf("beta_ij:\n");
		printMat(beta_ij);
#endif
		// B_ij = mu * (1 1; 1 1) + beta_ij * (1 -1; -1 1)
		Mat2 B_ij;

		matMul(A, dF_hat.diagonal(), dP_hat.diagonal());
		for (unsigned k = 0; k < 3; k++)
		{
			B_ij(0, 0) = B_ij(1, 1) = mu + beta_ij(k);
			B_ij(0, 1) = B_ij(1, 0) = mu - beta_ij(k);

			// pair-wise coordinates
			const unsigned i0 = matEntryIdx[k][0],
						   j0 = matEntryIdx[k][1],
						   i1 = matEntryIdx[k][2],
						   j1 = matEntryIdx[k][3];
			Vec2 dP_hat_ij;
			matMul(B_ij, Vec2(dF_hat(i0, j0), dF_hat(i1, j1)), dP_hat_ij);
#ifdef PRINT_DBG
			printf("B_%d%d:\n", i0, j0);
			printMat(B_ij);

			printf("dP_hat_%d%d_%d%d:\n", i0, j0, i1, j1);
			printMat(dP_hat_ij);
#endif
			dP_hat(i0, j0) = dP_hat_ij(0);
			dP_hat(i1, j1) = dP_hat_ij(1);
		}
#ifdef PRINT_DBG
		printf("dP_hat:\n");
		printMat(dP_hat);
#endif
		Mat3 dP;
		matMul(U, dP_hat, dF_hat); // use dF_hat as a temp
		matMul(dF_hat, V.transpose(), dP);

		return dP;
	}
};

struct add_M_h2_op : public thrust::binary_function<const Vec3 &, Tuple<const float, const Vec3 &>, Vec3>
{
	// argument indices
	enum
	{
		arg_m,
		arg_x,
	};

	const float _1_h2;

	add_M_h2_op(const float _1_h2) : _1_h2(_1_h2) {}

	__host__ __device__ Vec3 operator()(const Vec3 &Ax, const Tuple<const float, const Vec3 &> &ti) const
	{
		const float m = thrust::get<arg_m>(ti);
		const Vec3 &x = thrust::get<arg_x>(ti);

		return Ax + _1_h2 * m * x;
	}
};

struct Vec3_to_arr3_op : public thrust::unary_function<Tuple<const Vec3 &, float *>, void>
{
	// argument indices
	enum
	{
		arg_vec,
		arg_arr,
	};

	__host__ __device__ void operator()(Tuple<const Vec3 &, float *> ti) const
	{
		const Vec3 &vec = thrust::get<arg_vec>(ti);
		float *arr = thrust::get<arg_arr>(ti);

		arr[0] = vec[0], arr[1] = vec[1], arr[2] = vec[2];
	}
};

struct arr3_to_Vec3_op : public thrust::unary_function<float *, Vec3>
{
	__host__ __device__ Vec3 operator()(float *arr) const
	{
		return Eigen::Map<Vec3>(arr);
	}
};

void find_F_dF(SimCfg &scfg, collection<Vec3> &in, collection<float> &out)
{
	IncrItr idx(0);

	// find F
	// this extra iterator iterate through {vert0[i], ..., vert3[i], &F[9 * i]}
	auto scfg_itr_x0x1x2x3F = make_SimCfg_iterator(scfg, idx,
												   make_zipped_itr(
													   make_indexed_ref_itr(in, idx, scfg.tetrIdces, 0, 4),
													   make_indexed_ref_itr(in, idx, scfg.tetrIdces, 1, 4),
													   make_indexed_ref_itr(in, idx, scfg.tetrIdces, 2, 4),
													   make_indexed_ref_itr(in, idx, scfg.tetrIdces, 3, 4),
													   make_strided_itr(out, idx, 9)));
	// put Ds * dmInv into the compact array
	thrust::for_each_n(EXEC_POL, scfg_itr_x0x1x2x3F, scfg.cgBuffer.NT, find_F_dF_op());
}

void rectify_svd(SimCfg &scfg)
{
	IncrItr idx(0);

	SolverBuffer &buf = scfg.cgBuffer;

	collection<float> &arrU = buf.arrU;
	collection<float> &arrVS = buf.arrVS;
	collection<float> &arrV = buf.arrV;

	// fix bad svds
	auto svd_itr_UvSV = make_zipped_itr(make_strided_itr(arrU, idx, 9),
										make_strided_itr(arrVS, idx, 3),
										make_strided_itr(arrV, idx, 9));
	thrust::for_each_n(EXEC_POL, svd_itr_UvSV, buf.NT, rectify_SVD_op(scfg.sThresh));
}

using fptr_itr = StridedPtrItr<float>;
using mult_K_itr = TetrItr<ZippedItr<fptr_itr, fptr_itr, fptr_itr,
									 fptr_itr, TetrConstfloatItr, fptr_itr>>;
inline mult_K_itr make_mul_K_itr(SimCfg &scfg, collection<float> &arrU, collection<float> &arrVS, collection<float> &arrV,
								 collection<float> &dF, collection<float> &Kx, const float scale)
{
	IncrItr idx(0);

	return make_SimCfg_iterator(scfg, idx,
								make_zipped_itr(
									make_strided_itr(arrU, idx, 9),
									make_strided_itr(arrVS, idx, 3),
									make_strided_itr(arrV, idx, 9),
									make_strided_itr(dF, idx, 9),
									thrust::make_constant_iterator(scale),
									make_strided_itr(Kx, idx, 12)));
}

void compute_f(SimCfg &scfg, collection<float> &arrU, collection<float> &arrVS, collection<float> &arrV,
			   collection<float> &arrFv, collection<float> &arrOut)
{
	IncrItr idx(0);

	const unsigned NT = scfg.cgBuffer.NT;
	const float gamma = scfg.gamma;

	// zero-out output
	thrust::fill(arrOut.begin(), arrOut.end(), 0);

	// add elastic force
	// Fv doesn't do anything here
	thrust::for_each_n(EXEC_POL, make_mul_K_itr(scfg, arrU, arrVS, arrV, arrFv, arrOut, 1), NT, add_prod_K_op(false));

	// add damping force
	// consider F (F of velocity) with Rayleigh damping
	thrust::for_each_n(EXEC_POL, make_mul_K_itr(scfg, arrU, arrVS, arrV, arrFv, arrOut, gamma), NT, add_prod_K_op(true));
}

void gather_partial_forces(SimCfg &scfg, collection<float> &in, collection<Vec3> &out)
{
	IncrItr idx(0);

	// permuted partial force and corresponding indices
	auto partial_force_ptr_itr = thrust::make_permutation_iterator(make_strided_itr(in, idx, 3), scfg.referingTetrIdces.begin());
	auto partial_force_vec_itr = thrust::make_transform_iterator(partial_force_ptr_itr, arr3_to_Vec3_op());

	// use tetr indices as key to reduce (+) forces
	const collection<unsigned> &key = scfg.referredVertIdces;
	thrust::reduce_by_key(EXEC_POL, key.begin(), key.end(), partial_force_vec_itr, thrust::make_discard_iterator(), out.begin());
}

void add_external_forces(collection<unsigned> &idx, collection<Vec3> &f, collection<Vec3> &out)
{
	auto affected_vert_itr_in = thrust::make_permutation_iterator(out.begin(), idx.begin());
	auto affected_vert_itr_out = thrust::make_permutation_iterator(out.begin(), idx.begin());

	thrust::transform(EXEC_POL, f.begin(), f.end(), affected_vert_itr_in, affected_vert_itr_out, thrust::plus<Vec3>());
}

class MultKFunc : public FEM::CG::PDFunc
{
	SimCfg &scfg;

public:
	MultKFunc(SimCfg &scfg) : scfg(scfg) {}

	void operator()(collection<Vec3> &x, collection<Vec3> &Ax) const
	{
		IncrItr idx(0);

		SolverBuffer &buf = scfg.cgBuffer;

		collection<float> &arrU = buf.arrU;
		collection<float> &arrVS = buf.arrVS;
		collection<float> &arrV = buf.arrV;
		collection<float> &arrDF = buf.arrF;
		collection<float> &arrH = buf.arrH;

		const unsigned NT = buf.NT;
		const float _1_gamma_dt = 1 + scfg.gamma / scfg.dt;

		// Ax = ((1 + gamma / dt) K + m / h^2) x

		// calculate dF for x
		find_F_dF(scfg, x, arrDF);

		// initlaize
		// zero-out partial forces
		thrust::fill(EXEC_POL, arrH.begin(), arrH.end(), 0);
		// add (1 + gamma / h) K x
		thrust::for_each_n(EXEC_POL, make_mul_K_itr(scfg, arrU, arrVS, arrV, arrDF, arrH, _1_gamma_dt), NT, add_prod_K_op(true));

		// add M / h^2
		auto scfg_itr_mx = make_zipped_itr(scfg.m.begin(), x.begin());

		gather_partial_forces(scfg, buf.arrH, Ax);

		thrust::transform(EXEC_POL, Ax.begin(), Ax.end(), scfg_itr_mx, Ax.begin(), add_M_h2_op(1 / (scfg.dt * scfg.dt)));
	}
};

void initBuffer(SimCfg &scfg)
{
	SolverBuffer &buf = scfg.cgBuffer;

	const unsigned NT = buf.NT = scfg.tetrKinds.size();
	const unsigned NV = buf.NV = scfg.x.size();
	const unsigned MAT_ARR_SIZE = NT * 9;
	const unsigned VEC_ARR_SIZE = NT * 3;

	buf.arrF = collection<float>(MAT_ARR_SIZE);
	buf.arrU = collection<float>(MAT_ARR_SIZE);
	buf.arrVS = collection<float>(VEC_ARR_SIZE);
	buf.arrV = collection<float>(MAT_ARR_SIZE);
	// H is for each tetrahedra's each vertex
	buf.arrH = collection<float>(NT * 12);

	buf.dx = collection<Vec3>(NV);
	buf.Adx = collection<Vec3>(NV);
	buf.q = collection<Vec3>(NV);
	buf.r = collection<Vec3>(NV);
}

void evolve(SimCfg &scfg)
{
	cudaDeviceSynchronize();
	SolverBuffer &buf = scfg.cgBuffer;

	// x += v * dt
	CG::saxpy(scfg.dt, scfg.v, scfg.x, scfg.x);
	cudaDeviceSynchronize();

	// calculate F for x and then svd
	find_F_dF(scfg, scfg.x, buf.arrF);
	cudaDeviceSynchronize();

	SVD::svd(buf.arrF, buf.arrU, buf.arrVS, buf.arrV);
	cudaDeviceSynchronize();
	rectify_svd(scfg);
	cudaDeviceSynchronize();
#ifdef PRINT_DBG
	std::cout << "after rectify_svd:" << std::endl;
	std::cout << "F:" << std::endl;
	printMat<Mat3>(buf.arrF);
#endif
	// calculate F for v
	find_F_dF(scfg, scfg.v, buf.arrF);
	cudaDeviceSynchronize();
#ifdef PRINT_DBG
	std::cout << "after find_F_dF:" << std::endl;
	std::cout << "dF:" << std::endl;
	printMat<Mat3>(buf.arrF);
#endif
	// add force
	compute_f(scfg, buf.arrU, buf.arrVS, buf.arrV, buf.arrF, buf.arrH);
	cudaDeviceSynchronize();
#ifdef PRINT_DBG
	std::cout << "after compute_f:" << std::endl;
	std::cout << "H:";
	printMat<Mat34>(buf.arrH);
#endif
	// gather partial forces of a vertex from each referring tetrahedra
	gather_partial_forces(scfg, buf.arrH, buf.Adx);
	cudaDeviceSynchronize();
	add_external_forces(scfg.extForceIdx, scfg.extForce, buf.Adx);
	cudaDeviceSynchronize();
#ifdef PRINT_DBG
	std::cout << "gathered forces:" << std::endl;
	for (int i = 0; i < buf.Adx.size(); i++)
	{
		std::cout << "Adx[" << i << "]:\n";
		printMat(buf.Adx[i]);
	}
#endif

	// solve CG
	thrust::fill(buf.dx.begin(), buf.dx.end(), Vec3::Zero());
	CG::solveCG(buf, MultKFunc(scfg), buf.dx, buf.Adx, scfg.constrained);

	// v += dx / dt
	CG::saxpy(1 / scfg.dt, buf.dx, scfg.v, scfg.v);

	// x += dx
	CG::saxpy(1, buf.dx, scfg.x, scfg.x);

	// CG::saxpy(1 * scfg.dt, buf.Adx, scfg.v, scfg.v);
}

} // namespace Solver
} // namespace FEM
