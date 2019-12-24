#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

#include <Eigen/Core>

namespace FEM
{

using Vec3 = Eigen::Vector3f;
using Mat3 = Eigen::Matrix3f;
using Mat34 = Eigen::Matrix<float, 3, 4>;
using Vec3View = Eigen::Map<Vec3>;
using Mat3View = Eigen::Map<Mat3>;
using Mat34View = Eigen::Map<Mat34>;
using ConstVec3View = Eigen::Map<const Vec3>;
using ConstMat3View = Eigen::Map<const Mat3>;

// #define HOST_EXEC

template <class T>
#ifdef HOST_EXEC
using collection = thrust::host_vector<T, std::allocator<T>>;
const auto EXEC_POL = thrust::host;
#else
using collection = thrust::device_vector<T, thrust::device_allocator<T>>;
const auto EXEC_POL = thrust::device;
#endif

struct SolverBuffer
{
	unsigned NT, NV;
	// underlying storage
	collection<float> arrF, arrU, arrVS, arrV, arrH;
	// CG buffers
	collection<Vec3> dx, Adx, q, r;
};

struct SimCfg
{
	// per-vertex information
	collection<Vec3> x, v;
	collection<float> m;
	// the index of vertices a partial force corresponds to
	collection<unsigned> referredVertIdces;
	// for each index in the H array, which vertex should the partial force be
	// added to
	collection<unsigned> referingTetrIdces;

	// per-tetrahedron information
	// index of v0...3
	collection<unsigned> tetrIdces;
	collection<unsigned> tetrKinds;

	// per-kind information
	collection<Mat3> dmInvs;
	collection<float> restVs;

	// indices in x and v that corresponds to constrained vertices
	collection<unsigned> constrained;

	// which vertex receives which force
	collection<unsigned> extForceIdx;
	collection<Vec3> extForce;

	float sThresh;
	float mu, lambda;
	float gamma;
	float t, dt;

	SolverBuffer cgBuffer;
};

} // namespace FEM
