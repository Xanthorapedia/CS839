#pragma once

#include <Eigen/Core>

namespace cuCG
{

using Vec3 = Eigen::Vector3f;
using Mat3 = Eigen::Matrix3f;

// # of device threads
const int N_THREAD = 1024;

// # of maximum CG iterations
const int CG_MAX_ITERATION = 100;

struct SimCfg
{
	unsigned *tetrIdcs, *tetrKind;
	Vec3 *x, *v;
	unsigned N_PART, N_TETR, N_CELL;
	Mat3 *dmInvs;
	float *m, *restVs;
	// indices in x and v that corresponds to constrained vertices
	unsigned *constrained;
	unsigned N_CONSTR;

	float mu, lambda;
	float gamma;
	float t, dt;

	Vec3 *dx, *b, *p, *q, *r;
	float *scratch_arr;
	unsigned *mutex;
};
	
} // namespace cuCG
