#pragma once

#include "FEM.h"
#include "Internals.h"

#include <thrust/copy.h>

namespace FEM
{
namespace Solver
{

void initBuffer(SimCfg &scfg);
void evolve(SimCfg &scfg);

class BackEulerSolver
{
private:
	SimCfg scfg;
	LatticeCfg constraint_handle;
	ConstraintForceFunc &getCAndF;
	std::vector<Vec3> x_constr_buf;
	std::vector<Vec3> v_constr_buf;

public:
	BackEulerSolver(const FEM::LatticeCfg &lcfg,
					const float mu, const float lambda, const float gamma, const float sThresh,
					ConstraintForceFunc &constraintForceFunc = DEFAULT_CF)
		: getCAndF(constraintForceFunc)
	{
		scfg.x = lcfg.x;
		scfg.v = collection<Vec3>(lcfg.x.size(), Vec3::Zero());

		x_constr_buf.resize(scfg.x.size());
		v_constr_buf.resize(scfg.v.size());

		scfg.m = lcfg.m;
		scfg.referredVertIdces = collection<unsigned>(lcfg.tetrIdces.size());
		scfg.referingTetrIdces = collection<unsigned>(lcfg.tetrIdces.size());

		scfg.tetrIdces = lcfg.tetrIdces;
		scfg.tetrKinds = lcfg.tetrKinds;

		populate_refTetrIdx();

		scfg.dmInvs = lcfg.dmInvs;
		scfg.restVs = lcfg.restVs;

		scfg.sThresh = sThresh;
		scfg.mu = mu;
		scfg.lambda = lambda;
		scfg.gamma = gamma;

		scfg.t = scfg.dt = 0;

		initBuffer(scfg);
	}

	void update(const float dt)
	{
		scfg.dt = dt;
		scfg.t += dt;

		if (!getCAndF.skipFrame(scfg.t))
		{
			thrust::copy(scfg.x.begin(), scfg.x.end(), x_constr_buf.begin());
			thrust::copy(scfg.v.begin(), scfg.v.end(), v_constr_buf.begin());

			std::vector<unsigned> cidx;
			std::vector<unsigned> fidx;
			std::vector<Vec3> f;
			getCAndF.hasConstraint = getCAndF.hasConstraint = false;
			getCAndF(x_constr_buf, v_constr_buf, scfg.t, f, fidx, cidx);

			if (getCAndF.hasConstraint)
			{
				scfg.constrained.resize(cidx.size());
				if (cidx.size() > 0)
					thrust::copy(scfg.constrained.begin(), scfg.constrained.end(), cidx.begin());

				thrust::copy(x_constr_buf.begin(), x_constr_buf.end(), scfg.x.begin());
				thrust::copy(v_constr_buf.begin(), v_constr_buf.end(), scfg.v.begin());
			}

			if (getCAndF.hasForce)
			{
				scfg.extForceIdx.resize(fidx.size());
				scfg.extForce.resize(fidx.size());
				if (fidx.size() > 0)
				{
					thrust::copy(fidx.begin(), fidx.end(), scfg.extForceIdx.begin());
					thrust::copy(f.begin(), f.end(), scfg.extForce.begin());
				}
			}
		}

		evolve(scfg);
	}

	void getRslt(std::vector<Vec3> &particleX)
	{
		particleX.resize(scfg.x.size());
		thrust::copy(scfg.x.begin(), scfg.x.end(), particleX.begin());
	}

private:
	void populate_refTetrIdx()
	{
		// the following actions will overwrite this array
		scfg.referredVertIdces = scfg.tetrIdces;

		// fill with 0, 1, 2, 3, ...
		thrust::sequence(EXEC_POL, scfg.referingTetrIdces.begin(), scfg.referingTetrIdces.end());
		// sort the above by vertex index so that we have contiguous segments of referring tetraheron ids
		thrust::stable_sort_by_key(EXEC_POL, scfg.referredVertIdces.begin(), scfg.referredVertIdces.end(), scfg.referingTetrIdces.begin());

		// // count the # of numbers in each segment
		// thrust::reduce_by_key(EXEC_POL, tetrIdces.begin(), tetrIdces.end(), thrust::make_constant_iterator(1),
		// 					  thrust::make_discard_iterator(), scfg.referredVertIdces.begin());

		// |<------------------------------------- 4 * NT ------------------------------------->|
		// referingTetrIdces:
		// |...........................................|.....................................|...
		// |<---- indices of tetr referring vert0 ---->|<- indices of tetr referring vert1 ->|...
		// referredVertIdces:
		// |0000000000000000000000000000000000000000000|1111111111111111111111111111111111111|...
		// |<------------------ N0 ------------------->|<--------------- N1 ---------------->|...
	}
};

} // namespace Solver
} // namespace FEM
