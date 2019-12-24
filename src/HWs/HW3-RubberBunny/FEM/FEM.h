#pragma once

#include <vector>
#include <Eigen/Core>

namespace FEM
{
using Vec3 = Eigen::Vector3f;
using Mat3 = Eigen::Matrix3f;

namespace Solver
{

struct ConstraintForceFunc
{
	bool hasConstraint = false;
	bool hasForce = false;

	// tells the solver not to copy x, v, etc. since there is no constraint/force
	// in the current frame t
	virtual bool skipFrame(float t) { return false; };

	virtual void operator()(std::vector<Vec3> &x, std::vector<Vec3> &v, float t,
							std::vector<Vec3> &f, std::vector<unsigned> &fidx,
							std::vector<unsigned> &cidx){};
};

// only the C++ compiler holds this object
#ifdef __NVCC__
extern ConstraintForceFunc DEFAULT_CF;
#else
ConstraintForceFunc DEFAULT_CF;
#endif

} // namespace Solver

struct LatticeCfg
{
	// per-vertex information
	std::vector<Vec3> x;
	std::vector<Vec3> v;
	std::vector<float> m;

	// per-tetrahedron information
	// index of v0...3
	std::vector<unsigned> tetrIdces;
	std::vector<unsigned> tetrKinds;

	// per-kind information
	std::vector<Mat3> dmInvs;
	std::vector<float> restVs;

	LatticeCfg()
	{
		x = std::vector<Vec3>();
		v = std::vector<Vec3>();
		m = std::vector<float>();
		tetrIdces = std::vector<unsigned>();
		tetrKinds = std::vector<unsigned>();
		dmInvs = std::vector<Mat3>();
		restVs = std::vector<float>();
	}
};

} // namespace FEM
