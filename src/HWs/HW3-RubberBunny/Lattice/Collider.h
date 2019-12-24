#pragma once

#include <Eigen/Dense>
#include "HWIncludes.hpp"

#include <igl/AABB.h>

namespace Lattice
{

using Vec3 = Eigen::Vector3f;

class Collider
{
private:
	// elements for mesh surface and tetr collider
	Eigen::MatrixXi F, T;
	igl::AABB<MatrixXf, 3> faceTree, tetrTree;

public:
	Collider();

	initMesh(std::vector<Vec3> &meshVerts, std::vector<unsigned> &fVerts)
	{
		F = Map<Eigen::MatrixXi>(fVerts.data());
	}

	initCollidingElement(std::vector<Vec3> &tetrVerts, std::vector<unsigned> &tVerts)
	{
		T = Map<Eigen::MatrixXi>(tVerts.data());
	}

	void update(std::vector<Vec3> &verts)
	{
	}
};

} // namespace Lattice
