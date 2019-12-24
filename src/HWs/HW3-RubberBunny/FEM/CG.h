#pragma once

#include "Internals.h"

namespace FEM
{
namespace CG
{

// error threshold
const float CG_EPS = 1e-4f;

// # of maximum CG iterations
const int CG_MAX_ITERATION = 100;

// a possitive definite linear function A(x, b)
class PDFunc
{
public:
	virtual void operator()(collection<Vec3> &x, collection<Vec3> &Ax) const = 0;
};

// calculates {z} = {c * x + y}
void saxpy(const float c, collection<Vec3> &x, collection<Vec3> &y, collection<Vec3> &z);

// calculates sum{<x, y>}
float dot_prod(collection<Vec3> &x, collection<Vec3> &y);

// calculates max{||x||_1}
float oneNorm(collection<Vec3> &x);

void solveCG(SolverBuffer &buf, const PDFunc &A, collection<Vec3> &x, collection<Vec3> &b,
			 const collection<unsigned> &constraints,
			 const float eps = CG_EPS, const unsigned max_itr = CG_MAX_ITERATION);

} // namespace CG
} // namespace FEM
