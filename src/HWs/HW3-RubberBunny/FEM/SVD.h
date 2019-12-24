#pragma once

#include "Internals.h"

namespace FEM
{
namespace SVD
{

const float svd_tol = 1e-7f;
const float svd_maxSweeps = 50;

// (U, S, V) = svd(A)
void svd(collection<float> &A, collection<float> &U, collection<float> &S, collection<float> &V,
		 const float tol = svd_tol, const int maxSweeps = svd_maxSweeps);

void destroyCUSolverHandle();

} // namespace SVD
} // namespace FEM
