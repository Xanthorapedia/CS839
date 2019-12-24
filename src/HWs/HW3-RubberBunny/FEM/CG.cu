#include "CG.h"
#include "ThrustUtils.h"

namespace FEM
{
namespace CG
{

struct saxpy_op : public thrust::binary_function<Vec3, Vec3, Vec3>
{
	const float c;

	saxpy_op(float c) : c(c) {}

	__host__ __device__ Vec3 operator()(const Vec3 &x, const Vec3 &y) const
	{
		return c * x + y;
	}
};

struct tuple_dot_op : public thrust::unary_function<thrust::tuple<Vec3, Vec3>, float>
{
	__host__ __device__ float operator()(const thrust::tuple<Vec3, Vec3> &pair) const
	{
		return thrust::get<0>(pair).dot(thrust::get<1>(pair));
	}
};

struct one_norm_op : public thrust::unary_function<Vec3, float>
{
	__host__ __device__ float operator()(const Vec3 &vec) const
	{
		return vec.cwiseAbs().maxCoeff();
	}
};

// calculates {z} = {c * x + y}
void saxpy(const float c, collection<Vec3> &x, collection<Vec3> &y, collection<Vec3> &z)
{
	thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), saxpy_op(c));
}

// calculates sum{<x, y>}
float dot_prod(collection<Vec3> &x, collection<Vec3> &y)
{
	// make pairs (x, y), transform by tuple_dot into float and reduce by sum
	return thrust::transform_reduce(make_zipped_itr(x.begin(), y.begin()),
									make_zipped_itr(x.end(), y.end()),
									tuple_dot_op(), 0.0f, thrust::plus<float>());
}

// calculates max{||x||_1}
float oneNorm(collection<Vec3> &x)
{
	// make pairs (x, y), transform by tuple_dot into float and reduce by sum
	return thrust::transform_reduce(x.begin(), x.end(),
									one_norm_op(), 0.0f, thrust::maximum<float>());
}

void clearConstrained(collection<Vec3> &x, const collection<unsigned> &constraints)
{
	// clear out constrained elements
	thrust::fill_n(thrust::make_permutation_iterator(x.begin(), constraints.begin()),
				   constraints.size(), Vec3::Zero());
}

void solveCG(SolverBuffer &buf, const PDFunc &A, collection<Vec3> &x, collection<Vec3> &p,
			 const collection<unsigned> &constraints,
			 const float eps, const unsigned max_itr)
{
	collection<Vec3> &q = buf.q,
					 &r = buf.r;

	float rho_old, r_norm;

	// A * x0
	A(x, q);

	// r = - A * dx0 + b
	saxpy(-1, q, p, r);

	// project r
	clearConstrained(r, constraints);

	unsigned i;
	for (i = 0; i < max_itr; i++)
	{
		r_norm = oneNorm(r);

		if (r_norm < eps)
			break;

		// rho = r * r
		float rho = dot_prod(r, r);

		// p = rho / rho_old * p + r
		saxpy((i == 0 ? 0 : rho / rho_old), p, r, p);

		// q = A * p
		A(p, q);

		// Project q
		clearConstrained(q, constraints);

		float p_dot_q = dot_prod(p, q);
		float alpha = rho / p_dot_q;
		if (p_dot_q <= 0)
			printf("CG: matrix appears indefinite or singular, p_dot_q/p_dot_p= %f\n",
				   p_dot_q / dot_prod(p, p));

		// x = +alpha * p + x
		saxpy(+alpha, p, x, x);

		// r = -alpha * q + r
		saxpy(-alpha, q, r, r);

		rho_old = rho;
	}

	if (i == CG_MAX_ITERATION)
		printf("CG: exploded with norm(r) = %f after %d iterations.\n", r_norm, CG_MAX_ITERATION);
}

} // namespace CG
} // namespace FEM
