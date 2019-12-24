#pragma once

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace FEM
{

template <typename... Ts>
using Tuple = thrust::tuple<Ts...>;

// tuple data types
using TetrVecs = Tuple<Vec3 &, Vec3 &, Vec3 &, Vec3 &>;
using Tetrfloats = Tuple<float, float, float, float>;

using TetrDm = Tuple<Mat3 &, float>;
enum
{
	T_D_dmInv,
	T_D_restV,
};

using TetrConstfloats = Tuple<float, float, float, float, float>;
enum
{
	T_C_dt,
	T_C_lambda,
	T_C_mu,
	T_C_gamma,
};

// iterator types
// {0, 1, 2, ...}
using IncrItr = thrust::counting_iterator<unsigned>;

// a bunch of iterators
template <typename... Iterators>
using ZippedItr = thrust::zip_iterator<Tuple<Iterators...>>;

template <typename... Itrs>
inline ZippedItr<Itrs...> make_zipped_itr(Itrs... iterators)
{
	return thrust::make_zip_iterator(thrust::make_tuple(iterators...));
}

// extracts information from a collection type by index
// tetrIdces remembers the index of the information in storage of the offest'th
// information of the idx'th tetrahedron
template <typename T>
struct by_idx_op : public thrust::unary_function<const unsigned, T &>
{
	// only raw type is allowed in kernel
	T *storage;
	const unsigned *indexPkg, groupSz, offset;

	by_idx_op(collection<T> &storage, const collection<unsigned> &indexPkg,
			  const unsigned offset, const unsigned groupSz)
		: storage(thrust::raw_pointer_cast(storage.data())),
		  indexPkg(thrust::raw_pointer_cast(indexPkg.data())),
		  groupSz(groupSz), offset(offset) {}

	__host__ __device__ T &operator()(const unsigned tetrIdx) const
	{
		return storage[indexPkg[tetrIdx * groupSz + offset]];
	}
};

// {T &}
template <typename T>
using IndexedRefItr = thrust::transform_iterator<by_idx_op<T>, IncrItr>;

template <typename T>
inline IndexedRefItr<T> make_indexed_ref_itr(collection<T> &storage, IncrItr &idx,
									  const collection<unsigned> &indexPkg,
									  const unsigned offset = 0, const unsigned groupSz = 1)
{
	return thrust::make_transform_iterator(idx, by_idx_op<T>(storage, indexPkg, offset, groupSz));
}

// {Vec3}
using TetrVecItr = IndexedRefItr<Vec3>;
// {Mat3}
using TetrMatItr = IndexedRefItr<Mat3>;
// {float}
using TetrfloatItr = IndexedRefItr<float>;
// {const float}
using TetrConstfloatItr = thrust::constant_iterator<float>;
// {(Vec3, Vec3, Vec3, Vec3)}
using Tetr4VecItr = ZippedItr<TetrVecItr, TetrVecItr, TetrVecItr, TetrVecItr>;
// {(float, float, float, float)}
using Tetr4floatItr = ZippedItr<TetrfloatItr, TetrfloatItr, TetrfloatItr, TetrfloatItr>;
// {(Mat3, float)}
using TetrDmItr = ZippedItr<TetrMatItr, TetrfloatItr>;
// {(float, float, float, float, float)}
using TetrConstItr = ZippedItr<TetrConstfloatItr, TetrConstfloatItr, TetrConstfloatItr,
							   TetrConstfloatItr, TetrConstfloatItr>;

template <typename XtraIterator>
using TetrItr = ZippedItr<Tetr4VecItr, Tetr4VecItr, Tetr4floatItr, TetrDmItr, TetrConstItr, XtraIterator>;

template <typename Extra>
using tetrIdces = Tuple<TetrVecs, TetrVecs, Tetrfloats, TetrDm, TetrConstfloats, Extra>;
// indices for selecting tetrahedra information in iterator tuple
enum
{
	T_X, // get particleX
	T_V, // get particleV
	T_M, // get mass
	T_D, // get restV and dmInv
	T_C, // get constants
	T_E, // get extra collections
};

using tetrIdcesPlaint = tetrIdces<thrust::null_type>;

template <typename T>
struct get_seg_start_op : public thrust::unary_function<const unsigned, T *>
{
	// only raw type is allowed in kernel
	T *storage;
	const unsigned stride;

	get_seg_start_op(collection<T> &storage, const unsigned stride = 1)
		: storage(thrust::raw_pointer_cast(storage.data())),
		  stride(stride) {}

#ifndef HOST_EXEC
	get_seg_start_op(thrust::host_vector<T> &storage, const unsigned stride = 1)
		: storage(thrust::raw_pointer_cast(storage.data())),
		  stride(stride) {}
#endif // HOST_EXEC

	__host__ __device__ T *operator()(const unsigned idx) const
	{
		return storage + idx * stride;
	}
};

// the iterator that iterates through collection<T> and gives the pointer to elements with some stride
template <typename T>
using StridedPtrItr = thrust::transform_iterator<get_seg_start_op<T>, IncrItr>;

template <typename T>
inline StridedPtrItr<T> make_strided_itr(collection<T> &storage, IncrItr &idx, const unsigned stride = 1)
{
	return thrust::make_transform_iterator(idx, get_seg_start_op<T>(storage, stride));
}
template <typename T>
inline StridedPtrItr<T> make_strided_itrh(thrust::host_vector<T> &storage, IncrItr &idx, const unsigned stride = 1)
{
	return thrust::make_transform_iterator(idx, get_seg_start_op<T>(storage, stride));
}

template <typename XtraIterator = thrust::null_type>
inline TetrItr<XtraIterator> make_SimCfg_iterator(SimCfg &scfg, IncrItr &idx, XtraIterator &exItr = XtraIterator())
{
	std::vector<TetrVecItr> xItr, vItr;
	std::vector<TetrfloatItr> mItr;
	// for vertex indexed i in its tetr
	for (size_t i = 0; i < 4; i++)
	{
		xItr.push_back(make_indexed_ref_itr(scfg.x, idx, scfg.tetrIdces, i, 4));
		vItr.push_back(make_indexed_ref_itr(scfg.v, idx, scfg.tetrIdces, i, 4));
		mItr.push_back(make_indexed_ref_itr(scfg.m, idx, scfg.tetrIdces, i, 4));
	}

	TetrItr<XtraIterator> tetrItr = make_zipped_itr(
		// particleX (T_X)
		make_zipped_itr(xItr[0], xItr[1], xItr[2], xItr[3]),
		// particleV(T_V)
		make_zipped_itr(vItr[0], vItr[1], vItr[2], vItr[3]),
		// mass (T_M)
		make_zipped_itr(mItr[0], mItr[1], mItr[2], mItr[3]),
		// dmInverse and restV (T_D)
		make_zipped_itr(
			make_indexed_ref_itr(scfg.dmInvs, idx, scfg.tetrKinds),
			make_indexed_ref_itr(scfg.restVs, idx, scfg.tetrKinds)),
		// constants (T_C)
		make_zipped_itr(
			thrust::make_constant_iterator(scfg.dt),
			thrust::make_constant_iterator(scfg.lambda),
			thrust::make_constant_iterator(scfg.mu),
			thrust::make_constant_iterator(scfg.gamma),
			thrust::make_constant_iterator(scfg.sThresh)),
		// extra iterators (T_E)
		exItr);

	return tetrItr;
}

} // namespace FEM
