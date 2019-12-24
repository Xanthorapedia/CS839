#pragma once

#include "Animesh.hpp"
#include "SceneControl.hpp"
#include "MeshLoader.hpp"

#include "FEM/FEM.h"

#include "AABB.h"

#include <Eigen/Dense>
#include <unordered_map>

namespace Lattice
{

using Vec3 = Eigen::Vector3f;
using Mat3 = Eigen::Matrix3f;

using Animesh = CS839::Animesh;
using ParsedMesh = CS839::ParsedMesh;

const float HASH_EPS = 1e-6f;

// hash function for Vec3
struct vec3HashFunc
{
	size_t operator()(const Vec3 &vec) const
	{
		size_t h1 = std::hash<float>()(std::roundf(vec.x() / HASH_EPS) * HASH_EPS);
		size_t h2 = std::hash<float>()(std::roundf(vec.y() / HASH_EPS) * HASH_EPS);
		size_t h3 = std::hash<float>()(std::roundf(vec.z() / HASH_EPS) * HASH_EPS);
		return (h1 ^ (h2 << 1)) ^ h3;
	}
};

// equals function for Vec3
struct vec3EqualsFunc
{
	bool operator()(const Vec3 &lhs, const Vec3 &rhs) const
	{
		return lhs.isApprox(rhs, HASH_EPS);
	}
};

using Vec3Map = std::unordered_map<Vec3, unsigned, vec3HashFunc, vec3EqualsFunc>;

template <int N_TETR, int N_VERT>
class Lattice
{
protected:
	// Mesh info
	Animesh &animesh;
	VtIntArray &fvCnts, &fvIdcs;
	VtVec3fArray &verts;
	bool showGrid;

	// Mesh info
	std::vector<Vec3> latticeWeights;
	std::vector<unsigned> containerIdcs;
	// (center) -> tetrahedron index
	Vec3Map latticeTetrIdxMap;

	AABBTree objTriTree;

private:
	inline int _addLatticeParticle(const Vec3 &lattice, bool showGrid, std::vector<Vec3> &x)
	{
		int idx = x.size();
		if (showGrid)
			verts.push_back(GfVec3f(lattice.data()));
		x.push_back(lattice);
		return idx;
	}

	void linkTetrahedron(Vec3 &pos, unsigned &tetrStart, Vec3 &weights, const FEM::LatticeCfg &lcfg)
	{
		Vec3 lattice = snapToGrid(pos);

		// locate cell
		unsigned cellTetrStart = -1;
		auto foundcellTetrStart = latticeTetrIdxMap.find(lattice);
		if (foundcellTetrStart != latticeTetrIdxMap.end())
			cellTetrStart = foundcellTetrStart->second;
		else
			throw std::logic_error("Unknown lattice");

		// check for each tetrahedron if the coordinate is inside
		for (tetrStart = cellTetrStart; tetrStart < cellTetrStart + N_TETR * 4; tetrStart += 4)
		{
			const Vec3 &x0 = lcfg.x[lcfg.tetrIdces[tetrStart]];
			const Mat3 &DmInv = lcfg.dmInvs[lcfg.tetrKinds[tetrStart >> 2]];
			// local coordinate in tetrahedron
			weights = DmInv * (pos - x0);

			if (weights.minCoeff() >= 0 && weights.sum() <= 1)
				break;
		}

		// if no suitable tetr found (tetrStart is out of range),
		// use the first one in cell
		if (tetrStart - cellTetrStart == N_TETR * 4)
			tetrStart = cellTetrStart;
	}

public:
	Lattice(
		// the mesh
		Animesh &mesh,
		// unit centers
		const std::vector<Vec3> &lattices,
		// vertex displacements
		const std::array<Vec3, N_VERT> &corners,
		const float gridsz,
		const float density,
		// the indices of each tetrahedron
		const std::array<std::array<unsigned, 4>, N_TETR> &tetrIdxGroups,
		// the indices of each face
		const std::vector<std::vector<unsigned>> &faceIdxGroups,
		// output
		FEM::LatticeCfg &lcfg,
		bool showGrid = true)
		: animesh(mesh),
		  fvCnts(mesh.getFaceVertexCounts()),
		  fvIdcs(mesh.getfaceVertexIndices()),
		  verts(mesh.getVertices()),
		  showGrid(showGrid)
	{
		initParticles(lattices, corners, gridsz, density, tetrIdxGroups, faceIdxGroups, lcfg, showGrid);
	}

	Lattice(
		// the mesh
		Animesh &mesh,
		bool showGrid = true)
		: animesh(mesh),
		  fvCnts(animesh.getFaceVertexCounts()),
		  fvIdcs(animesh.getfaceVertexIndices()),
		  verts(animesh.getVertices()),
		  showGrid(showGrid)
	{
	}

	// find out the lattice grid
	virtual Vec3 snapToGrid(Vec3 &pos) = 0;

	virtual std::vector<Vec3> determineGrid(const ParsedMesh &objMesh) { return std::vector<Vec3>(0); }

	void initParticles(
		const ParsedMesh &objMesh,
		// vertex displacements
		const std::array<Vec3, N_VERT> &corners,
		const float gridsz,
		const float density,
		// the indices of each tetrahedron
		const std::array<std::array<unsigned, 4>, N_TETR> &tetrIdxGroups,
		// the indices of each face
		const std::vector<std::vector<unsigned>> &faceIdxGroups,
		FEM::LatticeCfg &lcfg,
		const bool showGrid)
	{
		// create AABB tree
		Point p[3];
		std::vector<Triangle> &meshTris = objTriTree.getTriangleList();
		for (auto &&face : objMesh.faces)
		{
			for (int i = 0; i < face.size(); i++)
			{
				auto v = objMesh.verts[face[i]];
				p[i] = Point(v[0], v[1], v[2]);
			}
			meshTris.push_back(Triangle(p[0], p[1], p[2]));
		}

		objTriTree.insert();

		initParticles(determineGrid(objMesh), corners, gridsz, density, tetrIdxGroups, faceIdxGroups, lcfg, showGrid);

		// load vertices
		for (auto &vert : objMesh.verts)
		{
			// add the vertex to mesh
			// verts.push_back(vert);

			// find container info
			unsigned tetrStartIdx = -1;
			Vec3 weights;
			linkTetrahedron(Vec3(vert.data()), tetrStartIdx, weights, lcfg);
			containerIdcs.push_back(tetrStartIdx);
			latticeWeights.push_back(weights);
		}

		// // load faces
		// for (auto &&face : objMesh.faces)
		// {
		// 	for (auto &&vert : face)
		// 		fvIdcs.push_back(vert + meshVertStart);
		// 	fvCnts.push_back(face.size());
		// }
	}

	void initParticles(
		// unit centers
		const std::vector<Vec3> &lattices,
		// vertex displacements
		const std::array<Vec3, N_VERT> &corners,
		const float gridsz,
		const float density,
		// the indices of each tetrahedron
		const std::array<std::array<unsigned, 4>, N_TETR> &tetrIdxGroups,
		// the indices of each face
		const std::vector<std::vector<unsigned>> &faceIdxGroups,
		FEM::LatticeCfg &lcfg,
		const bool showGrid)
	{
		// (vertex) -> index in lcfg.x
		Vec3Map cornerIdxMap;

		// pre-compute rest shape for each kind of tetrahedron
		for (int i = 0; i < N_TETR; i++)
		{
			auto &&tetrIdxGroup = tetrIdxGroups[i];
			const Vec3 &X0 = corners[tetrIdxGroup[0]] * gridsz;
			// Dm = (X1 - X0 | X1 - X0 | X3 - X0)
			Mat3 Dm;
			for (size_t i = 1; i < 4; i++)
				Dm.col(i - 1) = corners[tetrIdxGroup[i]] * gridsz - X0;

			float restVolume = Dm.determinant() / 6;
			if (restVolume < 0)
				throw std::logic_error("Inverted element");

			lcfg.dmInvs.push_back(Dm.inverse().eval());
			lcfg.restVs.push_back(restVolume);
		}

		for (auto &&lattice : lattices)
		{
			// the cornor indices of the CubicLattice with 0 at lattice
			unsigned cornerIdx[N_VERT];

			// figure out the index of each corner
			for (int i = 0; i < N_VERT; i++)
			{
				Vec3 &&corner = lattice + corners[i] * gridsz;

				// find corner idx
				auto cachedCornerIdx = cornerIdxMap.find(corner);
				if (cachedCornerIdx == cornerIdxMap.end())
				{
					// add missing corner to the list
					cornerIdx[i] = _addLatticeParticle(corner, showGrid, lcfg.x);
					cornerIdxMap.insert({corner, cornerIdx[i]});
					lcfg.m.push_back(0);
				}
				else
					cornerIdx[i] = cachedCornerIdx->second;
			}

			// create faces for tet
			if (showGrid)
				for (auto &&faceIdxGroup : faceIdxGroups)
				{
					for (auto &&faceIdx : faceIdxGroup)
						fvIdcs.push_back(cornerIdx[faceIdx]);
					fvCnts.push_back(faceIdxGroup.size());
				}

			// register the cell's tetrahedrons' starting index
			latticeTetrIdxMap.insert({lattice, lcfg.tetrIdces.size()});
			// register rest shape for the master corner of each tetrahedron
			for (unsigned i = 0; i < N_TETR; i++)
			{
				for (unsigned j = 0; j < 4; j++)
				{
					// distribute mass across tetrahedra references
					lcfg.m[cornerIdx[tetrIdxGroups[i][j]]] += density * lcfg.restVs[i] / 4;
				}

				lcfg.tetrIdces.insert(lcfg.tetrIdces.end(), {cornerIdx[tetrIdxGroups[i][0]], cornerIdx[tetrIdxGroups[i][1]],
															 cornerIdx[tetrIdxGroups[i][2]], cornerIdx[tetrIdxGroups[i][3]]});
				lcfg.tetrKinds.push_back(i);
			}
		}
	}

	void savePoints(const std::vector<Vec3> &latticeX, const FEM::LatticeCfg &lcfg, VtVec3fArray &objMeshVerts)
	{
		for (unsigned i = 0; i < verts.size(); i++)
		{
			// update grid points
			verts[i].Set(latticeX[i].data());
		}

		for (unsigned i = 0; i < objMeshVerts.size(); i++)
		{
			unsigned tetrStart = containerIdcs[i];
			const Vec3 &weights = latticeWeights[i];
			const Vec3 &x0 = latticeX[lcfg.tetrIdces[tetrStart + 0]];
			const Vec3 &x1 = latticeX[lcfg.tetrIdces[tetrStart + 1]];
			const Vec3 &x2 = latticeX[lcfg.tetrIdces[tetrStart + 2]];
			const Vec3 &x3 = latticeX[lcfg.tetrIdces[tetrStart + 3]];
			const Vec3 &interpolated = (x1 - x0) * weights(0) +
									   (x2 - x0) * weights(1) +
									   (x3 - x0) * weights(2) +
									   x0;
			// interpolate in local coordinate
			objMeshVerts[i].Set(interpolated.data());
		}
	}

	// // find collision of query point with the mesh
	// // returns the offended tetrahedron index and
	// // the collision vector (direction and distance of offense)
	// unsigned meshCollide(const FEM::LatticeCfg &lcfg, const Vec3 &query, Vec3 &collVec)
	// {
	// 	Point q(query.x(), query.y(), query.z());

	// 	// first check if the point is inside
	// 	if (!objTriTree.isInside(query))
	// 	{
	// 		collVec.setZero();
	// 		return -1;
	// 	}

	// 	unsigned fid, tid;
	// 	collVec = objTriTree.vecToSurface(query, fid, tid);

	// 	unsigned affectedMeshVertexIdx = fid * 4 + tid;
	// 	containerIdcs[affectedMeshVertexIdx]
	// 	latticeWeights[affectedMeshVertexIdx]
	// 	return fvIdcs[fid * 4 + tid + meshVertStart];
	// }
};
} // namespace Lattice
