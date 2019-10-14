#pragma once

#include "Animesh.hpp"
#include "SceneControl.hpp"
#include "BackwardEuler.hpp"
#include "MeshLoader.hpp"

#include <Eigen/Dense>
#include <unordered_map>

#define HASH_EPS 1e-4
#define TETR_KIND_IDX 4

using Animesh = CS839::Animesh;
using ParsedMesh = CS839::ParsedMesh;

// hash function for Vec3
struct vec3HashFunc
{
	using Vec3 = Eigen::Vector3f;

	size_t operator()(const Vec3 &vec) const
	{
		size_t h1 = std::hash<float>()(vec.x());
		size_t h2 = std::hash<float>()(vec.y());
		size_t h3 = std::hash<float>()(vec.z());
		return (h1 ^ (h2 << 1)) ^ h3;
	}
};

// equals function for Vec3
struct vec3EqualsFunc
{
	using Vec3 = Eigen::Vector3f;
	bool operator()(const Vec3 &lhs, const Vec3 &rhs) const
	{
		return lhs.isApprox(rhs, HASH_EPS);
	}
};

typedef std::unordered_map<Eigen::Vector3f, int, vec3HashFunc, vec3EqualsFunc> Vec3Map;

template <int N_TETR, int N_VERT>
class Lattice
{
public:
	using Vec3 = Eigen::Vector3f;
	using Mat3 = Eigen::Matrix3f;

protected:
	// Mesh info
	Animesh &animesh;
	VtIntArray &fvCnts, &fvIdcs;
	VtVec3fArray &verts;
	bool showGrid;

	// Particle info
	std::vector<Vec3> particleX, particleV;
	std::vector<float> masses;
	std::array<Mat3, N_TETR> tetrDmInvs;
	std::array<float, N_TETR> tetrRestVs;
	// { x0, x1, x2, x3, tetr_kind }
	std::vector<std::array<unsigned, 5>> tetrIdcs;
	Vec3Map cornerIdxMap;
	Vec3Map latticeTetrIdxMap;

	// Mesh info
	unsigned meshStart;
	std::vector<unsigned> containerIdcs;
	std::vector<Vec3> latticeWeights;

	BackwardEuler::Solver<N_TETR> solver;

	inline int Lattice<N_TETR, N_VERT>::_addLatticeParticle(const Vec3 &lattice, bool showGrid)
	{
		int idx = particleX.size();
		if (showGrid)
			verts.push_back(GfVec3f(lattice.data()));
		particleX.push_back(lattice);
		particleV.push_back(Vec3::Zero());
		return idx;
	}

public:
	Lattice(
		// the mesh
		Animesh &mesh,
		// unit centers
		const std::vector<Vec3> &lattices,
		// vertex displacements
		const std::array<Vec3, N_VERT> &corners,
		float gridsz,
		// the indices of each tetrahedron
		const std::array<std::array<unsigned, 4>, N_TETR> &tetrIdxGroups,
		// the indices of each face
		const std::vector<std::vector<unsigned>> &faceIdxGroups,
		const float mu = 10.0,
		const float lambda = 10.0,
		const float rayleighCoefficient = 1e-7,
		const float mass = 100,
		bool showGrid = true)
		: animesh(mesh),
		  fvCnts(animesh.getFaceVertexCounts()),
		  fvIdcs(animesh.getfaceVertexIndices()),
		  verts(animesh.getVertices()),
		  showGrid(showGrid)
	{
	}

	Lattice(
		// the mesh
		Animesh &mesh,
		bool showGrid = false)
		: animesh(mesh),
		  fvCnts(animesh.getFaceVertexCounts()),
		  fvIdcs(animesh.getfaceVertexIndices()),
		  verts(animesh.getVertices()),
		  showGrid(showGrid)
	{
	}

	// find out the lattice grid
	virtual Vec3 snapToGrid(Vec3 &pos) = 0;

	void Lattice<N_TETR, N_VERT>::initParticles(
		// unit centers
		const std::vector<Vec3> &lattices,
		// vertex displacements
		const std::array<Vec3, N_VERT> &corners,
		const float gridsz,
		const float particleMass,
		// the indices of each face
		const std::vector<std::vector<unsigned>> &faceIdxGroups,
		// the indices of each tetrahedron
		const std::array<std::array<unsigned, 4>, N_TETR> &tetrIdxGroups,
		bool showGrid)
	{
		// pre-compute rest shape for each kind of tetrahedron
		for (int i = 0; i < N_TETR; i++)
		{
			auto &&tetrIdxGroup = tetrIdxGroups[i];
			const Vec3 &X0 = corners[tetrIdxGroup[0]] * gridsz;
			// Dm = (X1 - X0 | X1 - X0 | X3 - X0)
			Mat3 Dm;
			for (size_t i = 1; i < TETR_KIND_IDX; i++)
				Dm.col(i - 1) = corners[tetrIdxGroup[i]] * gridsz - X0;

			float restVolume = Dm.determinant() / 6;
			if (restVolume < 0)
				throw std::logic_error("Inverted element");

			tetrDmInvs[i] = Dm.inverse().eval();
			tetrRestVs[i] = restVolume;
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
					cornerIdx[i] = _addLatticeParticle(corner, showGrid);
					cornerIdxMap.insert({corner, cornerIdx[i]});
					masses.push_back(particleMass);
				}
				else
					cornerIdx[i] = cachedCornerIdx->second;
			}

			// create faces
			if (showGrid)
				for (auto &&faceIdxGroup : faceIdxGroups)
				{
					for (auto &&faceIdx : faceIdxGroup)
						fvIdcs.push_back(cornerIdx[faceIdx]);
					fvCnts.push_back(faceIdxGroup.size());
				}

			// register the cell's tetrahedrons' starting index
			latticeTetrIdxMap.insert({lattice, tetrIdcs.size()});
			// register rest shape for the master corner of each tetrahedron
			for (unsigned i = 0; i < N_TETR; i++)
			{
				for (unsigned j = 0; j < 4; j++)
				{
					// distribute mass across vertex references
					unsigned oldIdx = tetrIdxGroups[i][j];
					unsigned nShared = std::round(particleMass / masses[oldIdx]);
					masses[oldIdx] = particleMass / (nShared + 1);
				}

				tetrIdcs.push_back({{cornerIdx[tetrIdxGroups[i][0]], cornerIdx[tetrIdxGroups[i][1]],
									 cornerIdx[tetrIdxGroups[i][2]], cornerIdx[tetrIdxGroups[i][3]],
									 i}});
			}
		}
	}

	void linkTetrahedron(Vec3 &pos, unsigned &tetrIdx, Vec3 &weights)
	{
		Vec3 lattice = snapToGrid(pos);

		// locate cell
		unsigned cellTetrStart = -1;
		auto foundcellTetrStart = latticeTetrIdxMap.find(lattice);
		if (foundcellTetrStart != latticeTetrIdxMap.end())
			cellTetrStart = foundcellTetrStart->second;
		else
			std::logic_error("Unknown lattice");

		// check for each tetrahedron if the coordinate is inside
		tetrIdx = -1;
		for (tetrIdx = cellTetrStart; tetrIdx < cellTetrStart + N_TETR; tetrIdx++)
		{
			Vec3 &x0 = particleX[tetrIdcs[tetrIdx][0]];
			Mat3 &DmInv = tetrDmInvs[tetrIdcs[tetrIdx][TETR_KIND_IDX]];
			// local coordinate in tetrahedron
			weights = DmInv * (pos - x0);

			if (weights.minCoeff() >= 0 && weights.sum() <= 1)
				break;
		}
	}

	void initMesh(ParsedMesh &objMesh)
	{
<<<<<<< Updated upstream
		const int nParticles = particleX.size();
		std::vector<Vec3> force(nParticles, Vec3(0, -0.1, 0));

		addForce(force);
		for (int p = 0; p < nParticles; p++)
			particleX[p] += dt * particleV[p];
		for (int p = 0; p < nParticles; p++)
			particleV[p] += (dt / m_particleMass) * force[p];

		for (int p = 0; p < nParticles; p++)
			if (particleX[p][1] < -1)
			{
				particleX[p][1] = -1;
				particleV[p][1] = 0;
				// treadmill
			}
=======
		// load vertices
		for (auto &vert : objMesh.verts)
		{
			// add the vertex to mesh
			verts.push_back(vert);

			// find container info
			unsigned tetrIdx = -1;
			Vec3 weights;
			linkTetrahedron(Vec3(vert.data()), tetrIdx, weights);
			containerIdcs.push_back(tetrIdx);
			latticeWeights.push_back(weights);
		}

		// load faces
		for (auto &&face : objMesh.faces)
		{
			for (auto &&vert : face)
				fvIdcs.push_back(vert + meshStart);
			fvCnts.push_back(face.size());
		}
>>>>>>> Stashed changes
	}

	void Lattice<N_TETR, N_VERT>::simulateStep(const float dt, const int nSubsteps, const float frame)
	{
		for (int i = 0; i < nSubsteps; i++)
			solver.update(dt / nSubsteps);
		solver.getRslt(particleX);
		/*for (int i = 0; i < nSubsteps; i++)
		 	simulateSubstep(dt / nSubsteps);*/
		for (size_t i = 0; i < meshStart; i++)
			verts[i].Set(particleX[i].data());
		// for (int i = 0; i < verts.size(); i++)
		// 	verts[i].Set(particleX[i].data());
		animesh.commitFrame(frame);
	}
};

class CubicLattice : public Lattice<6, 8>
{
	using Lattice::Vec3;

private:
	float gridsz;

public:
	static std::array<Vec3, 8> corners;
	static std::vector<std::vector<unsigned>> faceIdxGroups;
	static std::array<std::array<unsigned, 4>, 6> tetrIdxGroups;

	CubicLattice(Animesh &animesh, const std::vector<Vec3> &lattices, float gridsz = 1,
				 BackwardEuler::ConstraintFunc constrFunc = NULL,
				 const float lambda = 10.0,
				 const float mu = 10.0,
				 const float rayleighCoefficient = 1e-7,
				 const float mass = 100,
				 bool showGrid = true)
		: Lattice<6, 8>(animesh), gridsz(gridsz)
	{
		initParticles(lattices, corners, gridsz, mass, faceIdxGroups, tetrIdxGroups, showGrid);
		solver.init(particleX, tetrIdcs, masses, tetrDmInvs, tetrRestVs, mu, lambda, rayleighCoefficient, constrFunc);
		meshStart = particleX.size();
		animesh.commitTop();
		animesh.commitFrame(0);
	}

	Vec3 snapToGrid(Vec3 &pos)
	{
		Vec3 signedOnes(pos[0] < 0 ? -1 : 1, pos[1] < 0 ? -1 : 1, pos[2] < 0 ? -1 : 1);

		// each grid point contains the cube centered at that point
		return ((pos + 0.5 * gridsz * signedOnes) / gridsz).cast<int>().cast<float>() * gridsz;
	}

	CubicLattice(Animesh &animesh, ParsedMesh &objMesh, float gridsz = 1,
				 BackwardEuler::ConstraintFunc constrFunc = NULL,
				 const float lambda = 10.0,
				 const float mu = 10.0,
				 const float rayleighCoefficient = 1e-7,
				 const float mass = 100,
				 bool showGrid = true)
		: Lattice<6, 8>(animesh, showGrid), gridsz(gridsz)
	{
		Vec3 min = snapToGrid(Vec3(objMesh.bounds.GetMin().data()));
		// prevent rounding error
		Vec3 max = snapToGrid(Vec3(objMesh.bounds.GetMax().data())) + 0.5 * gridsz * Vec3::Ones();

		std::vector<Vec3> gridPoints;
		for (float x = min[0]; x < max[0]; x += gridsz)
			for (float y = min[1]; y < max[1]; y += gridsz)
				for (float z = min[2]; z < max[2]; z += gridsz)
				{
					gridPoints.emplace_back(x, y, z);
				}

		initParticles(gridPoints, corners, gridsz, mass, faceIdxGroups, tetrIdxGroups, showGrid);
		solver.init(particleX, tetrIdcs, masses, tetrDmInvs, tetrRestVs, mu, lambda, rayleighCoefficient, constrFunc);
		meshStart = particleX.size();
		// initMesh(objMesh);
		animesh.commitTop();
		animesh.commitFrame(0);
	}
};
