#pragma once

#include "Animesh.hpp"
#include "SceneControl.hpp"

#include <Eigen/Dense>
#include <unordered_map>

#define HASH_EPS 1e-4
#define TETR_KIND_IDX 4

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

private:
	// Mesh info
	Animesh &animesh;
	VtIntArray &fvCnts, &fvIdcs;
	VtVec3fArray &verts;

	// Particle info
	std::vector<Vec3> particleX, particleV;
	std::array<Mat3, N_TETR> tetrDmInvs;
	std::array<float, N_TETR> tetrRestVs;
	// { x0, x1, x2, x3, tetr_kind }
	std::vector<std::array<unsigned, 5>> tetrIdcs;
	Vec3Map latticeIdxMap;

	// Physical properties
	float m_mu, m_lambda, m_rayleighCoefficient, m_particleMass;

	inline int Lattice<N_TETR, N_VERT>::_addLatticeParticle(const Vec3 &lattice)
	{
		int idx = verts.size();
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
		// the indices of each face
		const std::vector<std::vector<unsigned>> &faceIdxGroups,
		// the indices of each tetrahedron
		const std::array<std::array<unsigned, 4>, N_TETR> &tetrIdxGroups,
		const float mu = 10.0, const float lambda = 1.0, const float rayleighCoefficient = 0.1, const float mass = 1)
		: animesh(mesh),
		  fvCnts(animesh.getFaceVertexCounts()),
		  fvIdcs(animesh.getfaceVertexIndices()),
		  verts(animesh.getVertices()),
		  m_mu(mu), m_lambda(lambda), m_rayleighCoefficient(rayleighCoefficient), m_particleMass(mass)
	{
		initParticles(lattices, corners, faceIdxGroups, tetrIdxGroups);
	}

	void Lattice<N_TETR, N_VERT>::initParticles(
		// unit centers
		const std::vector<Vec3> &lattices,
		// vertex displacements
		const std::array<Vec3, N_VERT> &corners,
		// the indices of each face
		const std::vector<std::vector<unsigned>> &faceIdxGroups,
		// the indices of each tetrahedron
		const std::array<std::array<unsigned, 4>, N_TETR> &tetrIdxGroups)
	{
		// pre-compute rest shape for each kind of tetrahedron
		for (int i = 0; i < N_TETR; i++)
		{
			auto &&tetrIdxGroup = tetrIdxGroups[i];
			const Vec3 &X0 = corners[tetrIdxGroup[0]];
			// Dm = (X1 - X0 | X1 - X0 | X3 - X0)
			Mat3 Dm;
			for (int i = 1; i < TETR_KIND_IDX; i++)
				Dm.col(i - 1) = corners[tetrIdxGroup[i]] - X0;

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
			for (int i = 0; i < 8; i++)
			{
				Vec3 &&corner = lattice + corners[i];

				// find corner idx
				auto cachedCornerIdx = latticeIdxMap.find(corner);
				if (cachedCornerIdx == latticeIdxMap.end())
				{
					// add missing corner to the list
					cornerIdx[i] = _addLatticeParticle(corner);
					latticeIdxMap.insert({corner, cornerIdx[i]});
				}
				else
					cornerIdx[i] = cachedCornerIdx->second;
			}

			// create faces
			for (auto &&faceIdxGroup : faceIdxGroups)
			{
				for (auto &&faceIdx : faceIdxGroup)
					fvIdcs.push_back(cornerIdx[faceIdx]);
				fvCnts.push_back(faceIdxGroup.size());
			}

			// register rest shape for the master corner of each tetrahedron
			for (unsigned i = 0; i < N_TETR; i++)
				tetrIdcs.push_back({{cornerIdx[tetrIdxGroups[i][0]], cornerIdx[tetrIdxGroups[i][1]],
									 cornerIdx[tetrIdxGroups[i][2]], cornerIdx[tetrIdxGroups[i][3]],
									 i}});
		}
		animesh.commitTop();
		animesh.commitFrame(0);
	}

	void Lattice<N_TETR, N_VERT>::addForce(std::vector<Vec3> &f) const
	{
		for (auto &&element : tetrIdcs)
		{
			const int tetrKind = element[TETR_KIND_IDX];

			// Linear Elasticity
			Mat3 Ds;
			const Vec3 &x0 = particleX[element[0]];
			for (int i = 0; i < 3; i++)
				Ds.col(i) = particleX[element[i + 1]] - x0;
			Mat3 F = Ds * tetrDmInvs[tetrKind];

			Mat3 strain = .5 * (F + F.transpose()) - Mat3::Identity();
			Mat3 P = 2. * m_mu * strain + m_lambda * strain.trace() * Mat3::Identity();

			Mat3 H = -tetrRestVs[tetrKind] * P * tetrDmInvs[tetrKind].transpose();

			for (int i = 0; i < 3; i++)
			{
				f[element[i + 1]] += H.col(i);
				f[element[0]] -= H.col(i);
			}

			// Linear Damping
			Mat3 Ds_dot;
			const Vec3 &v0 = particleV[element[0]];
			for (int i = 0; i < 3; i++)
				Ds_dot.col(i) = particleV[element[i + 1]] - v0;
			Mat3 F_dot = Ds_dot * tetrDmInvs[tetrKind];

			Mat3 strain_rate = .5 * (F_dot + F_dot.transpose());
			Mat3 P_damping = m_rayleighCoefficient * (2. * m_mu * strain_rate + m_lambda * strain_rate.trace() * Mat3::Identity());

			Mat3 H_damping = -tetrRestVs[tetrKind] * P_damping * tetrDmInvs[tetrKind].transpose();

			for (int i = 0; i < 3; i++)
			{
				f[element[i + 1]] += H_damping.col(i);
				f[element[0]] -= H_damping.col(i);
			}
		}
		// Add constraint
	}

	void Lattice<N_TETR, N_VERT>::simulateSubstep(const float dt)
	{
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
	}

	void Lattice<N_TETR, N_VERT>::simulateStep(const float dt, const int nSubsteps, const float frame)
	{
		for (int i = 0; i < nSubsteps; i++)
			simulateSubstep(dt / nSubsteps);
		for (int i = 0; i < verts.size(); i++)
			verts[i].Set(particleX[i].data());
		animesh.commitFrame(frame);
	}
};

class CubicLattice : public Lattice<6, 8>
{
	using Lattice::Vec3;

private:
	static const std::array<Vec3, 8> corners;
	static const std::vector<std::vector<unsigned>> faceIdxGroups;
	static const std::array<std::array<unsigned, 4>, 6> tetrIdxGroups;

public:
	CubicLattice(Animesh &animesh, const std::vector<Vec3> &lattices)
		: Lattice<6, 8>(animesh, lattices, corners, faceIdxGroups, tetrIdxGroups)
	{
	}
};
