#pragma once

#include "Lattice/CubicLattice.h"
#include "SolverInterface.h"

using CS839::Animesh;
using CS839::ParsedMesh;

template <class T>
struct SimCF : public ConstraintForceFunc
{
	T &obj;

	SimCF(T &obj) : obj(obj) {}

	void operator()(std::vector<Vec3> &x, std::vector<Vec3> &v, float t,
					std::vector<Vec3> &f, std::vector<unsigned> &fidx,
					std::vector<unsigned> &cidx) override
	{
		obj.getConstraintsAndForces(x, v, t, f, fidx, cidx, hasConstraint, hasForce);
	}

	// asks obj if it wants to skip the frame at time t
	bool skipFrame(float t) override { return obj.skipFrame(t); };
};

FEM::LatticeCfg lcfg;

class SoftSim
{
	Animesh &mesh, &objMesh;
	Lattice::CubicLattice latticeMesh;
	std::vector<Vec3> latticeXBuffer;
	SimCF<SoftSim> cfFunctor;
	void *solver;

public:
	// SoftSim(Animesh &mesh, float scale, std::vector<Vec3> verts, bool showGrid = true, float density = 1,
	// 		float mu = 5, float lambda = 20, float gamma = 0.1, float sThresh = 100)
	// 	: mesh(mesh), latticeMesh(mesh, verts, lcfg, scale, density, showGrid), cfFunctor(*this)
	// {
	// 	// create embedding grid
	// 	solver = solverCreate(lcfg, mu, lambda, gamma, sThresh, cfFunctor);

	// 	mesh.commitFrame(0);
	// }

	SoftSim(Animesh &mesh, float scale, ParsedMesh &pmesh, Animesh &objMesh, bool showGrid = true, float density = 1,
			float mu = 5, float lambda = 20, float gamma = 0.1, float sThresh = 100)
		: mesh(mesh), objMesh(objMesh), latticeMesh(mesh, pmesh, lcfg, scale, density, showGrid), cfFunctor(*this)
	{
		// create embedding grid
		solver = solverCreate(lcfg, mu, lambda, gamma, sThresh, cfFunctor);

		VtVec3fArray &verts = objMesh.getVertices();
		// load vertices
		for (auto &vert : pmesh.verts)
		{
			// add the vertex to mesh
			verts.push_back(vert);
		}

		// load faces
		VtIntArray &fvCnts = objMesh.getFaceVertexCounts();
		VtIntArray &fvIdcs = objMesh.getfaceVertexIndices();
		for (auto &&face : pmesh.faces)
		{
			for (auto &&vert : face)
				fvIdcs.push_back(vert);
			fvCnts.push_back(face.size());
		}
		objMesh.commitTop();

		mesh.commitFrame(0);
		objMesh.commitFrame(0);
	}

	void simulateStep(const float dt, const int nSubsteps, const float frame)
	{
		for (int i = 0; i < nSubsteps; i++)
			solverUpdate(solver, dt / nSubsteps);

		solverGetResults(solver, latticeXBuffer);

		latticeMesh.savePoints(latticeXBuffer, lcfg, objMesh.getVertices());

		mesh.commitFrame(frame);
		objMesh.commitFrame(frame);
	}

	void getConstraintsAndForces(std::vector<Vec3> &x, std::vector<Vec3> &v, float t,
								 std::vector<Vec3> &f, std::vector<unsigned> &fidx,
								 std::vector<unsigned> &cidx, bool &updateC, bool &updateF)
	{
		float ground = -0.25;
		for (unsigned i = 0; i < x.size(); i++)
		{
			float m = lcfg.m[i];

			fidx.push_back(i);
			f.emplace_back(0, -0.98 * m, 0);

			if (x[i].y() < ground)
			{
				f[i].y() += (ground - x[i].y()) * 100;
			}
		}

		updateF = true;
	}

	bool skipFrame(float t)
	{
		return false;
	}

	~SoftSim()
	{
		solverDestroy(solver);
	}
};
