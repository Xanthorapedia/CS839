#include "Lattice.h"

namespace Lattice
{

static const std::array<Vec3, 8> cubicCorners{
	Vec3(-0.5, -0.5, -0.5), Vec3(+0.5, -0.5, -0.5), Vec3(+0.5, +0.5, -0.5), Vec3(-0.5, +0.5, -0.5),
	Vec3(-0.5, +0.5, +0.5), Vec3(-0.5, -0.5, +0.5), Vec3(+0.5, -0.5, +0.5), Vec3(+0.5, +0.5, +0.5)};

static const std::vector<std::vector<unsigned>> cubicFaceIdxGroups{{{0, 2, 1},
																	{0, 3, 2},
																	{0, 4, 3},
																	{0, 5, 4},
																	{0, 6, 5},
																	{0, 1, 6},
																	{7, 2, 3},
																	{7, 3, 4},
																	{7, 4, 5},
																	{7, 5, 6},
																	{7, 6, 1},
																	{7, 1, 2}}};

static const std::array<std::array<unsigned, 4>, 6> cubicTetrIdxGroups{{{0, 1, 2, 7},
																		{0, 2, 3, 7},
																		{0, 3, 4, 7},
																		{0, 4, 5, 7},
																		{0, 5, 6, 7},
																		{0, 6, 1, 7}}};

class CubicLattice : public Lattice<6, 8>
{

private:
	float gridsz;

public:
	CubicLattice(Animesh &animesh, const std::vector<Vec3> &lattices, FEM::LatticeCfg &lcfg,
				 float gridsz = 1, float density = 1, bool showGrid = true)
		: Lattice<6, 8>(animesh, lattices, cubicCorners, gridsz, density, cubicTetrIdxGroups, cubicFaceIdxGroups, lcfg, showGrid), gridsz(gridsz)
	{
		animesh.commitTop();
	}

	CubicLattice(Animesh &animesh, const ParsedMesh &objMesh, FEM::LatticeCfg &lcfg,
				 float gridsz = 1, float density = 1, bool showGrid = true)
		: Lattice<6, 8>(animesh, showGrid), gridsz(gridsz)
	{
		initParticles(objMesh, cubicCorners, gridsz, density, cubicTetrIdxGroups, cubicFaceIdxGroups, lcfg, showGrid);
		animesh.commitTop();
		std::cout << "Lattice: " << lcfg.x.size() << " vertices, " << lcfg.tetrIdces.size() / 4 << " tetrahedra." << std::endl;
	}

	Vec3 snapToGrid(Vec3 &pos) override
	{
		Vec3 signedOnes(pos[0] < 0 ? -1 : 1, pos[1] < 0 ? -1 : 1, pos[2] < 0 ? -1 : 1);

		// each grid point contains the cube centered at that point
		return ((pos + 0.5 * gridsz * signedOnes) / gridsz).cast<int>().cast<float>() * gridsz;
	}

	std::vector<Vec3> determineGrid(const ParsedMesh &objMesh) override
	{
		Vec3 min = snapToGrid(Vec3(objMesh.bounds.GetMin().data()));
		// prevent rounding error
		Vec3 max = snapToGrid(Vec3(objMesh.bounds.GetMax().data())) + 0.5 * gridsz * Vec3::Ones();

		// (sqrt(3) / 2)^2
		float dist2_thresh = gridsz * gridsz * 0.75;

		std::vector<Vec3> gridPoints;
		for (float x = min[0]; x < max[0]; x += gridsz)
			for (float y = min[1]; y < max[1]; y += gridsz)
				for (float z = min[2]; z < max[2]; z += gridsz)
				{
					Vec3 center(x, y, z);
					if (objTriTree.encloses(center) || objTriTree.squaredDistance(center) < dist2_thresh)
						gridPoints.emplace_back(x, y, z);
				}
		return gridPoints;
	}
};

} // namespace Lattice
