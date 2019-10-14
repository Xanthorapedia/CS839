#include "HWIncludes.hpp"
#include "MeshLoader.hpp"

#include <fstream>

namespace CS839
{

void readObjFile(std::string &path, std::vector<ParsedMesh> &meshes)
{
	std::ifstream infile(path);
	std::string line;
	ParsedMesh mesh;

	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		std::string header;
		iss >> header;

		if (header.compare("v") == 0)
		{
			float x, y, z;
			iss >> x >> y >> z;
			GfVec3f newVert(x, y, z);
			mesh.verts.push_back(newVert);
			mesh.bounds.UnionWith(newVert);
		}
		else if (header.compare("vn") == 0)
		{
			float x, y, z;
			iss >> x >> y >> z;
			mesh.normal.emplace_back(x, y, z);
		}
		else if (header.compare("f") == 0)
		{
			std::string v1, v2, v3;
			iss >> v1 >> v2 >> v3;

			int v1i, v2i, v3i;

			std::istringstream(v1) >> v1i;
			std::istringstream(v2) >> v2i;
			std::istringstream(v3) >> v3i;

			// obj index starts at 1
			mesh.faces.push_back({v1i - 1, v2i - 1, v3i - 1});
		}
		else if (header.compare("o") == 0)
		{
			if (mesh.verts.size() > 0)
			{
				meshes.push_back(mesh);
				mesh = ParsedMesh();
			}

			iss >> mesh.name;
		}
	}
	if (mesh.verts.size() > 0)
		meshes.push_back(mesh);
}

ParsedMesh::ParsedMesh(std::string path)
{
	std::vector<ParsedMesh> meshes;
	readObjFile(CS839_ROOT_DIR + path, meshes);
	*this = meshes[0];
};

} // namespace CS839
