#pragma once

#include "HWIncludes.hpp"

#include "pxr/base/gf/range1f.h"
#include "pxr/base/gf/range3f.h"

#include <string>

namespace CS839
{

struct ParsedMesh
{
    std::string name;
    std::vector<GfVec3f> verts, normal;
    std::vector<VtIntArray> faces;
    GfRange3f bounds;

public:
    ParsedMesh() {};

    ParsedMesh(std::string path);
};

void readObjFile(std::string &path, std::vector<ParsedMesh> &meshes);

} // namespace CS839
