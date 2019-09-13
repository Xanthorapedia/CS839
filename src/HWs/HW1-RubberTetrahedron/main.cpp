#include "pxr/pxr.h"

#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/base/vt/array.h"

#include "pxr/base/gf/range3f.h"

#include "physics.hpp"

#include <iostream>

PXR_NAMESPACE_USING_DIRECTIVE

int main(int argc, char *argv[])
{
    float stiff = 50;
    float damping = 2;

    // Create a UsdStage with that root layer.
    UsdStageRefPtr stage = UsdStage::Open(SdfLayer::CreateNew("rubberTetrahedron.usda"));

    // Copy objVerts into VtVec3fArray for Usd.

    Blob v0(1, GfVec3f(0, 2, 0)), v1(1, GfVec3f(0.6, 2, 0)), v2(1, GfVec3f(0, 2, 1)), v3(1, GfVec3f(0, 1, 0));
    std::array<Blob, 4> blobArr{v0, v1, v2, v3};
    Tetroid tet(blobArr, stiff, damping);

    VtVec3fArray vertices;
    VtIntArray faceVertexCounts, faceVertexIndices;

    tet.addToMesh(vertices, faceVertexCounts, faceVertexIndices);

    // Usd currently requires an extent, somewhat unfortunately.
    const int nFrames = 400;
    GfRange3f extent;

    // Create a mesh for this surface
    UsdGeomMesh mesh = UsdGeomMesh::Define(stage, SdfPath("/Tet0"));

    // Set up the timecode
    stage->SetStartTimeCode(0.);
    stage->SetEndTimeCode((double)nFrames);

    // Populate the mesh vertex data
    UsdAttribute pointsAttribute = mesh.GetPointsAttr();
    pointsAttribute.SetVariability(SdfVariabilityVarying);

    for (int frame = 1; frame <= nFrames; frame++)
    {
        for (auto &b : blobArr)
        {
            // gravity
            b.applyForce(GfVec3f(0, -1, 0));

            // ground
            if (b.x.data()[1] < 0)
            {
                b.x.data()[1] = 0;
                b.v.data()[1] = 0;
                b.a.data()[1] = 0;
            }
        }

        tet.update(0.02);
        tet.setVertices(vertices);
        pointsAttribute.Set(vertices, frame);
        for (const auto &pt : vertices)
        {
            extent.UnionWith(pt);
        }
    }

    // Now set the attributes.
    mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts);
    mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices);

    // Set extent.
    mesh.GetExtentAttr().Set(VtVec3fArray({extent.GetMin(), extent.GetMax()}));

    if (stage->GetRootLayer()->Save())
    {
        std::cout << "USD saved to: " << stage->GetRootLayer()->GetRealPath() << "!" << std::endl;
    }
    else
    {
        std::cout << "USD not saved!";
    }

    return 0;
}
