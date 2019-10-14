#pragma once

#include "HWIncludes.hpp"

#include "pxr/base/gf/range1f.h"
#include "pxr/base/gf/range3f.h"

namespace CS839 {

class Animesh
{
private:
    // mesh-related
    UsdGeomMesh m_mesh;
    GfRange3f m_extent;
    float lastTime;

    // geometry
    VtIntArray faceVertexCounts, faceVertexIndices;
    VtVec3fArray m_points;

public:
    // getters
    VtIntArray &getFaceVertexCounts() { return faceVertexCounts; }
    VtIntArray &getfaceVertexIndices() { return faceVertexIndices; }
    VtVec3fArray &getVertices() { return m_points; }

    // Refreshes the topology
    void commitTop()
    {
        m_mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts);
        m_mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices);
    }

    // Creates a new frame in time after the last frame
    void commitFrame(const int time)
    {
        std::cout << "Writing frame " << time << " ..." << std::endl;

        // Check that frames have been written in sequence
        if (time <= lastTime)
            throw std::logic_error("Writing frame back in time: " + std::to_string(time) + " after " + std::to_string(time));
        lastTime = time;

        // Check that there are any particles to write at all
        if (m_points.empty())
            throw std::logic_error("Empty array of input vertices");

        // Update extent
        for (const auto &pt : m_points)
            m_extent.UnionWith(pt);

        // Write the points attribute for the given frame
        m_mesh.GetPointsAttr().Set(m_points, (double)time);
    }

    Animesh(UsdGeomMesh &geomMesh) : lastTime(-1), m_mesh(geomMesh)
    {
        m_mesh.GetPointsAttr().SetVariability(SdfVariabilityVarying);
    }

    // Set extent and get tim eduration
    GfRange1f finalizeSpaceTime()
    {
        // Set the effective extent
        m_mesh.GetExtentAttr().Set(VtVec3fArray({m_extent.GetMin(), m_extent.GetMax()}));

        return GfRange1f({0, lastTime});
    }
};

}
