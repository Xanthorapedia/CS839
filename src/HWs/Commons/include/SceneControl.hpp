#pragma once

#include "HWIncludes.hpp"

#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/base/gf/range1f.h"

namespace CS839 {

class SceneControl
{
private:
	// IO stuff
	UsdStageRefPtr m_stage;

	// All known animated meshes
	std::vector<Animesh> meshes;

public:
	SceneControl(const std::string &fileName)
	{
		// Create a UsdStage with that root layer.
		m_stage = UsdStage::Open(SdfLayer::CreateNew(fileName + ".usda"));
	}

	// Creates a new mesh with the given name under "/"
	Animesh &getMesh(std::string name)
	{
		meshes.emplace_back(UsdGeomMesh::Define(m_stage, SdfPath("/" + name)));
		return meshes.back();
	}

	// retrieves an existent mesh
	Animesh &getMesh(int idx)
	{
		return meshes[idx];
	}

	// Save the animation to the USD file
	bool saveAnimation()
	{
		// Get max time range
		GfRange1f duration;
		for (auto &mesh : meshes)
			duration.UnionWith(mesh.finalizeSpaceTime());

		// Set up the timecode
		m_stage->SetStartTimeCode(duration.GetMin());
		m_stage->SetEndTimeCode(duration.GetMax());

		// Save USD file
		if (m_stage->GetRootLayer()->Save())
		{
			std::cout << "USD saved to: " << m_stage->GetRootLayer()->GetRealPath() << "!" << std::endl;
			return true;
		}
		else
		{
			std::cout << "USD not saved!";
			return false;
		}
	}
};

}
