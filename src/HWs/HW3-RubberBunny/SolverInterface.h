#pragma once

#include "FEM/FEM.h"

using Vec3 = Eigen::Vector3f;
using FEM::Solver::ConstraintForceFunc;

void *solverCreate(FEM::LatticeCfg &lcfg,
				   const float mu, const float lambda, const float gamma, const float sThresh,
				   ConstraintForceFunc &constraintForceFunc = FEM::Solver::DEFAULT_CF);

void solverUpdate(void *solverHandle, float dt);

void solverGetResults(void *solverHandle, std::vector<Vec3> &latticeXBuffer);

void solverDestroy(void *solver);
