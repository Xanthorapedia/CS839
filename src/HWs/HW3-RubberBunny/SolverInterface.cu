#include "FEM/FEM.h"
#include "FEM/Solver.h"
#include "FEM/SVD.h"
#include "SolverInterface.h"

using Vec3 = Eigen::Vector3f;
using BESolver = FEM::Solver::BackEulerSolver;

void *solverCreate(FEM::LatticeCfg &lcfg,
    const float mu, const float lambda, const float gamma, const float sThresh,
    ConstraintForceFunc &constraint_force)
{
    BESolver *solver = new BESolver(lcfg, mu, lambda, gamma, sThresh, constraint_force);
    return (void *) solver;
}

void solverUpdate(void *solverHandle, float dt)
{
    ((BESolver *) solverHandle)->update(dt);
}

void solverGetResults(void *solverHandle, std::vector<Vec3> &latticeXBuffer)
{
    ((BESolver *) solverHandle)->getRslt(latticeXBuffer);
}

void solverDestroy(void *solver)
{
    delete (BESolver *) solver;
    FEM::SVD::destroyCUSolverHandle();
}
