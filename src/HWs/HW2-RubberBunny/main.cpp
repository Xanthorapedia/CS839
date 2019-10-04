#include "LatticeFEM.hpp"

using Vec3 = Eigen::Vector3f;

int main(int argc, char *argv[])
{
    SceneControl control("rubberCube");

    std::vector<Vec3> points;
    for (int i = 0; i < 10; i++)
    for (int j = 0; j < 10; j++)
    for (int k = 0; k < 10; k++)
        if((i - 5)* (i - 5) + (j - 5) * (j - 5) + (k - 5) * (k - 5) < 5 * 5)
            points.emplace_back(Vec3(i * 1, j * 1, k * 1));
    CubicLattice bunny(control.getMesh("sphere"), points);
    // CubicLattice bunny(control.getMesh("cube"), {Vec3(0, 1, 0), Vec3(1, 1, 0), Vec3(0, 0, 0), Vec3(0, 1, 1), Vec3(0, 2, 0)});
    // CubicLattice bunny(control.getMesh("cube"), {Vec3(0, 0, 0)});

    for (int i = 1; i < 200; i++)
    {
        /* code */
        bunny.simulateStep(0.1, 10, i);
    }

    control.saveAnimation();

    return 0;
}
