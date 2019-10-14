#include "LatticeFEM.hpp"

using Vec3 = Eigen::Vector3f;
using SceneControl = CS839::SceneControl;
using ParsedMesh = CS839::ParsedMesh;

std::vector<unsigned> constraint(cuCG::SimCfg &scfg)
{
    std::vector<unsigned> fix;
    // drag
    if (scfg.dt * 20 > scfg.t)
    {
        for (size_t i = 0; i < scfg.N_PART; i++)
            if (scfg.x[i].y() > 0)
            {
                scfg.x[i].x() *= 1.1;
                // fix.push_back(i);
            }
            else if (scfg.x[i].y() < 0)
            {
                scfg.x[i].x() /= 1.1;
                // fix.push_back(i);
            }
    }

    return fix;
}

int main(int argc, char *argv[])
{
<<<<<<< Updated upstream
    SceneControl control("rubberSphere");

    float scale = 1;
    float size = 13;
    std::vector<Vec3> points;
    for (float i = 0; i < size; i += scale)
        for (float j = 0; j < size; j += scale)
            for (float k = 0; k < size; k += scale)
                if ((i - size / 2) * (i - size / 2) +
                        (j - size / 2) * (j - size / 2) +
                        (k - size / 2) * (k - size / 2) <
                    size / 2 * size / 2)
                    points.emplace_back(Vec3(i * scale, j * scale, k * scale));
    CubicLattice sphere(control.getMesh("sphere"), points);
=======
    SceneControl control("rubberCube");
    CubicLattice cube(control.getMesh("cube"), {Vec3(0, 0, 0)}, 1, constraint, 1, 1, 0.1, 10);
>>>>>>> Stashed changes

    for (int i = 1; i < 400; i++)
    {
        /* code */
<<<<<<< Updated upstream
        sphere.simulateStep(0.1, 10, i);
=======
        cube.simulateStep(1, 1, i);
>>>>>>> Stashed changes
    }

    control.saveAnimation();

    return 0;
}
