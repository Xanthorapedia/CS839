#include "LatticeFEM.hpp"

using Vec3 = Eigen::Vector3f;

int main(int argc, char *argv[])
{
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

    for (int i = 1; i < 400; i++)
    {
        /* code */
        sphere.simulateStep(0.1, 10, i);
    }

    control.saveAnimation();

    return 0;
}
