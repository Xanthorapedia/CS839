
#include "SoftSim.h"

using CS839::Animesh;
using CS839::ParsedMesh;
using CS839::SceneControl;
using Vec3 = Eigen::Vector3f;

int main(int argc, char *argv[])
{
    // float scale = 1;
    // float scale = 0.75;
    // float scale = 0.50;
    float scale = 1;
    int nSteps = 10;
    int nSub = 50;
    float dt = 0.025;
    if (argc > 1)
        scale = atof(argv[1]);
    if (argc > 2)
        nSteps = atoi(argv[2]);
    if (argc > 3)
        nSub = atoi(argv[3]);
    if (argc > 4)
        dt = atof(argv[4]);
    std::cout << "Sim settings:" << std::endl;
    std::cout << "scale: " << scale << std::endl;;
    std::cout << "nSteps: " << nSteps << std::endl;;
    std::cout << "nSub: " << nSub << std::endl;;
    std::cout << "dt: " << dt << std::endl;;
    std::cout << std::endl;

    std::vector<Vec3> verts;

    ParsedMesh meshobj("assets/bunny.obj");

    SceneControl control("rubberBunny");

    control.getMesh("grid");
    control.getMesh("bunny");

    Animesh &grid = control.getMesh(0);
    Animesh &bunny = control.getMesh(1);

    SoftSim sim(grid, scale, meshobj, bunny, true, 1 , 10, 20, 0.2, 10);

    for (int i = 1; i < nSteps; i++)
        sim.simulateStep(dt, nSub, i);

    control.saveAnimation();

    return 0;
}