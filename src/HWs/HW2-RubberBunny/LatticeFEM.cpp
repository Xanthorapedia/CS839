#include "LatticeFEM.hpp"

//     y
//     |
//     3-------2
//    /|      /|
//   4-+-----7 |
//   | 0-----|-1--x
//   |/      |/
//   5-------6
//  /
// z
std::array<CubicLattice::Vec3, 8> CubicLattice::corners{
    Vec3(-0.5, -0.5, -0.5), Vec3(+0.5, -0.5, -0.5), Vec3(+0.5, +0.5, -0.5), Vec3(-0.5, +0.5, -0.5),
    Vec3(-0.5, +0.5, +0.5), Vec3(-0.5, -0.5, +0.5), Vec3(+0.5, -0.5, +0.5), Vec3(+0.5, +0.5, +0.5)};
std::vector<std::vector<unsigned>> CubicLattice::faceIdxGroups{{
    {0, 2, 1},
    {0, 3, 2},
    {0, 4, 3},
    {0, 5, 4},
    {0, 6, 5},
    {0, 1, 6},
    {7, 2, 3},
    {7, 3, 4},
    {7, 4, 5},
    {7, 5, 6},
    {7, 6, 1},
    {7, 1, 2},
}};
std::array<std::array<unsigned, 4>, 6> CubicLattice::tetrIdxGroups{{
    {0, 1, 2, 7},
    {0, 2, 3, 7},
    {0, 3, 4, 7},
    {0, 4, 5, 7},
    {0, 5, 6, 7},
    {0, 6, 1, 7},
}};
