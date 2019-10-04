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
const std::array<CubicLattice::Vec3, 8> CubicLattice::corners{
    Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0),
    Vec3(0, 1, 1), Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(1, 1, 1)};
const std::vector<std::vector<unsigned>> CubicLattice::faceIdxGroups{{
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
const std::array<std::array<unsigned, 4>, 6> CubicLattice::tetrIdxGroups{{
    {0, 1, 2, 7},
    {0, 2, 3, 7},
    {0, 3, 4, 7},
    {0, 4, 5, 7},
    {0, 5, 6, 7},
    {0, 6, 1, 7},
}};
