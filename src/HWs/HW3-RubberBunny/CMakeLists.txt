create_simple_config(NAME
                     rubberBunny
                     SRCS
                     main.cc)

set(EIGEN3_INCLUDE_DIRS "C:/Program Files/Eigen3")

# Evaluation builds
# set(DEVICE_FLAGS "/DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
# set(DEVICE_FLAGS "/DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP /openmp")
# set(DEVICE_FLAGS "/DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
set(DEVICE_FLAGS "/DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA")

# FEM library add_library(LatticeFEM STATIC LatticeFEM.cpp)
include_directories(PUBLIC
                    .
                    ../Commons/include
                    ${USD_COMMON_INCLUDES}
                    ${EIGEN3_INCLUDE_DIRS})

# compile kernels
find_package(CUDA REQUIRED)
set(
  CUDA_NVCC_FLAGS
  "-Xcompiler \"/wd 4819 ${DEVICE_FLAGS}\" --expt-relaxed-constexpr -arch compute_50 -rdc=false $<$<CONFIG:Debug>:-G -g> $<$<CONFIG:Release>:-O3>"
  )

message(${CUDA_NVCC_FLAGS})

find_package(CGAL REQUIRED QUIET)

cuda_compile(cuFEM SolverInterface.cu FEM/CG.cu FEM/SVD.cu FEM/Solver.cu)
cuda_add_library(FEM ${cuFEM} STATIC)

# target_link_libraries(Lattice Eigen3::Eigen)
target_link_libraries(FEM ${CUDA_LIBRARIES} Eigen3::Eigen ${CUDA_cusolver_LIBRARY})

target_link_libraries(rubberBunny FEM hw-commons CGAL::CGAL)

