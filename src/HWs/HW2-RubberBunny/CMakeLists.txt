create_simple_config(NAME
                     rubberBunny
                     SRCS
                     main.cpp)

# FEM library
add_library(LatticeFEM STATIC LatticeFEM.cpp)
target_include_directories(LatticeFEM
                           PUBLIC . ../Commons/include ${USD_COMMON_INCLUDES})

# compile kernels
find_package(CUDA REQUIRED)
set(
  CUDA_NVCC_FLAGS
  "-Xcompiler \"/wd 4819\" --expt-relaxed-constexpr -arch compute_50 -rdc=false -G -g"
  )

include_directories(${EIGEN3_INCLUDE_DIRS})
set_source_files_properties(cuCG.cc
                            PROPERTIES
                            CUDA_SOURCE_PROPERTY_FORMAT
                            OBJ)
cuda_compile(cg-interface cuCG.cu cuCG.cc)
cuda_add_library(cuCG ${cg-interface} STATIC)

target_link_libraries(cuCG ${CUDA_LIBRARIES} Eigen3::Eigen)

target_link_libraries(LatticeFEM cuCG hw-commons)

target_link_libraries(rubberBunny LatticeFEM)
