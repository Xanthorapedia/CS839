cmake_minimum_required(VERSION 2.8.11)
set(CMAKE_CXX_STANDARD 11)

project(CS839)

add_compile_definitions(WIN32_LEAN_AND_MEAN)
add_compile_definitions(NOMINMAX)

set(CMAKE_CXX_FLAGS "/wd\"4244\" /wd\"4305\" /wd\"4267\" ${CMAKE_CXX_FLAGS}")

# only accepts python 2.7
set(Python_ADDITIONAL_VERSIONS 2.7)
find_package(PythonLibs REQUIRED)
find_package(OpenGL)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# set(USD_ROOT_DIR "$<$<CONFIG:Debug>:$ENV{USD_HOME_DBG}>$<$<CONFIG:Release>:$EN
# V{USD_HOME_RLS}>")
if(NOT USD_ROOT_DIR)
  set(USD_ROOT_DIR "$ENV{USD_HOME}")
endif()

set(USD_INC_DIR ${USD_ROOT_DIR}/include/)
set(USD_LIB_DIR ${USD_ROOT_DIR}/lib/)
# expends to boost_python27-vc142-mt-gd-x64-1_70.lib in debug mode
set(BOOST_PYTHON_LIB
    ${USD_LIB_DIR}/boost_python27-vc142-mt$<$<CONFIG:Debug>:-gd>-x64-1_70.lib)
set(USD_COMMON_LIBS
    ${USD_LIB_DIR}/gf.lib
    ${USD_LIB_DIR}/sdf.lib
    ${USD_LIB_DIR}/tf.lib
    ${USD_LIB_DIR}/usd.lib
    ${USD_LIB_DIR}/usdGeom.lib
    ${USD_LIB_DIR}/vt.lib
    ${USD_LIB_DIR}/tbb$<$<CONFIG:Debug>:_debug>.lib
    ${BOOST_PYTHON_LIB}
    ${PYTHON_LIBRARIES})
set(USD_COMMON_INCLUDES ${USD_INC_DIR} ${PYTHON_INCLUDE_PATH})

# config utility for each sub-projects
function(create_simple_config)
  cmake_parse_arguments(PARSED_ARGS # prefix of output variables
                        "" # list of names of the boolean arguments (only
                           # defined ones will be true)
                        "NAME" # list of names of mono-valued arguments
                        "SRCS" # list of names of multi-valued arguments (output
                               # variables are lists)
                        ${ARGN} # arguments of the function to parse, here we
                                # take the all original ones
                        )
  message("(${PARSED_ARGS_SRCS}) -> ${PARSED_ARGS_NAME}")

  # config for this target
  add_executable(${PARSED_ARGS_NAME} ${PARSED_ARGS_SRCS})
  target_include_directories(${PARSED_ARGS_NAME} PUBLIC ${USD_COMMON_INCLUDES})
  target_link_libraries(${PARSED_ARGS_NAME} ${USD_COMMON_LIBS} Eigen3::Eigen)

endfunction(create_simple_config)

add_subdirectory(src/Demos)
add_subdirectory(src/HWs)
