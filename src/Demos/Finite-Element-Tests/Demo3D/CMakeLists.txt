add_executable (Demo3D
        main.cpp
        )

find_package(PythonLibs)

set(EIGEN3_INC_DIR /usr/include/eigen3)

target_include_directories(Demo3D
        PUBLIC
            ${USD_INC_DIR}
            ${EIGEN3_INC_DIR}
            ${PYTHON_INCLUDE_PATH}
        )

target_link_libraries(
        Demo3D
                ${USD_LIB_DIR}/libgf.so
                ${USD_LIB_DIR}/libsdf.so
                ${USD_LIB_DIR}/libtf.so
                ${USD_LIB_DIR}/libusd.so
                ${USD_LIB_DIR}/libusdGeom.so
                ${USD_LIB_DIR}/libvt.so
                ${USD_LIB_DIR}/libboost_python.so # todo find library
                ${PYTHON_LIBRARY}
        )
