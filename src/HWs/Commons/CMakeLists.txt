add_library(hw-commons-header INTERFACE)
target_include_directories(hw-commons-header INTERFACE include)

add_library(hw-commons STATIC src/MeshLoader.cpp)
target_include_directories(hw-commons
                           PUBLIC ${USD_COMMON_INCLUDES}
                           INTERFACE include)

target_link_libraries(hw-commons hw-commons-header)
