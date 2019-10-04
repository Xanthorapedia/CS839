create_simple_config(NAME
                     rubberBunny
                     SRCS
                     LatticeFEM.cpp
                     main.cpp)
target_include_directories(rubberBunny
                           INTERFACE . ../Commons/include)
target_link_libraries(rubberBunny hw-commons)
