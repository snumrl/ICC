cmake_minimum_required(VERSION 2.8.6)
project(ICC)

add_compile_options(-fPIC)
add_compile_options(-std=c++11)
add_compile_options(-Wdeprecated-declarations)
# add_compile_options(-lm)

add_definitions(-DDPHY_DIR="${CMAKE_HOME_DIRECTORY}")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake_module")

find_package(DART REQUIRED COMPONENTS gui optimizer-ipopt collision-bullet collision-ode CONFIG)
find_package(Boost 1.66.0 COMPONENTS filesystem python3 numpy3)
find_package(PythonLibs 3.5 REQUIRED)
find_package(PythonInterp 3.5 REQUIRED)
find_package(TinyXML REQUIRED)

add_subdirectory( environment )
