cmake_minimum_required(VERSION 2.8.6)
project(ICC)

SET(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
#	FORCE
	)

add_compile_options(-fPIC)
add_compile_options(-std=c++11)
add_compile_options(-Wdeprecated-declarations)
# add_compile_options(-lm)

add_definitions(-DICC_DIR="${CMAKE_HOME_DIRECTORY}")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake_modules")

find_package(DART REQUIRED COMPONENTS gui optimizer-ipopt collision-bullet collision-ode CONFIG)
find_package(Boost 1.66.0 COMPONENTS filesystem python3 numpy3)
find_package(PythonLibs 3.5 REQUIRED)
find_package(PythonInterp 3.5 REQUIRED)
find_package(TinyXML REQUIRED)

add_subdirectory(environment)
add_subdirectory(environment_wrapper)
add_subdirectory(renderer)
add_subdirectory(interactive)
