cmake_minimum_required(VERSION 2.8.6)
project(renderer)



include_directories(${CMAKE_HOME_DIRECTORY}/environment/)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

add_compile_options(-DHAVE_CSTDDEF)
include_directories(${DART_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB srcs "*.h" "*.cpp")

add_executable(renderer ${srcs})
target_link_libraries(renderer
	${DART_LIBRARIES} 
	${Boost_LIBRARIES} 
	${PYTHON_LIBRARIES}
	GL GLU glut GLEW environment
)
