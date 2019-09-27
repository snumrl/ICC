cmake_minimum_required(VERSION 2.8.6)
project(interactive)



find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

add_compile_options(-DHAVE_CSTDDEF)
include_directories(${DART_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_HOME_DIRECTORY}/environment/)
include_directories(${CMAKE_HOME_DIRECTORY}/renderer/)

file(GLOB srcs "*.h" "*.cpp"
	"../renderer/Camera.*" "../renderer/DART_interface.*" "../renderer/GLfunctions.*" "../renderer/GLUTWindow.*")

add_executable(interactive ${srcs})
target_link_libraries(interactive
	${DART_LIBRARIES} 
	${Boost_LIBRARIES} 
	${PYTHON_LIBRARIES}
	GL GLU glut GLEW environment
)
