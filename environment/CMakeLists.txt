cmake_minimum_required(VERSION 2.8.6)
project(environment)


set(LIBRARY_OUTPUT_PATH ${CMAKE_HOME_DIRECTORY}/lib)

add_compile_options(-DHAVE_CSTDDEF)
include_directories(${DART_INCLUDE_DIRS})
include_directories(${TinyXML_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB srcs "*.h" "*.cpp")

add_library(environment ${srcs})
target_link_libraries(environment 
	${DART_LIBRARIES} 
	${Boost_LIBRARIES} 
	${TinyXML_LIBRARIES} 
	${PYTHON_LIBRARIES}
)
