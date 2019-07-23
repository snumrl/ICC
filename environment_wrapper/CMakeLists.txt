cmake_minimum_required(VERSION 2.8.6)
# add_subdirectory( boost )
project(environment_wrapper)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)

link_directories(../environment/)
include_directories(../environment/)

file(GLOB srcs "*.h" "*.cpp" "*.hpp")
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${DART_INCLUDE_DIRS})
include_directories(${TinyXML_INCLUDE_DIRS})

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_HOME_DIRECTORY}/policy)


add_library(environment_wrapper SHARED ${srcs})
target_link_libraries(environment_wrapper ${DART_LIBRARIES} ${Boost_LIBRARIES} ${TinyXML_LIBRARIES} ${PYTHON_LIBRARIES} environment)
set_target_properties(environment_wrapper PROPERTIES PREFIX "" )
