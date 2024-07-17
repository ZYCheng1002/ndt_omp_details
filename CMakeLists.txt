cmake_minimum_required(VERSION 3.10)
project(ndt_omp)

# pcl 1.7 causes a segfault when it is built with debug mode
set(CMAKE_BUILD_TYPE "RELEASE")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -ggdb ${CMAKE_CXX_FLAGS}")

if (BUILD_WITH_MARCH_NATIVE)
  add_compile_options(-march=native)  # 使用最新微架构进行优化
else()
  add_definitions(-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)   # 提高并行计算能力
  set(CMAKE_CXX_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include(cmake/packages.cmake)
include_directories(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(src)


