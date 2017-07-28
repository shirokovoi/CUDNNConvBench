cmake_minimum_required(VERSION 2.8)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake/Modules)

project(CUDNNConvBench CXX)
find_package(CUDA REQUIRED)
find_package(CUDNN REQUIRED)
set(CUDA_NVCC_FLAGS -std=c++11 -ccbin g++-5)
set_source_files_properties(Sources/conv_bench.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
cuda_add_executable(CUDNNConvBench Sources/conv_bench.cpp Sources/Types.cpp)
target_include_directories(CUDNNConvBench PRIVATE Sources/)
target_link_libraries(CUDNNConvBench CUDNN ${CUDA_curand_LIBRARY})
