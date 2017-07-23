cmake_minimum_required(VERSION 2.8)

project(CUDNNConvBench CXX)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -std=c++11 -ccbin g++-5)
CUDA_ADD_EXECUTABLE(CUDNNConvBench Sources/conv_bench.cpp Sources/Types.cpp)
target_include_directories(CUDNNConvBench PRIVATE /home/oleg/Prefixes/cudnn/include)
target_include_directories(CUDNNConvBench PRIVATE Sources/)
target_link_libraries(CUDNNConvBench /home/oleg/Prefixes/cudnn/lib64/libcudnn.so)
