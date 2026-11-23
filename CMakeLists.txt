cmake_minimum_required(VERSION 3.24)
project(engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CPU build works everywhere (Mac included). Turn this on only on the PC.
option(ENGINE_CUDA "Build the CUDA backend" OFF)

if(ENGINE_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 86) # RTX 3060 / Ampere
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
endif()

add_executable(engine src/main.cpp src/safetensors.cpp src/ops.cpp src/model.cpp)
target_compile_options(engine PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

find_package(Threads REQUIRED)
target_link_libraries(engine PRIVATE Threads::Threads)

if(ENGINE_CUDA)
  target_sources(engine PRIVATE src/model_cuda.cu)
  target_compile_definitions(engine PRIVATE ENGINE_CUDA)
  target_link_libraries(engine PRIVATE CUDA::cudart)
endif()
