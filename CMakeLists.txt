cmake_minimum_required(VERSION 3.20)
project(DiscoRL LANGUAGES CXX)

option(DISCORL_BUILD_APPS "Build the training programs" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# LibTorch's CUDA probe needs an architecture; "native" exists since 3.24.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND
   CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

add_library(discorl
  src/agent.cpp
  src/env.cpp
  src/meta_train.cpp
  src/networks.cpp
  src/nn.cpp
  src/params.cpp
  src/replay.cpp
  src/update_rules.cpp
  src/value.cpp
)
target_include_directories(discorl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(discorl PUBLIC ${TORCH_LIBRARIES} PRIVATE ZLIB::ZLIB)
target_compile_options(discorl PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

if(DISCORL_BUILD_APPS)
  add_executable(discorl_train apps/train.cpp)
  target_link_libraries(discorl_train PRIVATE discorl)
  add_executable(discorl_meta_train apps/meta_train.cpp)
  target_link_libraries(discorl_meta_train PRIVATE discorl)
endif()
