cmake_minimum_required(VERSION 2.6)
set(PROJECT_NAME Sophus)
project(${PROJECT_NAME})

set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
   set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
   set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ")
   add_definitions("-Wall -Werror -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas ")
endif()


find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(CLASSES so2
   se2
   se3
   so3
   scso3
   sim3
)

set(SOURCES)
set(SOURCE_DIR "sophus")
foreach(class ${CLASSES})
   list(APPEND SOURCES ${SOURCE_DIR}/${class}.cpp ${SOURCE_DIR}/${class}.h)
endforeach(class)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_include_directories(Sophus PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})