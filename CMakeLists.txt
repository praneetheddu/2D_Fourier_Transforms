cmake_minimum_required(VERSION 3.9.1)

PROJECT(p3)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Threads REQUIRED) 
FIND_PACKAGE(MPI REQUIRED) 

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
add_definitions(-DOMPI_SKIP_MPICXX)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--std=c++11)
#SET(CMAKE_CXX_FLAGS ${CMAKE_CSS_FLAGS} "--std=c++11 -pthread")
SET(CMAKE_CXX_STANDARD 17)

INCLUDE_DIRECTORIES(
        src
	cpu_src
        SYSTEM$(CUDA_INCLUDE_PATH)
        )

set(SOURCE_FILES
        src/complex_cuda.h
        src/input_image_cuda.h
        src/input_image_cuda.cc
        )

CUDA_ADD_EXECUTABLE(p33 ${SOURCE_FILES} p33.cu)
TARGET_LINK_LIBRARIES(p33 ${CUDA_LIBRARIES})

set(A_NAME
	cpu_src/complex.cc
	cpu_src/input_image.cc
        cpu_src/complex.h
        cpu_src/input_image.h
        )

ADD_EXECUTABLE(p31 p31.cc)
TARGET_LINK_LIBRARIES(p31 ${CMAKE_THREAD_LIBS_INIT})

ADD_EXECUTABLE(p32 ${A_NAME} p32.cpp)
TARGET_LINK_LIBRARIES(p32 PRIVATE MPI::MPI_C)

