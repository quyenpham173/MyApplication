
cmake_minimum_required(VERSION 3.4.1)

project(dewarp)

set(CMAKE_VERBOSE_MAKEFILE on)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nlopt-2.6.1)

add_library(lib_opencv SHARED IMPORTED)
add_library(lib_nlopt SHARED IMPORTED)

set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libopencv_java3.so)
set_target_properties(lib_nlopt PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libnlopt.so)

add_library(
        preprocess

        SHARED

        ${CMAKE_CURRENT_SOURCE_DIR}/page_dewarp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/C_preprocess.cpp
        )

target_link_libraries(
        preprocess
        lib_opencv
        lib_nlopt

        android
        log)