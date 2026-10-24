cmake_minimum_required(VERSION 3.16)
project(shg_profile LANGUAGES CXX)

add_executable(shg_profile profile_shg.cpp)
target_compile_features(shg_profile PRIVATE cxx_std_17)
target_compile_options(shg_profile PRIVATE -O3 -march=native)
target_include_directories(shg_profile PRIVATE vendor)

