cmake_minimum_required(VERSION 3.8)
project(math_in_C)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES main.c math_operations.c)
add_executable(math_in_C ${SOURCE_FILES})

target_link_libraries(math_in_C m)