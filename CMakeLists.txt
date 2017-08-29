cmake_minimum_required(VERSION 3.8)
project(supervised_rewiring_C)

set(CMAKE_C_STANDARD 99)

# test structure
set(MATH_TEST_SOURCE_FILES main_math_test.c math_operations.c test_math.c test_network_gradient.c)
add_executable(math_in_C ${MATH_TEST_SOURCE_FILES})
target_link_libraries(math_in_C m)

# main structure
set(SOURCE_FILES main_sparseNN_x86.c math_operations.c test_math.c test_network_gradient.c)
add_executable(sparseNN_x86 ${SOURCE_FILES})
target_link_libraries(sparseNN_x86 m)