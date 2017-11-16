cmake_minimum_required(VERSION 3.8)
project(supervised_rewiring_C)

set(CMAKE_C_STANDARD 99)

# test main in C
set(MATH_TEST_SOURCE_FILES main_test_math.c test_math.c math_operations.c)
add_executable(test_math ${MATH_TEST_SOURCE_FILES})
target_link_libraries(test_math m)

# test main in C
set(GRADIENT_TEST_SOURCE_FILES main_test_gradient.c test_network_gradient.c math_operations.c)
add_executable(test_gradient ${GRADIENT_TEST_SOURCE_FILES})
target_link_libraries(test_gradient m)

# test train network
set(TRAIN_SOURCE_FILES main_train_network.c math_operations.c load_mnist.c train_network.c )
add_executable(train_network ${TRAIN_SOURCE_FILES})
target_link_libraries(train_network m)

# test sparseNN_X86
#set(SOURCE_FILES main_sparseNN_x86.c math_operations.c test_math.c test_network_gradient.c)
#add_executable(sparseNN_x86 ${SOURCE_FILES})
#target_link_libraries(sparseNN_x86 m)

#test import MNIST dataset
set(MNIST_SOURCE_FILES test_load_mnist.c load_mnist.c)
add_executable(test_load_mnist ${MNIST_SOURCE_FILES})
target_link_libraries(test_load_mnist m)

