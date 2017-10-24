#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define MNIST_TRAIN_IMG_FILE "./MNIST/train-images-idx3-ubyte"
#define MNIST_TRAIN_LAB_FILE "./MNIST/train-labels-idx1-ubyte"
#define MNIST_TEST_IMG_FILE  "./MNIST/t10k-images-idx3-ubyte"
#define MNIST_TEST_LAB_FILE  "./MNIST/t10k-labels-idx1-ubyte"
#define NUM_TRAIN               60000
#define TRAIN_IMAGEF_BUFF    47040016
#define TRAIN_LABELF_BUFF       60008
#define NUM_TEST                10000
#define TEST_IMAGEF_BUFF      7840016
#define TEST_LABELF_BUFF        10008
#define IMAGE_ROWS                 28
#define IMAGE_COLS                 28
#define IMAGE_SIZE                784
#define NUM_CLASS                  10
#define TRAIN_IMAGE_OFFSET         16
#define TRAIN_LABEL_OFFSET         8

void read_bit_file(uint8_t *buff, size_t buff_len, char *file_path);

void get_next_image(float *this_image, float *this_label, uint16_t image_num, uint8_t *images_set, uint8_t *labels_set);


