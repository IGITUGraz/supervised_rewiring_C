#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define LAYER_CNT 4

#define MNIST_TRAIN_IMG_FILE "./MNIST/train-images-idx3-ubyte"
#define MNIST_TRAIN_LAB_FILE "./MNIST/train-labels-idx1-ubyte"
#define MNIST_TEST_IMG_FILE  "./MNIST/t10k-images-idx3-ubyte"
#define MNIST_TEST_LAB_FILE  "./MNIST/t10k-labels-idx1-ubyte"
#define NUM_TRAIN_EX            60000
#define TRAIN_IMAGEF_BUFF    47040016
#define TRAIN_LABELF_BUFF       60008
#define NUM_TEST_EX             10000
#define TEST_IMAGEF_BUFF      7840016
#define TEST_LABELF_BUFF        10008
#define IMAGE_ROWS                 28
#define IMAGE_COLS                 28

void read_bit_file(uint8_t *buff, size_t buff_len, char *file_path) {
    FILE *file;

    file = fopen(file_path, "rb");
    fread(buff, buff_len, 1, file);
}

int main() {
    uint16_t layers[LAYER_CNT] = {784, 400, 400, 10};
    uint16_t num_features = layers[0];

    // x86 specific input load (tcl loads input, santos will only need pointer)
    uint8_t *images = malloc(TRAIN_IMAGEF_BUFF*sizeof(uint8_t));
    uint8_t *labels = malloc(TRAIN_LABELF_BUFF*sizeof(uint8_t));
    read_bit_file(images, TRAIN_IMAGEF_BUFF, MNIST_TRAIN_IMG_FILE);
    read_bit_file(labels, TRAIN_LABELF_BUFF, MNIST_TRAIN_LAB_FILE);

    uint32_t magic_number = 0;
    for (int i=0; i<4; ++i)
        magic_number |= images[i] << (24-(8*i));

    if (magic_number != 2051) {
        printf("MNIST train images magic number %d != 2051. Abort...", magic_number);
        return 1;
    }
    // -------------------------------------------------------------------------

    return 0;
}
