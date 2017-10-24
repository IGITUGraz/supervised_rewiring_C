#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "load_mnist.h"


int main() {
    uint8_t *train_images = malloc(TRAIN_IMAGEF_BUFF*sizeof(uint8_t));
    uint8_t *train_labels = malloc(TRAIN_LABELF_BUFF*sizeof(uint8_t));
    uint8_t *test_images = malloc(TEST_IMAGEF_BUFF*sizeof(uint8_t));
    uint8_t *test_labels = malloc(TEST_LABELF_BUFF*sizeof(uint8_t));
    read_bit_file(train_images, TRAIN_IMAGEF_BUFF, MNIST_TRAIN_IMG_FILE);
    read_bit_file(train_labels, TRAIN_LABELF_BUFF, MNIST_TRAIN_LAB_FILE);
    read_bit_file(test_images, TEST_IMAGEF_BUFF, MNIST_TEST_IMG_FILE);
    read_bit_file(test_labels, TEST_LABELF_BUFF, MNIST_TEST_LAB_FILE);

    //check the correctness of the data set
    uint32_t magic_number = 0;
    for (int i=0; i<4; ++i) 
        magic_number |= train_labels[i] << (24-(8*i));
	if (magic_number != 2049) {
        printf("MNIST train_labels magic number %d != 2049. Abort...", magic_number);
        return 1;
    }

    magic_number = 0;
    for (int i=0; i<4; ++i) 
        magic_number |= train_images[i] << (24-(8*i));
	if (magic_number != 2051) {
        printf("MNIST train_images magic number %d != 2051. Abort...", magic_number);
        return 1;
    }

    magic_number = 0;
    for (int i=0; i<4; ++i)
        magic_number |= test_labels[i] << (24-(8*i));
	if (magic_number != 2049) {
        printf("MNIST test_labels magic number %d != 2049. Abort...", magic_number);
        return 1;
    }

    magic_number = 0;
    for (int i=0; i<4; ++i)
        magic_number |= test_images[i] << (24-(8*i));
	if (magic_number != 2051) {
        printf("MNIST test_images magic number %d != 2051. Abort...", magic_number);
        return 1;
    }
	// -------------------------------------------------------------------------

    float this_image[IMAGE_SIZE];
    float this_label[NUM_CLASS];
    uint16_t image_num  = 23;    //from [0,59999]

    get_next_image(this_image, this_label, image_num, train_images, train_labels);

    //display image
    for (int i = 0;i<IMAGE_SIZE;i++) {
        printf("%.0f ", this_image[i]);
        if ((i+1)%IMAGE_COLS == 0)
            printf("\n");
    }
    //display label
    printf("The %d th image's label is [ ", image_num);
    for (int i= 0; i < NUM_CLASS; i++)
        printf("%.1f ", this_label[i]);
    printf("]\n");







    return 0;
}
