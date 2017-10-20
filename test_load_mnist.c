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
#define IMAGE_SIZE                784
#define TRAIN_IMAGE_OFFSET         16
#define TRAIN_LABEL_OFFSET         8

void read_bit_file(uint8_t *buff, size_t buff_len, char *file_path) {
    FILE *file;

    file = fopen(file_path, "rb");
    fread(buff, buff_len, 1, file);
    fclose(file);
}

void get_next_image(float *this_image, float *this_label, uint16_t image_num, uint8_t *train_images, uint8_t *train_labels) {
    //ensure all pointers are valid
    if (NULL == this_image) {
        fprintf(stderr, "this_image points to NULL!\n");
        exit(-1);
    }
    if (NULL == this_label) {
        fprintf(stderr, "this_label points to NULL!\n");
        exit(-1);
    }
    if (NULL == train_images) {
        fprintf(stderr, "train_images points to NULL!\n");
        exit(-1);
    }
    if (NULL == train_labels) {
        fprintf(stderr, "train_labels points to NULL!\n");
        exit(-1);
    }

    float norma_max = 255.0;

    for (uint i = 0; i < IMAGE_SIZE; i++) {
        this_image[i] = *(train_images + TRAIN_IMAGE_OFFSET + image_num * IMAGE_SIZE + i) / norma_max;
    }
    this_label[0] = *(train_labels + TRAIN_LABEL_OFFSET + image_num);
}

int main() {
    uint16_t layers[LAYER_CNT] = {784, 400, 400, 10};
    uint16_t num_features = layers[0];

    // x86 specific input load (tcl loads input, santos will only need pointer)
    uint8_t *train_images = malloc(TRAIN_IMAGEF_BUFF*sizeof(uint8_t));
    uint8_t *train_labels = malloc(TRAIN_LABELF_BUFF*sizeof(uint8_t));
    read_bit_file(train_images, TRAIN_IMAGEF_BUFF, MNIST_TRAIN_IMG_FILE);
    read_bit_file(train_labels, TRAIN_LABELF_BUFF, MNIST_TRAIN_LAB_FILE);

    uint32_t magic_number = 0;
    for (int i=0; i<4; ++i) 
        magic_number |= train_labels[i] << (24-(8*i));
    
	if (magic_number != 2049) {
        printf("MNIST train train_labels magic number %d != 2049. Abort...", magic_number);
        return 1;
    }

    magic_number = 0;
    for (int i=0; i<4; ++i) 
        magic_number |= train_images[i] << (24-(8*i));
    
	if (magic_number != 2051) {
        printf("MNIST train train_images magic number %d != 2051. Abort...", magic_number);
        return 1;
    }
    
	// -------------------------------------------------------------------------

    float this_image[IMAGE_SIZE];
    float this_label[1];
    uint16_t image_num  = 0;    //from [0,59999]

    get_next_image(this_image, this_label, image_num, train_images, train_labels);

    for (int i = 0;i<IMAGE_SIZE;i++) {
        printf("%.0f ", this_image[i]);
        if ((i+1)%IMAGE_COLS == 0)
            printf("\n");
    }

    printf("The %d th train image is %.1f\n", image_num, *this_label);







    return 0;
}
