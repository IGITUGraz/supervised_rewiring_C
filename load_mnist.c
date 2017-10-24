#include "load_mnist.h"

void read_bit_file(uint8_t *buff, size_t buff_len, char *file_path) {
    FILE *file;

    file = fopen(file_path, "rb");
    fread(buff, buff_len, 1, file);
    fclose(file);
}

void get_next_image(float *this_image, float *this_label, uint16_t image_num, uint8_t *images_set, uint8_t *labels_set) {
    //ensure all pointers are valid
    if (NULL == this_image) {
        fprintf(stderr, "this_image points to NULL!\n");
        exit(-1);
    }
    if (NULL == this_label) {
        fprintf(stderr, "this_label points to NULL!\n");
        exit(-1);
    }
    if (NULL == images_set) {
        fprintf(stderr, "images_set points to NULL!\n");
        exit(-1);
    }
    if (NULL == labels_set) {
        fprintf(stderr, "labels_set points to NULL!\n");
        exit(-1);
    }

    float norma_max = 255.0;

    for (uint i = 0; i < IMAGE_SIZE; i++)
        this_image[i] = *(images_set + TRAIN_IMAGE_OFFSET + image_num * IMAGE_SIZE + i) / norma_max;

    for (uint i = 0; i < NUM_CLASS; i++) {
        if (i == *(labels_set + TRAIN_LABEL_OFFSET + image_num))
            this_label[i] = 1.0;
        else
            this_label[i] = 0.0;
    }
}


