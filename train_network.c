#include <time.h>
#include "train_network.h"

#define SET_WEIGHTS(w, n1, n2, connectivity)                 \
    sparse_weight_matrix w;                                  \
    set_dimensions(&(w),n1,n2);                              \
                                                             \
    uint16_t rows_of_##w[(w).max_entries];                   \
    uint16_t cols_of_##w[(w).max_entries];                   \
    float_t thetas_of_##w[(w).max_entries];                  \
    uint8_t bit_sign_storage_##w[(w).max_entries / 8 +1];    \
                                                             \
    (w).rows = rows_of_##w;                                  \
    (w).cols = cols_of_##w;                                  \
    (w).thetas = thetas_of_##w;                              \
    (w).bit_sign_storage = bit_sign_storage_##w;             \
                                                             \
    set_random_weights_sparse_matrix(&(w),connectivity);     //\
                                                             \
    printf("\n Weight %s: \n", #w);                          \
    print_sign_and_theta(&w);

#define NUM_EPOCH                   3
#define TRAIN_PERIOD             5000
#define TEST_PERIOD             10000



int train_network() {

    //define network parameter
    uint16_t n_pixel = IMAGE_SIZE;
    uint16_t n_1 = 300;
    uint16_t n_2 = 100;
    uint16_t n_class = NUM_CLASS;

    float connectivity_01 = 0.01;
    float connectivity_12 = 0.03;
    float connectivity_23 = 0.3;

    // Allocate the activity are error to be passed around
    float a_1[n_1];
    float a_2[n_2];
    float y[n_class];
    float prob[n_class];

    float delta_3[n_class];
    float delta_2[n_2];
    float delta_1[n_1];

    uint scoreboard;
    float accuracy;

    time_t rawtime;
    struct tm * timeinfo;
    clock_t t1, t2, t3, t4, t5;

    // Set weights
    SET_WEIGHTS(W_01, n_pixel, n_1, connectivity_01);
    SET_WEIGHTS(W_12, n_1, n_2, connectivity_12);
    SET_WEIGHTS(W_23, n_2, n_class, connectivity_23);

    const uint32_t target_01  = W_01.number_of_entries;
    const uint32_t target_12  = W_12.number_of_entries;
    const uint32_t target_23  = W_23.number_of_entries;

    //import data set
    //                      TRAIN_SET                                                              TEST_SET
    uint8_t *train_images = malloc(TRAIN_IMAGEF_BUFF*sizeof(uint8_t));      uint8_t *test_images = malloc(TEST_IMAGEF_BUFF*sizeof(uint8_t));
    uint8_t *train_labels = malloc(TRAIN_LABELF_BUFF*sizeof(uint8_t));      uint8_t *test_labels = malloc(TEST_LABELF_BUFF*sizeof(uint8_t));
    read_bit_file(train_images, TRAIN_IMAGEF_BUFF, MNIST_TRAIN_IMG_FILE);   read_bit_file(test_images, TEST_IMAGEF_BUFF, MNIST_TEST_IMG_FILE);
    read_bit_file(train_labels, TRAIN_LABELF_BUFF, MNIST_TRAIN_LAB_FILE);   read_bit_file(test_labels, TEST_LABELF_BUFF, MNIST_TEST_LAB_FILE);

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

    float train_image[n_pixel];         float test_image[n_pixel];
    float train_label[n_class];         float test_label[n_class];
    uint16_t train_image_num = 0;       uint16_t test_image_num = 0;

    //record beginning time
    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "begin at : %s", asctime (timeinfo) );

    printf("test report:\n");
    printf("epoch\titeration\tsynapses_W01\tsynapses_W12\tsynapses_W23\taccuracy\tt_iteration\t\tt_get_image\t\tt_forward\tt_backward\tt_rewiring\n");

    // BEGIN OF EPOCH
    for (uint epoch = 0; epoch < NUM_EPOCH; epoch++) {
        //BEGIN OF ITERATION
        for (train_image_num = 0; train_image_num < NUM_TRAIN; train_image_num++){
            //for (train_image_num = 0; train_image_num < 2; train_image_num++){

            //###################### TRAIN PHASE ######################
            //fetch time
            if ((train_image_num + 1) % TRAIN_PERIOD == 0)
                t1 = clock();

            //get current train image and label
            get_next_image(train_image, train_label, train_image_num, train_images, train_labels);

            //fetch time
            if ((train_image_num + 1) % TRAIN_PERIOD == 0)
                t2 = clock();

            // FORWARD PASS
            right_dot(train_image,NELEMS(train_image),&W_01,a_1,NELEMS(a_1));
            relu_in_place(a_1,NELEMS(a_1));

            right_dot(a_1,NELEMS(a_1),&W_12,a_2,NELEMS(a_2));
            relu_in_place(a_2,NELEMS(a_2));

            right_dot(a_2,NELEMS(a_2),&W_23,y,NELEMS(y));
            softmax(y,NELEMS(y),prob,NELEMS(prob));

            //fetch time
            if ((train_image_num + 1) % TRAIN_PERIOD == 0)
                t3 = clock();

            // BACKWARD PASS
            vector_substraction(prob,NELEMS(prob),train_label,NELEMS(train_label),delta_3,NELEMS(delta_3));
            back_prop_error_msg(&W_23, a_2, NELEMS(a_2), delta_3, NELEMS(delta_3), delta_2, NELEMS(delta_2));
            back_prop_error_msg(&W_12, a_1, NELEMS(a_1), delta_2, NELEMS(delta_2), delta_1, NELEMS(delta_1));

            update_weight_matrix(&W_01,train_image,NELEMS(train_image),delta_1,NELEMS(delta_1));
            update_weight_matrix(&W_12,a_1,NELEMS(a_1),delta_2,NELEMS(delta_2));
            update_weight_matrix(&W_23,a_2,NELEMS(a_2),delta_3,NELEMS(delta_3));

            //fetch time
            if ((train_image_num + 1) % TRAIN_PERIOD == 0)
                t4 = clock();

            rewiring(&W_01, (uint16_t)(target_01 - W_01.number_of_entries));
            rewiring(&W_12, (uint16_t)(target_12 - W_12.number_of_entries));
            rewiring(&W_23, (uint16_t)(target_23 - W_23.number_of_entries));

            //fetch time
            if ((train_image_num + 1) % TRAIN_PERIOD == 0)
                t5 = clock();

            //###################### TEST PHASE ########################
            //compute one accuracy every 1k training
            if ((train_image_num + 1) % TRAIN_PERIOD == 0){
                scoreboard  = 0;
                for (uint16_t test_loop = 0; test_loop < TEST_PERIOD; test_loop++){
                    //get current test image and label
                    get_next_image(test_image, test_label, test_image_num + test_loop, test_images, test_labels);

                    // FORWARD PASS for test
                    right_dot(test_image,NELEMS(test_image),&W_01,a_1,NELEMS(a_1));
                    relu_in_place(a_1,NELEMS(a_1));

                    right_dot(a_1,NELEMS(a_1),&W_12,a_2,NELEMS(a_2));
                    relu_in_place(a_2,NELEMS(a_2));

                    right_dot(a_2,NELEMS(a_2),&W_23,y,NELEMS(y));
                    softmax(y,NELEMS(y),prob,NELEMS(prob));

                    scoreboard   = scoreboard + (uint)(argmax(prob, NELEMS(prob)) == argmax(test_label,NELEMS(test_label)));
                }
                if(test_image_num + TEST_PERIOD == NUM_TEST) //update test_image_num
                    test_image_num = 0;
                else
                    test_image_num += TEST_PERIOD;
                accuracy    = (float)scoreboard / TEST_PERIOD;

                //show test result
                printf("\t%1d\t\t%5d\t\t\t%d\t\t\t%d\t\t\t\t%d\t\t%.3f\t\t\t%.5f\t\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%%\n",                   \
                        epoch, train_image_num, W_01.number_of_entries, W_12.number_of_entries, W_23.number_of_entries, accuracy,                    \
                       (t5-t1)/(float)CLOCKS_PER_SEC,(t2-t1)*100./(t5-t1),(t3-t2)*100./(t5-t1),(t4-t3)*100./(t5-t1),(t5-t4)*100.0/(t5-t1));

                //TODO save file, or use > math_in_C | tee outfile
            }//END OF TEST PHASE
        }//END OF ITERATION
        train_image_num = 0;
    }//END OF EPOCH

    //release memory
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);

    //record end time
    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "end at : %s", asctime (timeinfo) );

    return 0;
}
