#include "test_network_gradient.h"

int test_network() {

    uint n_pixel = 3;
    uint n_1 = 4;
    uint n_2 = 4;

    uint nb_turnover_01 = 2;
    uint nb_turnover_12 = 2;
    uint nb_turnover_23 = 2;

    uint n_class = 3;
    float sparsity = 0.8;

    // Allocate the activity are error to be passed around
    float a_1[n_1];
    float a_2[n_2];
    float y[n_class];
    float prob[n_class];

    float delta_3[n_class];
    float delta_2[n_2];
    float delta_1[n_1];

    // Set weights onto first layer
    sparse_weight_matrix W_01;
    set_dimensions(&W_01,n_pixel,n_1);
    uint16_t rows_of_W_01[W_01.max_entries];
    uint16_t cols_of_W_01[W_01.max_entries];
    float_t thetas_of_W_01[W_01.max_entries];
    uint8_t bit_sign_storage_W_01[W_01.max_entries / 8 +1];

    W_01.rows = rows_of_W_01;
    W_01.cols = cols_of_W_01;
    W_01.thetas = thetas_of_W_01;
    W_01.bit_sign_storage = bit_sign_storage_W_01;

    set_random_weights_sparse_matrix(&W_01,sparsity);

    printf("\n Weight W 01: \n");
    print_sign_and_theta(&W_01);

    // Set weight onto second hidden layer
    sparse_weight_matrix W_12;
    set_dimensions(&W_12,n_1,n_2);
    uint16_t rows_of_W_12[W_12.max_entries];
    uint16_t cols_of_W_12[W_12.max_entries];
    float_t thetas_of_W_12[W_12.max_entries];
    uint8_t bit_sign_storage_of_W_12[W_12.max_entries / 8 +1];

    W_12.rows = rows_of_W_12;
    W_12.cols = cols_of_W_12;
    W_12.thetas = thetas_of_W_12;
    W_12.bit_sign_storage = bit_sign_storage_of_W_12;

    set_random_weights_sparse_matrix(&W_12,sparsity);

    printf("\n Weight W 12: \n");
    print_sign_and_theta(&W_12);

    // Set output weights
    sparse_weight_matrix W_23;
    set_dimensions(&W_23,n_2,n_class);
    uint16_t rows_of_W_23[W_23.max_entries];
    uint16_t cols_of_W_23[W_23.max_entries];
    float_t thetas_of_W_23[W_23.max_entries];
    uint8_t bit_sign_storage_of_W_23[W_23.max_entries / 8 +1];

    W_23.rows = rows_of_W_23;
    W_23.cols = cols_of_W_23;
    W_23.thetas = thetas_of_W_23;
    W_23.bit_sign_storage = bit_sign_storage_of_W_23;

    set_random_weights_sparse_matrix(&W_23,sparsity);

    printf("\n Weight W 23: \n");
    print_sign_and_theta(&W_23);

    // BEGINNING OF ITERATION

    // Set image and label, here we give non zero value to the image to have something to compute
    int k;
    float I[n_pixel];
    float label[n_class];


    for(k=0; k<n_class; k++)
        label[k]= 0;
    label[2] = 1.;

    for(k=0; k<n_pixel; k++)
        I[k] = 1.;


    // FORWARD PASS
    right_dot(I,NELEMS(I),&W_01,a_1,NELEMS(a_1));
    relu_in_place(a_1,NELEMS(a_1));

    right_dot(a_1,NELEMS(a_1),&W_12,a_2,NELEMS(a_2));
    relu_in_place(a_2,NELEMS(a_2));

    right_dot(a_2,NELEMS(a_2),&W_23,y,NELEMS(y));
    softmax(y,NELEMS(y),prob,NELEMS(prob));

    printf("\n Image: \n");
    print_vector(I,NELEMS(I));

    printf("\n a_1: \n");
    print_vector(a_1,NELEMS(a_1));

    printf("\n a_2: \n");
    print_vector(a_2,NELEMS(a_2));

    printf("\n Probability: \n");
    print_vector(prob,NELEMS(prob));

    printf("\n Label: \n");
    print_vector(label,NELEMS(label));

    // BACKWARD PASS
    vector_substraction(prob,NELEMS(prob),label,NELEMS(label),delta_3,NELEMS(delta_3));
    back_prop_error_msg(&W_23, a_2, NELEMS(a_2), delta_3, NELEMS(delta_3), delta_2, NELEMS(delta_2));
    back_prop_error_msg(&W_12, a_1, NELEMS(a_1), delta_2, NELEMS(delta_2), delta_1, NELEMS(delta_1));


    printf("\n delta 3: \n");
    print_vector(delta_3,NELEMS(delta_3));

    printf("\n delta_2: \n");
    print_vector(delta_2,NELEMS(delta_2));

    printf("\n delta_1: \n");
    print_vector(delta_1,NELEMS(delta_1));


    printf("\nBEFORE UPDATE:\n");
    printf("\nW_01:\n");
    print_weight_matrix(&W_01);

    printf("\nW_12:\n");
    print_weight_matrix(&W_12);

    printf("\nW_23:\n");
    print_weight_matrix(&W_23);

    // PLOT THE GRADIENTS

    printf("\nGradients wrt to W_01:\n");
    for(k=0; k<W_01.number_of_entries; k++){
        printf("  %u, %u -> %.2g \n",W_01.rows[k],W_01.cols[k],gradient_wrt_theta_entry(&W_01,I,NELEMS(I),delta_1,NELEMS(delta_1),k));
    }

    printf("\nGradients wrt to W_12:\n");
    for(k=0; k<W_12.number_of_entries; k++){
        printf("  %u, %u -> %.2g \n",W_12.rows[k],W_12.cols[k],gradient_wrt_theta_entry(&W_12,a_1,NELEMS(a_1),delta_2,NELEMS(delta_2),k));
    }


    printf("\nGradients wrt to W_23:\n");
    for(k=0; k<W_23.number_of_entries; k++){
        printf("  %u, %u -> %.2g \n",W_23.rows[k],W_23.cols[k],gradient_wrt_theta_entry(&W_23,a_2,NELEMS(a_2),delta_3,NELEMS(delta_3),k));
    }


    update_weight_matrix(&W_01,I,NELEMS(I),delta_1,NELEMS(delta_1));
    update_weight_matrix(&W_12,a_1,NELEMS(a_1),delta_2,NELEMS(delta_2));
    update_weight_matrix(&W_23,a_2,NELEMS(a_2),delta_3,NELEMS(delta_3));

    printf("\nAFTER UPDATE:\n");
    printf("\nW_01:\n");
    print_weight_matrix(&W_01);

    printf("\nW_12:\n");
    print_weight_matrix(&W_12);

    printf("\nW_23:\n");
    print_weight_matrix(&W_23);

    turnover(&W_01,nb_turnover_01);
    turnover(&W_12,nb_turnover_12);
    turnover(&W_23,nb_turnover_23);

    printf("\nAFTER TURNOVER:\n");

    printf("\nW_01:\n");
    print_weight_matrix(&W_01);


    printf("\nW_12:\n");
    print_weight_matrix(&W_12);

    printf("\nW_23:\n");
    print_weight_matrix(&W_23);

    // END OF ITERATION

    return 0;
}