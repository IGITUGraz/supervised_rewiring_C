#include "test_network_gradient.h"

#define SET_WEIGHTS(w, n1, n2, connectivity)               \
    sparse_weight_matrix w;                                \
    set_dimensions(&w,n1,n2,connectivity);                 \
                                                           \
    uint16_t rows_of_##w[w.max_entries];                   \
    uint16_t cols_of_##w[w.max_entries];                   \
    float_t thetas_of_##w[w.max_entries];                  \
    uint8_t bit_sign_storage_##w[w.max_entries / 8 +1];    \
                                                           \
    w.rows = rows_of_##w;                                  \
    w.cols = cols_of_##w;                                  \
    w.thetas = thetas_of_##w;                              \
    w.bit_sign_storage = bit_sign_storage_##w;             \
    w.max_entries = ceil(n1*n2*connectivity);              \
                                                           \
    set_random_weights_sparse_matrix(&w,connectivity);     \
    w.max_entries = w.number_of_entries;                   \
    printf("\n Weight %s: \n", #w);                        \
    print_sign_and_theta(&w);


int test_network() {

    uint n_pixel = 3;
    uint n_1 = 4;
    uint n_2 = 4;

    float connectivity_01 = 0.8;
    float connectivity_12 = 0.8;
    float connectivity_23 = 0.8;

    uint n_class = 3;
    float learning_rate = 0.05;

    // Allocate the activity are error to be passed around
    float a_1[n_1];
    float a_2[n_2];
    float y[n_class];
    float prob[n_class];

    float delta_3[n_class];
    float delta_2[n_2];
    float delta_1[n_1];

    // Set weights
    SET_WEIGHTS(W_01, n_pixel, n_1, connectivity_01);
    SET_WEIGHTS(W_12, n_1, n_2, connectivity_12);
    SET_WEIGHTS(W_23, n_2, n_class, connectivity_23);

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


    update_weight_matrix(&W_01,I,NELEMS(I),delta_1,NELEMS(delta_1), learning_rate);
    update_weight_matrix(&W_12,a_1,NELEMS(a_1),delta_2,NELEMS(delta_2), learning_rate);
    update_weight_matrix(&W_23,a_2,NELEMS(a_2),delta_3,NELEMS(delta_3), learning_rate);

    printf("\nAFTER UPDATE:\n");
    printf("\nW_01:\n");
    print_weight_matrix(&W_01);

    printf("\nW_12:\n");
    print_weight_matrix(&W_12);

    printf("\nW_23:\n");
    print_weight_matrix(&W_23);

    delete_entry(&W_01,0);
    delete_entry(&W_01,1);

    delete_entry(&W_12,1);
    delete_entry(&W_12,2);

    delete_entry(&W_23,1);
    delete_entry(&W_23,2);

    rewiring(&W_01,2);
    rewiring(&W_12,2);
    rewiring(&W_23,2);

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
