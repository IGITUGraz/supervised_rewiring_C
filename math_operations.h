#ifndef _MATH_OPERATIONS_LIB_
#define _MATH_OPERATIONS_LIB_

    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include <assert.h>
    #include <stdint.h>
    #include <stdbool.h>

    #define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))

    /*
     * Random numbers
     */

    uint32_t mars_kiss32( void );
    float rand_kiss( void );
    float randn_kiss( void );

    /**
     * SPARSE MATRICES BASICS
     */

    struct sparse_weight_matrix {
        uint32_t max_entries;
        uint32_t number_of_entries;

        uint32_t n_rows;
        uint32_t n_cols;

        uint32_t *rows;
        uint32_t *cols;
        uint8_t *bit_sign_storage;
        float *thetas;
    };
    typedef struct sparse_weight_matrix sparse_weight_matrix;

    bool get_sign(sparse_weight_matrix *M, uint32_t entry_idx);
    void set_sign(sparse_weight_matrix *M, uint32_t entry_idx, bool val);
    float get_weight_by_entry(sparse_weight_matrix *M, int k);
    int is_entry_and_fetch(sparse_weight_matrix *M, int i, int j);
    float get_theta_by_row_col_pair(sparse_weight_matrix *M, int i, int j);
    int get_sign_by_row_col_pair(sparse_weight_matrix *M, int i, int j);

    void set_dimensions(sparse_weight_matrix *M, uint32_t n_rows, uint32_t n_cols);
    void check_sparse_matrix_format(sparse_weight_matrix *M);

    void set_random_weights_sparse_matrix(sparse_weight_matrix *M, float connectivity);
    void put_new_entry(sparse_weight_matrix *M, uint32_t row, uint32_t col, float value, bool sign, bool fail_if_exist);
    void delete_entry(sparse_weight_matrix *M, uint32_t entry_idx);

    void quickSort(sparse_weight_matrix *M, uint32_t low, uint32_t high);

    void put_new_random_entries(sparse_weight_matrix *M, uint32_t n_new);
    void delete_negative_entries(sparse_weight_matrix *M);

    void print_weight_matrix(sparse_weight_matrix *M);
    void print_sign_and_theta(sparse_weight_matrix *M);
    void print_vector(float *v, uint size);

    /**
     * MATH OPERATIONS
     */

    void relu_in_place(float *v, uint size_v);
    void vector_substraction(float *a, uint size_a, float* b, uint size_b, float *result, uint size_result);

    void left_dot(sparse_weight_matrix *M, float *v, uint size_v, float *result, uint size_result);
    void right_dot(float *v, uint size_v, sparse_weight_matrix *M, float *result, uint size_result);
    void term_by_term_multiplication(float *a, uint size_a, float *b, uint size_b, float *result, uint size_result);

    void back_prop_error_msg(sparse_weight_matrix *W, float *a_pre, uint size_a_pre, float *d_post,
                             uint size_d_post, float *d_pre, uint size_d_pre);
    float gradient_wrt_theta_entry(sparse_weight_matrix *weight_matrix, float *a_pre, uint size_a_pre, float *d_post, uint size_d_post, uint32_t entry_idx);
    void update_weight_matrix(sparse_weight_matrix *W, float *a_pre, uint size_a_pre, float *d_post, uint size_d_post, float learning_rate);
    void rewiring(sparse_weight_matrix *W, uint32_t rewiring_number);

    void softmax(float *a, uint size_a, float *result, uint size_result);

    uint8_t argmax (float *prob, uint8_t size_prob);
#endif