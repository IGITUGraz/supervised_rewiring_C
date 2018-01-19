#ifndef _MATH_OPERATIONS_LIB_
#define _MATH_OPERATIONS_LIB_

    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include <assert.h>
    #include <stdint.h>
    #include <stdbool.h>

    #define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))

//Get the sign of the k-th entry.
#define get_sign(M, entry_idx) (bool)(((M)->bit_sign_storage[(entry_idx) / 8] >> ((entry_idx) % 8)) & 0x1)
//Generate a random number with uniform distribution within 0 and 1.
#define rand_kiss() (float)(mars_kiss32() / 4294967296.0)
//Compute the rectified linear function of v in place.
#define relu_in_place(v, size_v) for(uint16_t kk = 0; kk < (size_v); kk++) if ((v)[kk] < 0) (v)[kk] = 0
//Make the substraction of two vectors
#define vector_substraction(a, size_a, b, size_b, result, size_result) for (uint16_t kk = 0; kk < (size_a); kk++) (result)[kk] = (a)[kk] - (b)[kk]
//Set the sign of the k-th entry in the matrix.
#define set_sign(M, entry_idx, val)                                             \
if ((val)) (M)->bit_sign_storage[(entry_idx) / 8] |= 1 << ((entry_idx) % 8);    \
else (M)->bit_sign_storage[(entry_idx) / 8] &= ~(1 << ((entry_idx) % 8))
//position(32bit)=row(16bit),col(bit)
#define get_flattened_position(M, entry)    (uint32_t)(((M)->rows[(entry)] << 16) + (M)->cols[(entry)])
//row is the high 16 bits of position, if col = n_col, row++
#define get_row_from_position(M, position)  (uint16_t)(((position)>>16)+((((position) & 0xffff)>=(M)->n_cols)?1:0))
//col is the low 16 bits of position, if col = n_col, col = 0
#define get_col_from_position(M, position)  (uint16_t)((((position) & 0xffff) == (M)->n_cols)?0:((position) & 0xffff))



    /**
     * Random numbers
    **/
    uint32_t mars_kiss32( void );
    float    randn_kiss( void );

    /**
     * SPARSE MATRICES BASICS
     */

    struct sparse_weight_matrix {
        uint16_t max_entries;
        uint16_t number_of_entries;

        uint16_t n_rows;
        uint16_t n_cols;

        uint16_t *rows;
        uint16_t *cols;
        uint8_t *bit_sign_storage;
        float *thetas;
    };
    typedef struct sparse_weight_matrix sparse_weight_matrix;

    float get_weight_by_entry(sparse_weight_matrix *M, int k);
    int is_entry_and_fetch(sparse_weight_matrix *M, int i, int j);
    float get_theta_by_row_col_pair(sparse_weight_matrix *M, int i, int j);
    int get_sign_by_row_col_pair(sparse_weight_matrix *M, int i, int j);

    void set_dimensions(sparse_weight_matrix *M, uint16_t n_rows, uint16_t n_cols, float connectivity);
    void check_sparse_matrix_format(sparse_weight_matrix *M);

    void set_random_weights_sparse_matrix(sparse_weight_matrix *M, float connectivity);
    void put_new_entry(sparse_weight_matrix *M, uint16_t row, uint16_t col, float value, bool sign, bool fail_if_exist);
    void delete_entry(sparse_weight_matrix *M, uint16_t entry_idx);

    void quickSort(sparse_weight_matrix *M, uint16_t entry_low, uint16_t entry_high);

    void put_new_random_entries(sparse_weight_matrix *M, uint16_t n_new);
    void delete_negative_entries(sparse_weight_matrix *M);

    void print_weight_matrix(sparse_weight_matrix *M);
    void print_sign_and_theta(sparse_weight_matrix *M);
    void print_vector(float *v, uint size);

    void check_order(sparse_weight_matrix *M, uint16_t entry_low, uint16_t entry_high);
    void print_weight_matrix_containers(sparse_weight_matrix *M, bool print_non_assigned);
    uint32_t rand_int_kiss(uint32_t low, uint32_t high);

    /**
     * MATH OPERATIONS
     */

    void left_dot(sparse_weight_matrix *M, float *v, uint size_v, float *result, uint size_result);
    void right_dot(float *v, uint size_v, sparse_weight_matrix *M, float *result, uint size_result);
    void term_by_term_multiplication(float *a, uint size_a, float *b, uint size_b, float *result, uint size_result);

    void back_prop_error_msg(sparse_weight_matrix *W, float *a_pre, uint size_a_pre, float *d_post,
                             uint size_d_post, float *d_pre, uint size_d_pre);
    float gradient_wrt_theta_entry(sparse_weight_matrix *weight_matrix, float *a_pre, uint size_a_pre, float *d_post, uint size_d_post, uint16_t entry_idx);
    void update_weight_matrix(sparse_weight_matrix *W, float *a_pre, uint size_a_pre, float *d_post, uint size_d_post, float learning_rate);
    void rewiring(sparse_weight_matrix *W);

    void softmax(float *a, uint size_a, float *result, uint size_result);

    uint8_t argmax (float *prob, uint8_t size_prob);
    void rewiring2(sparse_weight_matrix *M);
#endif