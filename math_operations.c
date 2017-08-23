#include "math_operations.h"

/**
 * Generate a random integer.
 * Implementation of a Marsaglia 32-bit KISS generator which uses no multiply instructions≈2^121 period
 * 209.9 nanosecs (i.e. 42 ticks) per call (on SpiNNaker 1 ARM core)
 * @return
 */
uint32_t mars_kiss32( void )
{
    static uint32_t x = 123456789, y = 234567891, z = 345678912, w = 456789123, c = 0; /* default seed variables */
    int32_t t;

    y ^= ( y << 5 ); y ^= ( y >> 7 ); y ^= ( y << 22 );
    t = z + w + c;
    z = w;
    c = t < 0;
    w =  t & 2147483647;
    x += 1411392427;

    return x + y + w;
}

/**
 * Generate a random number with uniform distribution within 0 and 1.
 * @return
 */
float rand_kiss( void ){
    return mars_kiss32() / 4294967296.0;
}

/**
 * Generate a random gaussian variable with mean 0 and standard deviation 1.
 * @return
 */
float randn_kiss( )
{
    static bool
            set = false;
    static float gset;
    float fac, rsq, v1, v2;

    if  ( !set ) {
        do {

            v1 = ( mars_kiss32() / 2147483648.0f ) - 1.0f;   // U(0,1) * 2 - 1
            v2 = ( mars_kiss32() / 2147483648.0f ) - 1.0f;
            rsq = v1*v1 + v2*v2;
        } while ( rsq >= 1.0f || rsq == 0.0f );

        fac = sqrtf( -2.0f * logf( rsq ) / rsq );
        gset = v1 * fac;
        set = true;
        return (float) v2 * fac;
    }
    else {

        set = false;
        return (float) gset;

    }
}


/**
 * Fill-in a sparse matrix with random value.
 * It uses a Xavier initialization of the form: mask * randn(n_in,n_out)/n_in,
 * if randn generates gaussian number, n_in is the number of inputs, n_out is the number of outputs.
 * The mask allows non zero values at each position with probability "sparsity": mask = rand(n_in,n_out) < sparsity
 * @param M
 * @param sparsity
 */
void set_random_weights_sparse_matrix(sparse_weight_matrix *M, float sparsity) {

    int k;
    int i;
    int j;
    float value;

    k = 0;
    for(i = 0; i < M->n_rows; i += 1){
        for(j = 0; j < M->n_cols; j += 1) {

            if(rand_kiss() < sparsity) {
                value = randn_kiss() / sqrt(M->n_rows);

                M->rows[k] = i;
                M->cols[k] = j;
                M->thetas[k] = value;

                if(rand_kiss() > 0.5)
                    set_sign(M,k,true);
                else
                    set_sign(M,k,false);

                k+=1;
                assert(k<=M->max_entries);
            }
        }
    }

    M->number_of_entries = k;
}

/**
 * Set the dimensions of the matrix and defines the maxium number of entries.
 * @param M
 * @param n_rows
 * @param n_cols
 */
void set_dimensions(sparse_weight_matrix *M, int n_rows, int n_cols) {
    M->n_rows = n_rows;
    M->n_cols = n_cols;
    M->max_entries = ceil(n_rows * n_cols * SPARSITY_LIMIT);
}

/**
 * Get the sign of the k-th entry.
 * @param M
 * @param entry_idx
 * @return
 */
bool get_sign(sparse_weight_matrix *M, uint16_t entry_idx) {
    uint8_t sign = M->bit_sign_storage[entry_idx/8];
    sign = (uint8_t)((sign >> (entry_idx%8)) & 0x1);

    return (bool)sign;
}

/**
 * Set the sign of the k-th entry in the matrix.
 * @param M
 * @param entry_idx
 * @param val
 */
void set_sign(sparse_weight_matrix *M, uint16_t entry_idx, bool val) {
    M->bit_sign_storage[entry_idx/8] |= (val ? 1 : 0) << (entry_idx%8);
}

/**
 * Find the weight of the k-th recorded entry.
 * @param M
 * @param k
 * @return
 */
float get_weight_by_entry(sparse_weight_matrix *M, int k ){
    bool s;
    assert(k<M->number_of_entries);

    if(M->thetas[k] <= 0)
        return 0.;
    else{
        s = get_sign(M,k);
        if(s)
            return M->thetas[k];
        else
            return - M->thetas[k];
    }

}

/**
 * Get the weight of a matrix at a given column index.
 * WARNING:
 *  This is very slow because it require to search in the array.
 *  Use get_weight_by_entry if efficiency is an issue.
 *
 * @param M
 * @param i
 * @param j
 * @return
 */
float get_weight_by_row_col_pair(sparse_weight_matrix *M, int i, int j) {
    float w;
    int k;

    w = 0;
    for(k=0; k < M->number_of_entries; k++){
        if(M->rows[k] == i && M->cols[k] == j)
            w = get_weight_by_entry(M,k);
    }

    return w;

}

/**
 * Print a weight matrix in the python matrix format.
 * @param v
 * @param size
 */
void print_weight_matrix(sparse_weight_matrix *M) {

    int i;
    int j;

    check_sparse_matrix_entry_ordering(M);

    printf("[");
    for(i = 0; i < M->n_rows; i += 1){
        if(i>0)
            printf("\n");

        printf("[");
        for(j = 0; j < M->n_cols; j += 1){
            printf("%.2g",get_weight_by_row_col_pair(M,i,j));
            if(j<M->n_cols-1)
                printf(", ");

        }
        printf("]");
        if(i<M->n_rows-1)
            printf(", ");
    }
    printf("] \n");
}

/**
 * Print an array of floats a vector in the python format.
 * @param v
 * @param size
 */
void print_vector(float *v, uint size){
    uint k;

    printf("[");
    for(k=0; k<size; k++){
        printf("%.2g",v[k]);
        if(k<size-1)
            printf(", ");

    }
    printf("] \n");

}

/**
 * The sparse matrix vector dot is done as follows:
 * - For each row,
 *      Check the entries in the matrix that are contained in that row.
 *      Multiply them with the corresponding vector entry
 *      Add the results
 *
 * @param M
 * @param v
 * @param result
 * @return
 */
void sparse_matrix_vector_dot(sparse_weight_matrix *M, float *v, uint size_v, float *result, uint size_result) {

    int i;
    int j;
    int k;

    float w;
    float res_i;

    check_sparse_matrix_entry_ordering(M);
    assert(size_v == M->n_cols);
    assert(size_result == M->n_rows);

    k=0;
    for(i=0; i<M->n_rows; i++){
        res_i = 0;

        assert(M->rows[k]>=i); // Until now necessarily the row of the next entry is larger than the current row
        while(M->rows[k] == i){

            j = M->cols[k];
            w = get_weight_by_entry(M,k);
            res_i += w * v[j];

            k++;
        }

        result[i] = res_i;

    }
}

/**
 * Check that the index of a matrix are well ordered.
 * @param M
 */
void check_sparse_matrix_entry_ordering(sparse_weight_matrix *M){
    int k;
    int last_i;
    int last_j;

    last_i = -1;
    last_j = -1;

    for(k=0; k<M->number_of_entries; k++){
        assert(M->rows[k] >= last_i);

        if(M->rows[k] == last_i)
            assert(M->cols[k] > last_j);

        last_i = M->rows[k];
        last_j = M->cols[k];

    }

}

/**
 * Compute the gradient with respect to a non-zero synaptic parameter theta.
 * @param weight_matrix
 * @param pre_synaptic_activation
 * @param post_synaptic_error_msg
 * @param entry_idx
 * @return
 */
float gradient_wrt_theta_entry(sparse_weight_matrix *weight_matrix, float *pre_synaptic_activation, float *post_synaptic_error_msg, uint16_t entry_idx){

    bool sign = get_sign(weight_matrix,entry_idx);
    int i = weight_matrix->rows[entry_idx];
    int j = weight_matrix->cols[entry_idx];
    float value = pre_synaptic_activation[i] * post_synaptic_error_msg[j];

    if(sign)
        return value;
    else
        return - value;
}

/**
 * Term by term vector multiplication.
 * @param a
 * @param size_a
 * @param b
 * @param size_b
 * @param result
 * @param size_result
 */
void term_by_term_multiplication(float *a, uint size_a, float *b, uint size_b, float *result, uint size_result){

    uint k;
    assert(size_a == size_b);
    assert(size_a == size_result);

    for(k=0; k<size_a; k++)
        result[k] = a[k] * b[k];
}

/**
 * Stable computation of a softmax function.
 *
 * @param a
 * @param result
 */
void softmax(float *a, uint size_a, float *result, uint size_result){

    float a_max;
    float sum;
    uint k;

    assert(size_a == size_result);

    a_max = a[0];
    for(k=0; k<size_a; k++){
        if(a_max<a[k])
            a_max = a[k];
    }


    sum=0;
    for(k=0; k<size_a; k++){
        result[k] = exp(a[k] - a_max);
        sum += result[k];
    }

    for(k=0; k<size_a; k++)
        result[k] /= sum;
}