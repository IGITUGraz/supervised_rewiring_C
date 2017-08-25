#include <jmorecfg.h>
#include "math_operations.h"

// a constant definition exported by library:
#define SPARSITY_LIMIT  .99 // SET to 0.02 to fit on the hardware
#define LEARNING_RATE 0.5
#define L1_COEFF 0.00001
#define NOISE_AMPLITUDE 0.000001
#define EPSILON_TURNOVER 0.000001

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
                M->thetas[k] = fabs(value);

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
 *  Add an entry in the matrix in a free slot.
 *  This is not trivial because the order in the sparse matrix need to be respected.
 *
 * @param M
 * @param row
 * @param col
 * @param value
 * @param sign
 */
void put_new_entry(sparse_weight_matrix *M, uint16_t row, uint16_t col, float value, bool sign, bool fail_if_exist){

    uint16_t k;

    uint16_t pushed_row = row;
    uint16_t pushed_col = col;
    float pushed_theta = value;
    bool pushed_sign = sign;

    uint16_t replaced_row;
    uint16_t replaced_col;
    float replaced_theta;
    bool replaced_sign;

    assert(row < M->n_rows);
    assert(col < M->n_cols);

    // Find the first entry that arrives after the new one.
    k=0;
    while(M->rows[k] < row)
        k++;

    while(M->rows[k] == row && M->cols[k] < col)
        k++;

    // Make sure that the entry does not exist already.
    if(fail_if_exist)
        assert(M->rows[k] != row || M->cols[k] != col);
    else{
        if(M->rows[k] == row && M->cols[k] == col){
            return;
        }
    }

    // Push-replace the value in the vector until there is only the last entry left.
    while(k < M->number_of_entries){
        replaced_row = M->rows[k];
        replaced_col = M->cols[k];
        replaced_theta = M->thetas[k];
        replaced_sign = get_sign(M,k);

        M->rows[k] = pushed_row;
        M->cols[k] = pushed_col;
        M->thetas[k] = pushed_theta;
        set_sign(M,k,pushed_sign);

        pushed_row = replaced_row;
        pushed_col = replaced_col;
        pushed_theta = replaced_theta;
        pushed_sign = replaced_sign;

        k++;
    }

    // Insert the last one
    M->rows[k] = pushed_row;
    M->cols[k] = pushed_col;
    M->thetas[k] = pushed_theta;
    set_sign(M,k,pushed_sign);

    M->number_of_entries += 1;
    assert(M->number_of_entries < M->max_entries);
    check_sparse_matrix_format(M);
}

/**
 * Compute the rectified linear function of v in place.
 * @param v
 * @param size_v
 */
void relu_in_place(float *v, uint size_v){
    uint k;

    for(k=0; k<size_v; k++){
        if (v[k] < 0)
            v[k] = 0;
    }
}

/**
 * Delete an entry in the matrix.
 * @param M
 * @param entry_idx
 */
void delete_entry(sparse_weight_matrix *M, uint16_t entry_idx){

    uint16_t k;
    assert(entry_idx < M->number_of_entries);

    for(k = entry_idx; k < M->number_of_entries-1; k ++){
        M->rows[k] = M->rows[k+1];
        M->cols[k] = M->cols[k+1];
        M->thetas[k] = M->thetas[k+1];
        set_sign(M,k,get_sign(M,k+1));
    }

    M->number_of_entries -= 1;

}

/**
 * Make the substraction of two vectors
 * @param a
 * @param size_a
 * @param b
 * @param size_b
 * @param result
 * @param size_result
 */
void vector_substraction(float *a, uint size_a, float* b, uint size_b, float *result, uint size_result){
    assert(size_a == size_b);
    assert(size_a == size_result);

    uint k;

    for(k=0; k<size_a; k++){
        result[k] = a[k] - b[k];
    }

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
 * Get the weight of a matrix at a given pair of row/column index.
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
 * Get the parameter theta of a matrix at a given pair of row/column index.
 * WARNING:
 *  This is very slow because it require to search in the array.
 *  Use get_weight_by_entry if efficiency is an issue.
 *
 * @param M
 * @param i
 * @param j
 * @return
 */
float get_theta_by_row_col_pair(sparse_weight_matrix *M, int i, int j) {
    float th;
    int k;

    th = 0;
    for(k=0; k < M->number_of_entries; k++){
        if(M->rows[k] == i && M->cols[k] == j)
            th = M->thetas[k];
    }

    return th;

}


/**
 * Get the sign of a matrix at a given pair of row/column index.
 * WARNING:
 *  This is very slow because it require to search in the array.
 *  Use get_weight_by_entry if efficiency is an issue.
 *
 * @param M
 * @param i
 * @param j
 * @return
 */
int get_sign_by_row_col_pair(sparse_weight_matrix *M, int i, int j) {
    int s;
    int k;

    s = 0;
    for(k=0; k < M->number_of_entries; k++){
        if(M->rows[k] == i && M->cols[k] == j)
            if(get_sign(M,k))
                s = 1;
            else
                s = -1;
    }

    return s;

}

/**
 * Print a weight matrix in the python matrix format.
 * @param v
 * @param size
 */
void print_weight_matrix(sparse_weight_matrix *M) {

    int i;
    int j;

    check_sparse_matrix_format(M);

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
 * Print a weight matrix in the python matrix format.
 * @param v
 * @param size
 */
void print_sign_and_theta(sparse_weight_matrix *M) {

    int i;
    int j;
    bool s;

    check_sparse_matrix_format(M);

    printf("sign:\n [");
    for(i = 0; i < M->n_rows; i += 1){
        if(i>0)
            printf("\n");

        printf("[");
        for(j = 0; j < M->n_cols; j += 1){
            printf("%d",get_sign_by_row_col_pair(M,i,j));

            if(j<M->n_cols-1)
                printf(", ");

        }
        printf("]");
        if(i<M->n_rows-1)
            printf(", ");
    }
    printf("] \n");

    printf("\nthetas:\n [");
    for(i = 0; i < M->n_rows; i += 1){
        if(i>0)
            printf("\n");

        printf("[");
        for(j = 0; j < M->n_cols; j += 1){
            printf("%.2g",get_theta_by_row_col_pair(M,i,j));

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
 * The sparse matrix vector right-dot is done as follows:
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
void left_dot( sparse_weight_matrix *M, float *v, uint size_v, float *result, uint size_result) {

    int i;
    int j;
    int k;

    float w;

    check_sparse_matrix_format(M);
    assert(size_v == M->n_cols);
    assert(size_result == M->n_rows);

    for(i=0; i<size_result; i++)
        result[i] = 0;

    for(k=0; k<M->number_of_entries; k++){
        i = M->rows[k];
        j = M->cols[k];
        w = get_weight_by_entry(M,k);

        if(v[j] != 0)
            result[i] += w * v[j];
    }
}


/**
 * The sparse matrix vector right-dot is done as follows:
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
void right_dot(float *v, uint size_v, sparse_weight_matrix *M, float *result, uint size_result) {

    int i;
    int j;
    int k;

    float w;

    check_sparse_matrix_format(M);
    assert(size_v == M->n_rows);
    assert(size_result == M->n_cols);

    for(j=0; j<size_result; j++)
        result[j] = 0;

    for(k=0; k<M->number_of_entries; k++){
        i = M->rows[k];
        j = M->cols[k];
        w = get_weight_by_entry(M,k);

        if(v[i] != 0)
            result[j] += w * v[i];
    }
}

/**
 * Check that the index of a matrix are well ordered.
 * @param M
 */
void check_sparse_matrix_format(sparse_weight_matrix *M){
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

        assert(M->thetas > 0); //Thetas has to be positive if we do not simulate disconnected synapses

    }

}

/**
 * Compute the gradient with respect to a non-zero synaptic parameter theta.
 * @param weight_matrix
 * @param pre_synaptic_activity
 * @param post_synaptic_error_msg
 * @param entry_idx
 * @return
 */
float gradient_wrt_theta_entry(sparse_weight_matrix *weight_matrix, float *a_pre, uint size_a_pre, float *d_post, uint size_d_post, uint16_t entry_idx){

    bool sign = get_sign(weight_matrix,entry_idx);
    int i = weight_matrix->rows[entry_idx];
    int j = weight_matrix->cols[entry_idx];
    float value;

    assert(size_a_pre > i);
    assert(size_d_post > j);

    if(a_pre[i] != 0 && d_post[j] != 0)
        value = a_pre[i] * d_post[j];
    else
        value = 0;

    if(sign)
        return value;
    else
        return - value;
}

/**
 * Fill the vector of error message at the pre-synaptic layer with the corresponding values.
 * @param weight_matrix
 * @param a_pre
 * @param size_a_pre
 * @param delta_post
 * @param size_d_post
 * @param delta_pre
 * @param size_delta_pre
 */
void back_prop_error_msg(sparse_weight_matrix *W, float *a_pre, uint size_a_pre, float *d_post,
                         uint size_d_post, float *d_pre, uint size_d_pre){

    int i;
    int j;
    int k;

    float w;

    check_sparse_matrix_format(W);
    assert(size_d_post == W->n_cols);
    assert(size_d_pre == W->n_rows);
    assert(size_a_pre == W->n_rows);

    for(i=0; i<size_d_pre; i++)
        d_pre[i] = 0;

    for(k=0; k<W->number_of_entries; k++){
        i = W->rows[k];
        j = W->cols[k];
        w = get_weight_by_entry(W,k);

        if(d_post[j] != 0 && a_pre[i] > 0) // Dot the term-by-term product directly with the derivative of a_pre
            d_pre[i] += w * d_post[j];
    }
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
        if(a[k] !=0 && b[k] != 0)
            result[k] = a[k] * b[k];
        else
            result[k] = 0;
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

/**
 *
 * @param W
 * @param a_pre
 * @param size_a_pre
 * @param d_post
 * @param size_d_post
 */
void update_weight_matrix(sparse_weight_matrix *W, float *a_pre, uint size_a_pre, float *d_post, uint size_d_post){

    uint16_t k;
    float grad;
    float dtheta;

    for(k=0; k<W->number_of_entries; k++){
        grad = gradient_wrt_theta_entry(W,a_pre,size_a_pre,d_post,size_d_post,k);
        dtheta = LEARNING_RATE * ( - L1_COEFF - grad + NOISE_AMPLITUDE * randn_kiss());
        W->thetas[k] += dtheta;

    }

    k = W->number_of_entries;
    while(k>0){
        k -=1;

        if(W->thetas[k] <= 0)
            delete_entry(W,k);  // Delete the element
    }

}


void turnover(sparse_weight_matrix *W, uint16_t turnover_number){
    uint k;
    uint i;
    uint j;
    bool sign;

    for(k = 0; k<turnover_number; k++){

        i = trunc(rand_kiss() * W->n_rows);
        j = trunc(rand_kiss() * W->n_cols);
        sign = rand_kiss() > 0.5;

        put_new_entry(W,i,j,EPSILON_TURNOVER,sign,false);
    }

}