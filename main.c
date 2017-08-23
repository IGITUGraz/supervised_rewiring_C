#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <expat.h>

#include "math_operations.h"

int main() {
    uint i;
    i = sizeof(bool);
    printf("%u \n",i);

    // Test for creating a matrix and printing it

    sparse_weight_matrix M;
    set_dimensions(&M,3,4);
    uint16_t rows_of_M[M.max_entries];
    uint16_t cols_of_M[M.max_entries];
    float_t thetas_of_M[M.max_entries];
    uint8_t bit_sign_storage[M.max_entries / 8 +1];

    M.rows = rows_of_M;
    M.cols = cols_of_M;
    M.thetas = thetas_of_M;
    M.bit_sign_storage = bit_sign_storage;

    set_random_weights_sparse_matrix(&M,.5);

    printf("\nPrint list of (theta, weight value):\n");
    int k;
    for(k=0; k<M.number_of_entries; k++)
        printf("(%.2g -> %.2g) \n",M.thetas[k],get_weight_by_entry(&M,k));

    printf("\nPrint weight matrix:\n");
    print_weight_matrix(&M);

    // Test math calculations

    float a[4] = {2,1,3,4};
    float b[4] = {.1,0,.01,-1};
    float dot_result[3];
    float mult_result[4];
    float softmax_result[4];

    printf("\nDot result of weight matrix with the following:\n");
    printf("a: ");
    print_vector(a,NELEMS(a));
    sparse_matrix_vector_dot(&M,a,NELEMS(a),dot_result,NELEMS(dot_result));
    printf("W*a: ");
    print_vector(dot_result,NELEMS(dot_result));

    printf("\nTerm by term multiplication result:\n");
    printf("a: ");
    print_vector(a,NELEMS(a));
    printf("b: ");
    print_vector(b,NELEMS(b));
    printf("a.b : ");
    term_by_term_multiplication(a,NELEMS(a),b,NELEMS(b),mult_result,NELEMS(mult_result));
    print_vector(mult_result,NELEMS(mult_result));

    printf("\nTest of gradient wrt to theta (i,j -> dtheta):\n");
    printf("a:");
    print_vector(a,NELEMS(a));
    printf("c:");
    print_vector(dot_result,NELEMS(dot_result));
    printf("sign_W.(theta_W > 0).(a x c):\n");
    for(k=0; k<M.number_of_entries; k++){
        printf("  %u, %u -> %.2g \n",M.rows[k],M.cols[k],gradient_wrt_theta_entry(&M,dot_result,a,k));
    }


    printf("\nsoftmax:\n");
    printf("a: ");
    print_vector(a,NELEMS(a));
    softmax(a,NELEMS(a),softmax_result,NELEMS(softmax_result));
    printf("softmax(a): ");
    print_vector(softmax_result,NELEMS(softmax_result));



}