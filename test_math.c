#include "test_math.h"

int test_math() {
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

    printf("\nAdding a new entry:\n");
    put_new_entry(&M,1,2,0.1,true,true);
    print_weight_matrix(&M);

    printf("\nDelete an entry:\n");
    delete_entry(&M,1);
    print_weight_matrix(&M);

    // Test math calculations

    float a[4] = {2,1,-1,3};
    float b[4] = {.1,0,.01,-1};
    float c[3] = {.1,0,-0.1};
    float left_dot_result[3];
    float right_dot_result[4];
    float mult_result[4];
    float softmax_result[4];

    printf("\nLeft-Dot result of weight matrix with the following:\n");
    printf("a: ");
    print_vector(a,NELEMS(a));
    left_dot(&M, a, NELEMS(a), left_dot_result, NELEMS(left_dot_result));
    printf("W*a: ");
    print_vector(left_dot_result,NELEMS(left_dot_result));


    printf("\nRight-Dot result of weight matrix with the following:\n");
    printf("c: ");
    print_vector(c,NELEMS(c));
    right_dot(c, NELEMS(c), &M, right_dot_result, NELEMS(right_dot_result));
    printf("c*W: ");
    print_vector(right_dot_result,NELEMS(right_dot_result));

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
    print_vector(c,NELEMS(c));
    printf("sign_W.(theta_W > 0).(a x c):\n");
    for(k=0; k<M.number_of_entries; k++){
        printf("  %u, %u -> %.2g \n",M.rows[k],M.cols[k],gradient_wrt_theta_entry(&M,c,NELEMS(c),a,NELEMS(a),k));
    }

    printf("\nBackprop error msg:\n");
    printf("d_post:");
    print_vector(a,NELEMS(a));
    printf("a_pre:");
    print_vector(c,NELEMS(c));
    printf("a' . d_post*W:\n");
    back_prop_error_msg(&M,c,NELEMS(c), a, NELEMS(a), left_dot_result, NELEMS(left_dot_result));
    print_vector(left_dot_result,NELEMS(left_dot_result));


    printf("\nTest of the softmax:\n");
    printf("a: ");
    print_vector(a,NELEMS(a));
    softmax(a,NELEMS(a),softmax_result,NELEMS(softmax_result));
    printf("softmax(a): ");
    print_vector(softmax_result,NELEMS(softmax_result));

    printf("\nTest of the argmax:\n");
    printf("softmax_result: ");
    print_vector(softmax_result,NELEMS(softmax_result));
    uint8_t max_index = argmax(softmax_result,NELEMS(softmax_result));
    printf("argmax(softmax_result): %d\n", max_index);

    printf("\nTest put random entries (we add a couple of entries for to try complex cases)\n");
    put_new_entry(&M,0,3,0.1,true,true);
    put_new_entry(&M,1,0,-0.1,true,true);
    put_new_entry(&M,0,0,9,false,true);
    put_new_entry(&M,2,2,9,false,true);
    put_new_entry(&M,2,3,9,false,true);
    printf("Old matrix: \n");
    print_weight_matrix(&M);
    printf("New matrix with k new zeros: \n");
    put_new_random_entries(&M,1);
    print_weight_matrix(&M);

    printf("\nTest delete random entries (Should delete the zeros... )\n");
    delete_negative_entries(&M);
    print_weight_matrix(&M);




    return 0;
}