#include "math_operations.h"
#include <time.h>

/***
 * Library of math operations required to implement backprop in feedforward relu units with rewiring.
 * The weight matrices are implement as sparse matrices to optimize the speed of memory requirement of the matrices
 * and the matrix multiplications. The additional constraint are that the code only stores variables on the stack
 * (no usage of malloc and calloc).
 * The matrices are stored as a quadruple of arrays with (row, column, weight amplitude (theta), weight sign).
 *
 * Authors:
 *      Guillaume Bellec - TU Graz (sparse matrices and neural network related operations),
 *      Florian Kelber - TU Dresden (bit shift to store the weight signs efficiently)
 *
 * Special thanks to Michael Hopkins for providing the mars_kiss32() random number generator.
 *
 * First creation: 25th of August 2017
 * Latest update: 16th of November 2017
 */

#define L1_COEFF 1e-4
#define NOISE_AMPLITUDE 3e-4
#define SKIP_CHECK true

/**
 * Generate a random integer.
 * Implementation of a Marsaglia 32-bit KISS generator which uses no multiply instructions≈2^121 period
 * 209.9 nanosecs (i.e. 42 ticks) per call (on SpiNNaker 1 ARM core)
 * @return
 */
uint32_t mars_kiss32(void) {
    static uint32_t x = 123456789, y = 234567891, z = 345678912, w = 456789123, c = 0; /* default seed variables */
    int32_t t;

    y ^= (y << 5);
    y ^= (y >> 7);
    y ^= (y << 22);
    t = z + w + c;
    z = w;
    c = (t < 0) ? 1 : 0;
    w = (uint32_t) t & 2147483647;
    x += 1411392427;

    return x + y + w;
}


/**
 * Generate a random int32 integer with uniform distribution.
 * @return
 */
uint32_t rand_int_kiss(uint32_t low, uint32_t high) {
    assert(high > low);
    uint32_t i = mars_kiss32();
    i = i % (high - low);
    return low + i;
}

/**
 * Generate a random number with uniform distribution within 0 and 1.
 * @return
 */
float rand_kiss(void) {
    return (float) (mars_kiss32() / 4294967296.0);
}

/**
 * Generate a random gaussian variable with mean 0 and standard deviation 1.
 * @return
 */
float randn_kiss() {
    static bool
            set = false;
    static float gset;
    float fac, rsq, v1, v2;

    if (!set) {
        do {
            v1 = (mars_kiss32() / 2147483648.0f) - 1.0f;   // U(0,1) * 2 - 1
            v2 = (mars_kiss32() / 2147483648.0f) - 1.0f;
            rsq = v1 * v1 + v2 * v2;
        } while (rsq >= 1.0f || rsq == 0.0f);

        fac = sqrtf(-2.0f * logf(rsq) / rsq);
        gset = v1 * fac;
        set = true;
        return (float) v2 * fac;
    } else {
        set = false;
        return (float) gset;

    }
}


/**
 * Compute the rectified linear function of v in place.
 * @param v
 * @param size_v
 */
void relu_in_place(float *v, uint size_v) {
    uint k;

    for (k = 0; k < size_v; k++) {
        if (v[k] < 0)
            v[k] = 0;
    }
}


/**
 * Print a weight matrix in the python matrix format.
 * @param v
 * @param size
 */
void print_weight_matrix(sparse_weight_matrix *M) {

    int i;
    int j;
    int entry;

    check_sparse_matrix_format(M);

    printf("[");
    for (i = 0; i < M->n_rows; i += 1) {
        if (i > 0)
            printf("\n ");

        printf("[");
        for (j = 0; j < M->n_cols; j += 1) {
            entry = is_entry_and_fetch(M, i, j);
            if (entry == -1)
                printf("_");
            else
                printf("%.2g", get_weight_by_entry(M, entry));
            if (j < M->n_cols - 1)
                printf(", ");

        }
        printf("]");
        if (i < M->n_rows - 1)
            printf(", ");
    }
    printf("] \n");
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
void put_new_entry(sparse_weight_matrix *M, uint16_t row, uint16_t col, float value, bool sign, bool fail_if_exist) {

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
    k = 0;
    while (M->rows[k] < row && k < M->number_of_entries)
        k++;

    while (M->rows[k] == row && M->cols[k] < col && k < M->number_of_entries)
        k++;

    // Make sure that the entry does not exist already.
    if (fail_if_exist)
        assert(M->rows[k] != row || M->cols[k] != col);
    else {
        if (M->rows[k] == row && M->cols[k] == col) {
            return;
        }
    }

    // Push-replace the value in the vector until there is only the last entry left.
    while (k < M->number_of_entries) {
        replaced_row = M->rows[k];
        replaced_col = M->cols[k];
        replaced_theta = M->thetas[k];
        replaced_sign = get_sign(M, k);

        M->rows[k] = pushed_row;
        M->cols[k] = pushed_col;
        M->thetas[k] = pushed_theta;
        set_sign(M, k, pushed_sign);

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
    set_sign(M, k, pushed_sign);

    M->number_of_entries += 1;
    assert(M->number_of_entries < M->max_entries);
    check_sparse_matrix_format(M);
}


uint32_t coord_to_position(sparse_weight_matrix *M, uint32_t row, uint32_t col) {
    uint32_t n_col = M->n_cols;
    return row * n_col + col;
}

uint32_t get_flattened_position(sparse_weight_matrix *M, uint16_t entry) {
    uint32_t row = M->rows[entry];
    uint32_t n_col = M->n_cols;
    uint32_t col = M->cols[entry];
    return row * n_col + col;
}

uint16_t get_row_from_position(sparse_weight_matrix *M, uint32_t position) {
    return position / M->n_cols;
}

uint16_t get_col_from_position(sparse_weight_matrix *M, uint32_t position) {
    return position % M->n_cols;
}

/**
 * Print a weight matrix in the python matrix format.
 * @param v
 * @param size
 */
void print_weight_matrix_containers(sparse_weight_matrix *M, bool print_non_assigned) {

    int entry;

    //check_sparse_matrix_format(M);

    for (entry = 0; entry < M->max_entries; entry++) {
        if (entry == M->number_of_entries) {
            if (print_non_assigned) {
                printf("--\n");
            } else {
                break;
            }
        }
        printf("(%d,%d)->%d \t th: %.2f sign: %d \n", M->rows[entry], M->cols[entry], get_flattened_position(M, entry),
               M->thetas[entry], get_sign(M, entry));
    }
}

/**
 * Raise an error if the element are not sorted from entry_low to entry_high-1 (included)
 * @param M
 * @param entry_low
 * @param entry_high
 */
void check_order(sparse_weight_matrix *M, uint16_t entry_low, uint16_t entry_high) {

    if (SKIP_CHECK) {
        return;
    }

    uint16_t k;
    uint32_t last_pos = -1;
    for (k = entry_low; k < entry_high; k++) {
        if (k > entry_low && get_flattened_position(M, k) <= last_pos) {
            printf("k-1: %d(%d,%d) \t k: %d:(%d,%d) \t k+1: %d(%d,%d)",
                   k - 1, M->rows[k - 1], M->cols[k - 1],
                   k, M->rows[k], M->cols[k],
                   k + 1, M->rows[k + 1], M->cols[k + 1]);
            assert(get_flattened_position(M, k) > last_pos);
        }

        last_pos = get_flattened_position(M, k);
    }
}

void set_position(sparse_weight_matrix *M, uint16_t entry, uint32_t position) {
    uint16_t row = get_row_from_position(M, position);
    uint16_t col = get_col_from_position(M, position);

    assert(row < UINT16_MAX);
    assert(col < UINT16_MAX);
    assert(row <= M->n_rows);
    assert(col < M->n_cols);

    M->rows[entry] = (uint16_t) row;
    M->cols[entry] = (uint16_t) col;
}

/**
 * Swap two elements of the matrix
 * @param M
 * @param entry_a
 * @param entry_b
 */
void swap(sparse_weight_matrix *M, uint16_t entry_a, uint16_t entry_b) {
    if(entry_a == entry_b)
        return;

    uint16_t tmp_row;
    uint16_t tmp_col;
    float tmp_theta;
    bool tmp_sign;

    assert(entry_a < M->number_of_entries);
    assert(entry_b < M->number_of_entries);

    uint32_t n_flattened = M->n_rows * M->n_cols;

    assert(get_flattened_position(M, entry_a) <= n_flattened);
    assert(get_flattened_position(M, entry_b) <= n_flattened);

    tmp_row = M->rows[entry_a];
    tmp_col = M->cols[entry_a];
    tmp_theta = M->thetas[entry_a];
    tmp_sign = get_sign(M, entry_a);

    M->rows[entry_a] = M->rows[entry_b];
    M->cols[entry_a] = M->cols[entry_b];
    M->thetas[entry_a] = M->thetas[entry_b];
    set_sign(M, entry_a, get_sign(M, entry_b));

    M->rows[entry_b] = tmp_row;
    M->cols[entry_b] = tmp_col;
    M->thetas[entry_b] = tmp_theta;
    set_sign(M, entry_b, tmp_sign);

}

/* This function takes last element as pivot, places
   the pivot element at its correct position in sorted
    array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right
   of pivot */
uint16_t partition(sparse_weight_matrix *M, uint16_t entry_low, uint16_t entry_high) {
    uint32_t pivot = get_flattened_position(M, entry_high);    // pivot
    uint16_t i = (entry_low - 1);  // Index of smaller element

    for (uint16_t j = entry_low; j <= entry_high - 1; j++) {
        // If current element is smaller than or
        // equal to pivot
        if (get_flattened_position(M, j) <= pivot) {
            i++;    // increment index of smaller element
            swap(M, i, j);
        }
    }

    swap(M, i + 1, entry_high);

    return (i + 1);
}

/**
 * Sort the sparse M from index low to index high using quickSort
 * @param M
 * @param entry_low
 * @param entry_high
 */
void quickSort(sparse_weight_matrix *M, uint16_t entry_low, uint16_t entry_high) {

    if (entry_low < entry_high) {
        /* pi is partitioning index, arr[p] is now at right place */
        uint16_t pi = partition(M, entry_low, entry_high);

        // Separately sort elements before
        // partition and after partition
        if (pi > 0)
            quickSort(M, entry_low, pi - 1);
        quickSort(M, pi + 1, entry_high);
    }
}


/**
 * Usage of a function to move the randomly added entries to avoid having doubles.
 *
 *
 * @param M
 * @param k
 * @param k_to_insert
 * @param new_n_entries
 * @return
 */
void ignore_or_slide_copied_specific_position(sparse_weight_matrix *M, uint16_t k, uint16_t k_to_insert) {
    //assert(k_to_insert == k + 1);

    uint32_t previous_pos = get_flattened_position(M, k);
    uint32_t pos_i = get_flattened_position(M, k_to_insert);
    uint32_t n_flattened = M->n_rows * M->n_cols;

    uint16_t ignore_it = 0;
    uint16_t i = k_to_insert;

    // There is a need of a loop in case many elements are following each other.
    // Then we will push all of them by a position of 1 until it fits into the sparse array or until we have to delete elements.
    while (pos_i < n_flattened && previous_pos == pos_i) {
        M->rows[i] = get_row_from_position(M, pos_i + 1);
        M->cols[i] = get_col_from_position(M, pos_i + 1);

        previous_pos = get_flattened_position(M, i);

        if (previous_pos == n_flattened)
            ignore_it = 1;

        if (i + 1 < M->number_of_entries)
            i++;
        else
            break;

        pos_i = get_flattened_position(M, i);
    }

    M->number_of_entries -= ignore_it;

}


/**
 * Sort algorithm that is taking an array that is sorted on both sides of an element.
 *
 * It also assumes that the first sorted part is much larger than the second one.
 * @param M
 * @param split_index
 */
void sort_concatenation_of_two_sorted_arrays(sparse_weight_matrix *M, uint16_t split_index) {

    uint16_t k_left = 0; // first index of the left array
    uint16_t k_right = split_index; // first index of the right array

    // Define the position (ordered value) with high precision on temporary variables
    uint32_t k_right_position = get_flattened_position(M, k_right);
    uint32_t k_left_position = get_flattened_position(M, k_left);

    // Allocate variables needed to swap place a left member into the right array
    uint16_t j;
    uint32_t position_j;
    uint32_t position_j_next;

    while (k_left < k_right && k_right < M->number_of_entries) {

        // The loop condition to start the loop should be verified
        check_order(M, k_left, k_right);
        check_order(M, k_right, M->number_of_entries);

        // Move the k_left index to point to the index where we can insert k_right
        while (k_left < k_right && k_left_position <= k_right_position) {

            if (k_right_position == k_left_position) { // This special case is important to ensure that k is at least 1
                ignore_or_slide_copied_specific_position(M, k_left,k_right); // Move the position of split_index if equality
                k_right_position = get_flattened_position(M,k_right); // Fetch the new position, if split_index was thrown out of the array its position is equal to the array size and the alogirhtm will stop.
                assert(k_right_position > k_left_position);
            } else {
                k_left += 1;
                k_left_position = get_flattened_position(M, k_left);  // Increase k
            }

        }

        // Insert the new element in the left array
        swap(M, k_left, k_right);
        k_left_position = get_flattened_position(M, k_left);
        k_right_position = get_flattened_position(M, k_right);

        check_order(M, 0, k_right); // check that the left array is sorted

        // Swap until the left right is sorted
        j = k_right;
        position_j = get_flattened_position(M, j);
        position_j_next = get_flattened_position(M, j + 1);

        // Push the old element in the right array, until the right array end up being sorted
        while (j < M->number_of_entries - 1
               && position_j >= position_j_next) {

            if (position_j == position_j_next) {
                // equality has to be considered carefully
                ignore_or_slide_copied_specific_position(M, j, j + 1);
            } else {
                swap(M, j, j + 1);
                j++;
            }

            position_j = get_flattened_position(M, j);
            position_j_next = get_flattened_position(M, j + 1);

        }
        check_order(M, k_right, M->number_of_entries); // check the the right array is sorted
    }

    check_order(M, 0, M->number_of_entries); // Every thing is sorted
}

/**
 * Usage of a function to move the randomly added entries to avoid having doubles
 * @param M
 * @param k
 * @param k_to_insert
 * @param number_of_entries
 * @return
 */
void slide_or_ignore_all_doubles(sparse_weight_matrix *M, uint16_t from_k) {
    uint16_t k;
    uint16_t next_k;
    uint32_t position_k;
    uint32_t position_next_k;

    for (k = M->number_of_entries - 2; k >= from_k - 1; k--) {
        position_k = get_flattened_position(M, k);

        next_k = k + 1;
        position_next_k = get_flattened_position(M, next_k);

        // We need a while because there might be repetitions
        while (position_k == position_next_k) {

            ignore_or_slide_copied_specific_position(M, k, next_k);

            position_k = get_flattened_position(M, k);
            position_next_k = get_flattened_position(M, next_k);
        }
    }

}


void put_new_random_entries(sparse_weight_matrix *M, uint16_t n_new) {

    if (n_new == 0) {
        return;
    }

    uint32_t n_flattened = M->n_rows * M->n_cols;
    uint16_t old_n_entries = M->number_of_entries;
    M->number_of_entries = old_n_entries + n_new;


    uint16_t k;

    // tmp variables necessary to generate the new positions
    uint32_t new_position;

    // Generate the new ordered entries and append them into the matrix
    for (k = 0; k < n_new; k++) {
        new_position = rand_int_kiss(old_n_entries, n_flattened - 1);
        M->rows[old_n_entries + k] = get_row_from_position(M, new_position);
        M->cols[old_n_entries + k] = get_col_from_position(M, new_position);
        M->thetas[old_n_entries + k] = 0;
        set_sign(M, old_n_entries + k, rand_kiss() > 0.5);
    }

    // Sort the appended values to accelerate the insertion inside the array
    quickSort(M, old_n_entries, M->number_of_entries - 1);

    // Make sure there is no equality in the numbers that are the appended section of the array.
    //
    // For all pair of consecutive entries that are equal in the new elements do:
    // - Put left or right the positions to avoid overlap
    // - If non of them is possible, push to the end and ignore this new entry

    slide_or_ignore_all_doubles(M, old_n_entries);

    // The algorithm now takes two separate part of the array: early part (old elements) and later part (new elements)
    // Those two parts are respectively ordered and we need to order them in a single array. One may want to use quick
    // sort here again but I think it more efficient to do as follow:
    //
    // Hold an index k in the early part and an index new_k in the later parts.
    // Everything below k in the early part should correspond to the single array that will be sorted at the end.
    // We proceed to the loop:

    sort_concatenation_of_two_sorted_arrays(M, old_n_entries);

    assert(M->number_of_entries <= M->max_entries);
    check_sparse_matrix_format(M);
}

/**
 * Delete an entry in the matrix.
 * @param M
 * @param entry_idx
 */
void delete_entry(sparse_weight_matrix *M, uint16_t entry_idx) {

    uint16_t k;
    assert(entry_idx < M->number_of_entries);

    for (k = entry_idx; k < M->number_of_entries - 1; k++) {
        M->rows[k] = M->rows[k + 1];
        M->cols[k] = M->cols[k + 1];
        M->thetas[k] = M->thetas[k + 1];
        set_sign(M, k, get_sign(M, k + 1));
    }

    M->number_of_entries -= 1;

}

/**
 * Delete all negative entries
 * @param M
 * @param entry_idx
 */
void delete_negative_entries(sparse_weight_matrix *M) {

    uint16_t k;
    uint16_t n_negative = 0;
    uint32_t n_flattened = M->n_cols * M->n_rows;

    for (k = 0; k < M->number_of_entries; k++) {

        if (M->thetas[k] < 0) {
            M->rows[k - n_negative] = get_row_from_position(M, n_flattened);
            M->cols[k - n_negative] = get_col_from_position(M, n_flattened);
            n_negative += 1;
        } else if (n_negative > 0) {
            swap(M, k, k - n_negative);

        }
    }

    M->number_of_entries -= n_negative;
    check_sparse_matrix_format(M);
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
void vector_substraction(float *a, uint size_a, float *b, uint size_b, float *result, uint size_result) {
    assert(size_a == size_b);
    assert(size_a == size_result);

    uint k;

    for (k = 0; k < size_a; k++) {
        result[k] = a[k] - b[k];
    }

}

/**
 * Set the dimensions of the matrix and defines the maxium number of entries.
 * @param M
 * @param n_rows
 * @param n_cols
 */
void set_dimensions(sparse_weight_matrix *M, uint16_t n_rows, uint16_t n_cols, float connectivity) {
    M->n_rows = n_rows;
    M->n_cols = n_cols;
    M->max_entries = (uint16_t) ceilf((uint16_t) n_cols * (uint16_t) n_rows *
                                      connectivity); //ceil((float)M->n_rows * (float)M->n_cols * connectivity);
}

/**
 * Get the sign of the k-th entry.
 * @param M
 * @param entry_idx
 * @return
 */
bool get_sign(sparse_weight_matrix *M, uint16_t entry_idx) {
    uint8_t sign = M->bit_sign_storage[entry_idx / 8];
    sign = (uint8_t) ((sign >> (entry_idx % 8)) & 0x1);

    return (bool) sign;
}

/**
 * Set the sign of the k-th entry in the matrix.
 * @param M
 * @param entry_idx
 * @param val
 */
void set_sign(sparse_weight_matrix *M, uint16_t entry_idx, bool val) {
    if (val) M->bit_sign_storage[entry_idx / 8] |= 1 << (entry_idx % 8);
    else M->bit_sign_storage[entry_idx / 8] &= ~(1 << (entry_idx % 8));
}

/**
 * Find the weight of the k-th recorded entry.
 * @param M
 * @param k
 * @return
 */
float get_weight_by_entry(sparse_weight_matrix *M, int k) {
    bool s;
    assert(k < M->number_of_entries);

    if (M->thetas[k] <= 0)
        return 0.;
    else {
        s = get_sign(M, k);
        if (s)
            return M->thetas[k];
        else
            return -M->thetas[k];
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
int is_entry_and_fetch(sparse_weight_matrix *M, int i, int j) {
    int entry = -1;
    int k;

    for (k = 0; k < M->number_of_entries; k++) {
        if (M->rows[k] == i && M->cols[k] == j)
            entry = k;
    }

    return entry;

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
    for (k = 0; k < M->number_of_entries; k++) {
        if (M->rows[k] == i && M->cols[k] == j)
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
    for (k = 0; k < M->number_of_entries; k++) {
        if (M->rows[k] == i && M->cols[k] == j)
            if (get_sign(M, k))
                s = 1;
            else
                s = -1;
    }

    return s;

}


/**
 * Fill-in a sparse matrix with random value.
 * It uses a Xavier initialization of the form: mask * randn(n_in,n_out)/n_in,
 * if randn generates gaussian number, n_in is the number of inputs, n_out is the number of outputs.
 * The mask allows non zero values at each position with probability "connectivity": mask = rand(n_in,n_out) < connectivity
 * @param M
 * @param connectivity
 */
void set_random_weights_sparse_matrix(sparse_weight_matrix *M, float connectivity) {

    uint16_t k = 0;
    uint32_t pos = 0;
    uint32_t n_flattened = M->n_rows * M->n_cols;
    float value;

    while (pos < n_flattened && k < M->max_entries) {

        if (rand_kiss() < connectivity) {
            value = (randn_kiss() / sqrtf(M->n_rows));

            set_position(M, k, pos);
            M->thetas[k] = fabsf(value);

            if (rand_kiss() > 0.5)
                set_sign(M, k, true);
            else
                set_sign(M, k, false);

            k += 1;

            M->number_of_entries = k;
            assert(k <= M->max_entries);
        }
        pos++;

    }

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
    for (i = 0; i < M->n_rows; i += 1) {
        if (i > 0)
            printf("\n");

        printf("[");
        for (j = 0; j < M->n_cols; j += 1) {
            printf("%d", get_sign_by_row_col_pair(M, i, j));

            if (j < M->n_cols - 1)
                printf(", ");

        }
        printf("]");
        if (i < M->n_rows - 1)
            printf(", ");
    }
    printf("] \n");

    printf("\nthetas:\n [");
    for (i = 0; i < M->n_rows; i += 1) {
        if (i > 0)
            printf("\n");

        printf("[");
        for (j = 0; j < M->n_cols; j += 1) {
            printf("%.2g", get_theta_by_row_col_pair(M, i, j));

            if (j < M->n_cols - 1)
                printf(", ");

        }
        printf("]");
        if (i < M->n_rows - 1)
            printf(", ");
    }
    printf("] \n");
}

/**
 * Print an array of floats a vector in the python format.
 * @param v
 * @param size
 */
void print_vector(float *v, uint size) {
    uint k;

    printf("[");
    for (k = 0; k < size; k++) {
        printf("%.2g", v[k]);
        if (k < size - 1)
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
void left_dot(sparse_weight_matrix *M, float *v, uint size_v, float *result, uint size_result) {

    int i;
    int j;
    int k;

    float w;

    check_sparse_matrix_format(M);
    assert(size_v == M->n_cols);
    assert(size_result == M->n_rows);

    for (i = 0; i < size_result; i++)
        result[i] = 0;

    for (k = 0; k < M->number_of_entries; k++) {
        i = M->rows[k];
        j = M->cols[k];
        w = get_weight_by_entry(M, k);

        if (v[j] != 0)
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

    for (j = 0; j < size_result; j++)
        result[j] = 0;

    for (k = 0; k < M->number_of_entries; k++) {
        i = M->rows[k];
        j = M->cols[k];
        w = get_weight_by_entry(M, k);

        if (v[i] != 0)
            result[j] += w * v[i];
    }
}

/**
 * Check that the index of a matrix are well ordered.
 * @param M
 */
void check_sparse_matrix_format(sparse_weight_matrix *M) {
    if (SKIP_CHECK) {
        return;
    }

    int k;
    int last_i;
    int last_j;

    last_i = -1;
    last_j = -1;

    for (k = 0; k < M->number_of_entries; k++) {
        if (k > 0) {
            assert(get_flattened_position(M, k) > coord_to_position(M, last_i, last_j));
        }


        assert((int) M->rows[k] >= last_i);

        if ((int) M->rows[k] == last_i)
            assert((int) M->cols[k] > last_j);

        last_i = M->rows[k];
        last_j = M->cols[k];

        //assert(M->thetas > 0); //Thetas has to be positive if we do not simulate disconnected synapses

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
float gradient_wrt_theta_entry(sparse_weight_matrix *weight_matrix, float *a_pre, uint size_a_pre, float *d_post,
                               uint size_d_post, uint16_t entry_idx) {

    float theta = weight_matrix->thetas[entry_idx];
    if (theta < 0) {
        return 0;
    }


    bool sign = get_sign(weight_matrix, entry_idx);
    int i = weight_matrix->rows[entry_idx];
    int j = weight_matrix->cols[entry_idx];
    float de_dtheta;

    assert(size_a_pre > i);
    assert(size_d_post > j);

    if (a_pre[i] != 0 && d_post[j] != 0)
        de_dtheta = a_pre[i] * d_post[j];
    else
        de_dtheta = 0;

    if (sign)
        return de_dtheta;
    else
        return -de_dtheta;
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
                         uint size_d_post, float *d_pre, uint size_d_pre) {

    int i;
    int j;
    int k;

    float w;

    check_sparse_matrix_format(W);
    assert(size_d_post == W->n_cols);
    assert(size_d_pre == W->n_rows);
    assert(size_a_pre == W->n_rows);

    for (i = 0; i < size_d_pre; i++)
        d_pre[i] = 0;

    for (k = 0; k < W->number_of_entries; k++) {
        i = W->rows[k];
        j = W->cols[k];
        w = get_weight_by_entry(W, k);

        if (d_post[j] != 0 && a_pre[i] > 0) // Dot the term-by-term product directly with the derivative of a_pre
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
void term_by_term_multiplication(float *a, uint size_a, float *b, uint size_b, float *result, uint size_result) {

    uint k;
    assert(size_a == size_b);
    assert(size_a == size_result);

    for (k = 0; k < size_a; k++)
        if (a[k] != 0 && b[k] != 0)
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
void softmax(float *a, uint size_a, float *result, uint size_result) {

    float a_max;
    float sum;
    uint k;

    assert(size_a == size_result);

    a_max = a[0];
    for (k = 0; k < size_a; k++) {
        if (a_max < a[k])
            a_max = a[k];
    }

    sum = 0;
    for (k = 0; k < size_a; k++) {
        result[k] = expf(a[k] - a_max);
        sum += result[k];
    }

    for (k = 0; k < size_a; k++)
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
void update_weight_matrix(sparse_weight_matrix *W, float *a_pre, uint size_a_pre, float *d_post, uint size_d_post,
                          float learning_rate) {

    uint16_t k;
    float grad;
    float dtheta;

    for (k = 0; k < W->number_of_entries; k++) {
        grad = gradient_wrt_theta_entry(W, a_pre, size_a_pre, d_post, size_d_post, k);
        dtheta = learning_rate * (-L1_COEFF - grad + NOISE_AMPLITUDE * randn_kiss());
        W->thetas[k] += dtheta;
    }

}

void rewiring(sparse_weight_matrix *W) {
    delete_negative_entries(W);
    put_new_random_entries(W, W->max_entries - W->number_of_entries);
}

/**
 * Argmax function, compute the index of the maximum element in the vector
 *
 * @param prob
 * @param size_prob
 */
uint8_t argmax(float *prob, uint8_t size_prob) {
    uint8_t max_index = 0;
    float max = *prob;

    for (uint8_t i = 1; i < size_prob; i++) {
        if (*(prob + i) > max) {
            max = *(prob + i);
            max_index = i;
        }
    }

    return max_index;
}


/**
 *ensure any 2 entries don't have the same position. If there exists, slide one to the next position
 *If not possible, delele this entry
 *The precondition to use this function is that the matrix has been sorted
 * @param M
 */
void eliminate_same_position(sparse_weight_matrix *M){
    uint16_t k;
    uint32_t position_k, position_k_pre;
    uint32_t n_flattened = M->n_rows * M->n_cols;
    bool old_entry_k, no_slide;

    //it is possible that several occupay the same position, the for loop only one, so we need the while loop
    //only when no slide happens during the for loop, the while loop can be jumped out.
    do {
        no_slide = true;
        for (k = M->number_of_entries - 1; k > 0; k--){
            position_k     = get_flattened_position(M, k);
            position_k_pre = get_flattened_position(M, k - 1);

            //from end to beginning, compare the position of 2 adjacent entries.
            // If k is the old entry, k - 1 must be new rewired, we slide k - 1 to the next position and exchange k - 1 and k to keep the order
            // If k is the new entry, we slide k to the next position.
            if (position_k == position_k_pre) {
                old_entry_k    = (M->thetas[k] == 0)?false:true;    //according to thetas to differentiate the old and new entry
                no_slide = false;
                if (old_entry_k){
                    M->rows[k - 1]   = get_row_from_position(M, position_k + 1);
                    M->cols[k - 1]   = get_col_from_position(M, position_k + 1);
                    swap(M, k - 1, k);

                } else {
                    M->rows[k]   = get_row_from_position(M, position_k + 1);
                    M->cols[k]   = get_col_from_position(M, position_k + 1);

                }
                //If slided position beyonds scope, delete this entry
                if (position_k + 1 == n_flattened)
                    M->number_of_entries -= 1;
            }
        }
    }while(!no_slide);
}


/**
 * rewiring 2.0 function, directly generate new entries to replace the negative entries, sort the order and eliminate the entries with same positions
 *
 * @param M
 */
void rewiring2(sparse_weight_matrix *M) {
    uint16_t k;
    uint16_t k_middle = M->number_of_entries - 1;
    uint32_t new_position;
    uint32_t n_flattened = M->n_rows * M->n_cols;


    // Generate new entries to replace the entries with negative weights
    for (k = 0; k < M->number_of_entries; k++) {
        if (M->thetas[k] <= 0) {
            new_position = rand_int_kiss(0, n_flattened);
            assert(new_position < n_flattened);
            M->rows[k]   = get_row_from_position(M, new_position);
            M->cols[k]   = get_col_from_position(M, new_position);
            M->thetas[k] = 0;
            set_sign(M, k, rand_kiss() > 0.5);
        }
        //record the entry number of one element, whose position generally in the middle of the matrix space
        if(M->rows[k] == M->n_rows / 2 - 1 || M->rows[k] == M->n_rows / 2)
            k_middle = k;
    }

    //swap the middle-position element with the last entry, so that the first pivot picked up by quick sort holds the middle position
    //the purpose is to split the matrix into 2 partitions with same size. In this case quick sort requires the minimum memory.
    swap(M, k_middle, M->number_of_entries - 1);

    //sort the matrix, elements with same position are allowed
    quickSort(M, 0, M->number_of_entries - 1);

    //eliminate elements with same position
    eliminate_same_position(M);
}


