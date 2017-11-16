#include <jmorecfg.h>
#include "math_operations.h"
#include <time.h>

/***
 * Library of math opeartions required to implement backprop in feedforward relu units with rewiring.
 * The weights are very sparse therefore it uses a minimal amounf of memory.
 *
 * Authors: Guillaume Bellec
 * Date: 25th of August 2017
 */

// a constant definition exported by library:
//#define LEARNING_RATE 0.005
#define L1_COEFF 0.0001
#define NOISE_AMPLITUDE 0.000001
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

    check_sparse_matrix_format(M);

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
        quickSort(M, entry_low, pi - 1);
        quickSort(M, pi + 1, entry_high);
    }
}


/**
 * Usage of a function to move the randomly added entries to avoid having doubles
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

    while (pos_i < n_flattened && previous_pos == pos_i) {

        M->rows[i] = get_row_from_position(M, pos_i + 1);
        M->cols[i] = get_col_from_position(M, pos_i + 1);

        previous_pos = get_flattened_position(M, i);

        if (previous_pos == n_flattened)
            ignore_it = 1;

        i++;
        pos_i = get_flattened_position(M, i);
    }

    M->number_of_entries -= ignore_it;

}

/**
 * Sort algorithm that is taking an array that is sorted on both sides of an element somewhere.
 * It also assumes that the first sorted part is much larger than the second one.
 * @param M
 * @param first_index
 * @param split_index
 * @param last_index
 */
void sort_partly_sorted(sparse_weight_matrix *M, uint16_t first_index, uint16_t split_index) {

    if (first_index == split_index || split_index == M->number_of_entries) {
        return;
    }

    uint16_t k = first_index;
    uint16_t swap_count;

    check_order(M,first_index,split_index);
    check_order(M,split_index,M->number_of_entries);

    {
        uint32_t split_position = get_flattened_position(M, split_index);
        uint32_t position_k = get_flattened_position(M, k);

        while (k < split_index && position_k <= split_position) {

            if (split_position == position_k) {
                ignore_or_slide_copied_specific_position(M, k, split_index);
                split_position = get_flattened_position(M, split_index);
                assert(split_position > position_k);
            }
            else{
                k++;
                position_k = get_flattened_position(M, k);
            }

        }

        swap_count = 0;
        while (split_index + swap_count < M->number_of_entries
               && k + swap_count < split_index
               && get_flattened_position(M, split_index + swap_count) < position_k) {
            swap_count++;
        }
    }

    for (uint16_t j = 0; j < swap_count; j++)
        swap(M, k + j, split_index + j);
    k += swap_count;

    sort_partly_sorted(M, split_index, split_index + swap_count);
    sort_partly_sorted(M, k, split_index);
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
    uint16_t n_ignored = 0;

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

    sort_partly_sorted(M, 0, old_n_entries);

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
        if (M->thetas[k] <= 0) {
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
    M->max_entries = ceil((uint16_t) n_cols * (uint16_t) n_rows *
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
            value = (randn_kiss() / (float) sqrt(M->n_rows));

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
        result[k] = exp(a[k] - a_max);
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
    put_new_random_entries(W,W->max_entries - W->number_of_entries);
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