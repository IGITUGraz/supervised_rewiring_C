#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <expat.h>
#include <string.h>



#include "math_operations.h"

#include "test_network_gradient.h"
#include "test_math.h"
#include "train_network.h"

int main (int argc, char **argv) {

    printf(getenv("PROJ_GRAZ_PATH"));
    if (argc>1 && strcmp(argv[1], "test_math") == 0){
        return test_math();
    }
    else if (argc>1 && strcmp(argv[1], "test_network") == 0){
        return test_network();
    }
    else
        return train_network();
}