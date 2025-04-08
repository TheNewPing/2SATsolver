#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>

#include "../include/literal.cu"
#include "../include/cuda_error.cu"

__device__ __host__ void print_array(int* array, int length, int values_per_row) {
    for (int i = 0; i < length; ++i) {
        printf("%d ", array[i]);
        if ((i + 1) % values_per_row == 0) {
            printf("\n");
        }
    }
    if (length % values_per_row != 0) {
        printf("\n");
    }
}
__device__ __host__ void print_array(bool* array, int length, int values_per_row) {
    for (int i = 0; i < length; ++i) {
        printf("%d ", array[i]);
        if ((i + 1) % values_per_row == 0) {
            printf("\n");
        }
    }
    if (length % values_per_row != 0) {
        printf("\n");
    }
}

/*
Args:
    n_comp: number of components.
    n_sol: number of candidate solutions.
    n_vars: number of variables.
    candidates: matrix of candidate solutions.
        each row contains a valid permutation of the sccs.
    infl_comp: matrix of influence between components.
        each row contains the influence of the i-th component due to negated variables.
    comp: vertex component mapping.
    sol_comp: matrix of solutions on components.
    sol_var: matrix of solutions on variables.
*/
__global__ void kernel_solve_2SAT(int n_comp, int n_sol, int n_vars, int* candidates, bool* infl_comp, int* comp, bool* sol_comp, bool* sol_var) {
    if (blockIdx.x >= n_sol) return;
    if (threadIdx.x >= n_comp) return;

    __shared__ int curr_comp;
    __shared__ bool val_i;
    __shared__ bool* infl_i;
    __shared__ bool* sol_i;
    for (int i = n_comp-1; i >= 0; --i) {
        // load in shared mem the i-th component of the current candidate solution
        if (threadIdx.x == 0) {
            curr_comp = candidates[blockIdx.x * n_comp + i];
            val_i = sol_comp[blockIdx.x * n_comp + curr_comp];
            infl_i = infl_comp + (curr_comp * n_comp);
            sol_i = sol_comp + (blockIdx.x * n_comp);
        }
        __syncthreads();
        // propagate the effect of the i-th component to all other components
        for (int j = 0; j < n_comp; j += blockDim.x) {
            if (j + threadIdx.x < n_comp && j + threadIdx.x != curr_comp) {
                bool infl_j = infl_i[j + threadIdx.x];
                sol_i[j + threadIdx.x] = (infl_j && !val_i) || (!infl_j && sol_i[j + threadIdx.x]);
            }
        }
    }
}