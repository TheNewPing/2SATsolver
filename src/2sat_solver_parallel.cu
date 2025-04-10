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
*/
__global__ void kernel_solve_2SAT(int n_comp, int n_sol, int n_vars, int* candidates, bool* infl_comp, int* comp, bool* sol_comp) {
    __shared__ int curr_comp;
    __shared__ bool val_i;
    __shared__ bool* infl_i;
    __shared__ bool* sol_i;
    for (int i = 0; i < n_sol; i += gridDim.x) {
        int curr_sol = blockIdx.x + i;
        if (curr_sol >= n_sol) return;
        for (int j = n_comp-1; j >= 0; --j) {
            // load in shared mem the i-th component of the current candidate solution
            if (threadIdx.x == 0) {
                curr_comp = candidates[curr_sol * n_comp + j];
                sol_i = sol_comp + (curr_sol * n_comp);
                infl_i = infl_comp + (curr_comp * n_comp);
                val_i = sol_i[curr_comp];
            }
            __syncthreads();
            // propagate the effect of the i-th component to all other components
            for (int k = 0; k < n_comp; k += blockDim.x) {
                int comp_idx = threadIdx.x + k;
                if (comp_idx >= n_comp) return;
                if (comp_idx != curr_comp) {
                    bool infl_j = infl_i[comp_idx];
                    sol_i[comp_idx] = (infl_j && !val_i) || (!infl_j && sol_i[comp_idx]);
                }
            }
            __syncthreads();
        }
    }
}

__global__ void kernel_comp_to_var(int n_comp, int n_vars, int n_sol, int* comp, bool* sol_comp, bool* sol_var) {
    __shared__ int var_comp;

    for (int i = 0; i < n_vars; i += gridDim.x) {
        int curr_var = blockIdx.x + i;
        if (curr_var >= n_vars) return;
        // load in shared mem the component of the current variable
        if (threadIdx.x == 0) {
            var_comp = comp[curr_var];
        }
        __syncthreads();
        for (int j = 0; j < n_sol; j += blockDim.x) {
            int curr_sol = threadIdx.x + j;
            if (curr_sol >= n_sol) return;
            sol_var[curr_sol * n_vars + curr_var] = sol_comp[curr_sol * n_comp + var_comp];
        }
        __syncthreads();
    }
}

__global__ void kernel_filter_min_dist(int n_sol, int n_vars, bool* sol_var, bool* sol_var_min_dist, int min_dist) {
    __shared__ int counter;
    
    for (int i = 0; i < n_sol * n_sol; i += gridDim.x) {
        int curr_sol_row = (blockIdx.x + i) / n_sol;
        if (curr_sol_row >= n_sol) return;
        int curr_sol_col = (blockIdx.x + i) % n_sol;
        if (curr_sol_col >= n_sol) return; // cannot happen

        if (curr_sol_row < curr_sol_col) return; // skip symmetric pairs
        if (curr_sol_row == curr_sol_col) {
            // a solution is always at distance 0 from itself, but setting it to true is required for the next kernel
            sol_var_min_dist[curr_sol_row * n_sol + curr_sol_col] = true;
            return;
        }

        if (threadIdx.x == 0) {
            counter = 0;
        }
        __syncthreads();

        for (int j = 0; j < n_vars; j += blockDim.x) {
            int curr_var = threadIdx.x + j;
            if (curr_var >= n_vars) return;
            
            if (sol_var[curr_sol_row * n_vars + curr_var] != sol_var[curr_sol_col * n_vars + curr_var]) {
                atomicAdd(&counter, 1); // should be at block level, but this version is compatible with all architectures
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            sol_var_min_dist[curr_sol_row * n_sol + curr_sol_col] = counter >= min_dist;
            sol_var_min_dist[curr_sol_col * n_sol + curr_sol_row] = counter >= min_dist;
        }
        __syncthreads();
    }
}
