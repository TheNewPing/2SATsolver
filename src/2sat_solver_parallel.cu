#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <numeric>

#include "../include/literal.cu"
#include "../include/cuda_error.cu"
#include "../include/cuda_utilities.cu"

__device__ __host__ void print_array(int* array, int length, int values_per_row, const char* row_prefix) {
    for (int i = 0; i < length; ++i) {
        if (i % values_per_row == 0) {
            printf("%s", row_prefix);
        }
        printf("%d ", array[i]);
        if ((i + 1) % values_per_row == 0) {
            printf("\n");
        }
    }
    if (length % values_per_row != 0) {
        printf("\n");
    }
}
__device__ __host__ void print_array(bool* array, int length, int values_per_row, const char* row_prefix = "") {
    for (int i = 0; i < length; ++i) {
        if (i % values_per_row == 0) {
            printf("%s", row_prefix);
        }
        printf("%d ", array[i]);
        if ((i + 1) % values_per_row == 0) {
            printf("\n");
        }
    }
    if (length % values_per_row != 0) {
        printf("\n");
    }
}

__device__ __host__ void print_array(int* array, int num_rows, int* row_end_indices, const char* row_prefix = "") {
    int start_index = 0;
    for (int row = 0; row < num_rows; ++row) {
        printf("%s", row_prefix);
        for (int index = start_index; index < row_end_indices[row]; ++index) {
            printf("%d ", array[index]);
        }
        printf("\n");
        start_index = row_end_indices[row];
    }
}

/*
Runs 1 block per candidate solution.
Each block loads the value of the j-th component of the candidate solution and its influence on the other components.
Then, it propagates the influence of the j-th component to all other components.
Repeats for all components of the candidate solution, starting from the last one.

Args:
    n_comp: number of components.
    n_sol: number of candidate solutions.
    n_vars: number of variables.
    candidates: matrix of candidate solutions.
        each row contains a valid permutation of the sccs.
    infl_comp: matrix of influence between components.
        each row contains the influence of the i-th component due to negated variables.
    comp: vertex component mapping.
    sol_comp: matrix of solutions on components, must be initialized with all values = true,
        it will contain the results.
*/
__global__ void kernel_solve_2SAT(int n_comp, int n_sol, int n_vars, int* candidates, int* comp, int* infl_comp, int* infl_comp_end_idx, bool* sol_comp) {
    __shared__ int curr_comp;
    __shared__ bool val_i;
    __shared__ int* infl_i;
    __shared__ int infl_i_size;
    __shared__ bool* sol_i;
    for (int i = 0; i < n_sol; i += gridDim.x) {
        int curr_sol = blockIdx.x + i;
        if (curr_sol >= n_sol) return;
        for (int j = n_comp-1; j >= 0; --j) {
            // load in shared mem the j-th component of the current candidate solution
            if (threadIdx.x == 0) {
                curr_comp = candidates[curr_sol * n_comp + j];
                sol_i = sol_comp + (curr_sol * n_comp);
                int offset = curr_comp == 0 ? 0 : infl_comp_end_idx[curr_comp - 1];
                infl_i_size = infl_comp_end_idx[curr_comp] - offset;
                infl_i = infl_comp + offset;
                val_i = sol_i[curr_comp];
            }
            __syncthreads();
            // propagate the effect of the j-th component to all other components
            for (int k = 0; k < infl_i_size; k += blockDim.x) {
                int infl_idx = threadIdx.x + k;
                if (infl_idx >= infl_i_size) return;
                int target_comp = infl_i[infl_idx];
                sol_i[target_comp] = !val_i;
            }
            __syncthreads();
        }
    }
}

void compute_sccs_solutions(int max_threads, int max_blocks, int n_comp, int n_sol, int n_vars, int n_vertices,
                            int* h_candidates, int* h_infl_comp, int* h_infl_comp_end_idx, size_t infl_comp_bytes, int* h_comp,
                            int** d_comp, bool** d_sol_comp) {
    int threads_per_block = std::min(max_threads, n_comp);
    int n_blocks = std::min(max_blocks, n_sol);

    bool* h_sol_comp = (bool*)malloc(n_sol * n_comp * sizeof(bool));
    std::fill(h_sol_comp, h_sol_comp + n_sol * n_comp, true);
    
    size_t candidates_size = n_sol * n_comp * sizeof(int);
    size_t comp_size = n_vertices * sizeof(int);
    size_t sol_comp_size = n_sol * n_comp * sizeof(bool);
    size_t infl_comp_sizes_size = n_comp * sizeof(int);
    size_t max_heap_size = candidates_size + infl_comp_bytes + infl_comp_sizes_size + comp_size + sol_comp_size;
    set_heap_size(max_heap_size);

    int* d_candidates;
    int* d_infl_comp;
    int* d_infl_comp_sizes;
    HANDLE_ERROR(cudaMalloc((void**)&d_candidates, candidates_size));
    HANDLE_ERROR(cudaMemcpy(d_candidates, h_candidates, candidates_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void**)&d_infl_comp, infl_comp_bytes));
    HANDLE_ERROR(cudaMemcpy(d_infl_comp, h_infl_comp, infl_comp_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void**)&d_infl_comp_sizes, infl_comp_sizes_size));
    HANDLE_ERROR(cudaMemcpy(d_infl_comp_sizes, h_infl_comp_end_idx, infl_comp_sizes_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void**)d_comp, comp_size));
    HANDLE_ERROR(cudaMemcpy(*d_comp, h_comp, comp_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void**)d_sol_comp, sol_comp_size));
    HANDLE_ERROR(cudaMemcpy(*d_sol_comp, h_sol_comp, sol_comp_size, cudaMemcpyHostToDevice));

    // printf("Computing sccs solutions...\n");
    // printf("n_blocks: %d, threads_per_block: %d\n", n_blocks, threads_per_block);
    kernel_solve_2SAT<<<n_blocks, threads_per_block>>>(n_comp, n_sol, n_vars, d_candidates, *d_comp, d_infl_comp, d_infl_comp_sizes, *d_sol_comp);
    cudaDeviceSynchronize();
    checkCUDAError("computed sccs solutions");
    // printf("Computing sccs solutions done.\n");

    // Copy results back to host
    HANDLE_ERROR(cudaMemcpy(h_sol_comp, *d_sol_comp, n_sol * n_comp * sizeof(bool), cudaMemcpyDeviceToHost));
    // print_array(h_sol_comp, n_sol * n_comp, n_comp);

    // Free device memory
    HANDLE_ERROR(cudaFree(d_candidates));
    HANDLE_ERROR(cudaFree(d_infl_comp));

    // Free host memory
    free(h_candidates);
    free(h_infl_comp);
    free(h_comp);
}


/*
Runs 1 block per variable.
Each block load the component that contains its variable.
Then, for each candidate solution, it copies the value of the component to the variable.

Args:
    n_comp: number of components.
    n_sol: number of candidate solutions.
    n_vars: number of variables.
    comp: vertex component mapping.
    sol_comp: matrix of solutions on components, it should contain the results already computed.
    sol_var: matrix of solutions on variables, must be initialized with all values = false,
        it will contain the results.
*/
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

void solutions_sccs_to_vars(int max_threads, int max_blocks, int n_comp, int n_sol, int n_vars, int n_vertices,
                            int* d_comp, bool* d_sol_comp, bool** h_sol_var) {
    int threads_per_block = std::min(max_threads, n_sol);
    int n_blocks = std::min(max_blocks, n_vars);

    *h_sol_var = (bool*)malloc(n_sol * n_vars * sizeof(bool));
    std::fill(*h_sol_var, *h_sol_var + n_sol * n_vars, false);
    
    size_t sol_var_size = n_sol * n_vars * sizeof(bool);
    size_t comp_size = n_vertices * sizeof(int);
    size_t sol_comp_size = n_sol * n_comp * sizeof(bool);
    size_t max_heap_size = comp_size + sol_comp_size + sol_var_size;
    set_heap_size(max_heap_size);

    bool* d_sol_var;
    HANDLE_ERROR(cudaMalloc((void**)&d_sol_var, sol_var_size));

    // printf("Converting sccs solutions to variable solutions...\n");
    kernel_comp_to_var<<<n_blocks, threads_per_block>>>(n_comp, n_vars, n_sol, d_comp, d_sol_comp, d_sol_var);
    cudaDeviceSynchronize();
    checkCUDAError("converted sccs solutions to variable solutions");
    // printf("Converting sccs solutions to variable solutions done.\n");

    // Copy results back to host
    HANDLE_ERROR(cudaMemcpy(*h_sol_var, d_sol_var, n_sol * n_vars * sizeof(bool), cudaMemcpyDeviceToHost));
    // print_array(*h_sol_var, n_sol * n_vars, n_vars);

    // Free device memory
    HANDLE_ERROR(cudaFree(d_comp));
    HANDLE_ERROR(cudaFree(d_sol_comp));
    HANDLE_ERROR(cudaFree(d_sol_var));
}


/*
Runs 1 block per candidate solution pair.
Each block identifies the i-th and j-th candidate solutions, then initializes a counter to 0.
Each thread compares the values of 1 variable of the i-th and j-th candidate solutions.
If the values are different, it increments the counter.
When all threads finish, the block checks if the counter is greater than or equal to min_dist.

Args:
    n_sol: number of candidate solutions.
    n_vars: number of variables.
    sol_var: matrix of solutions on variables, it should contain the results already computed.
    sol_var_min_dist: matrix of compatibility between solutions, it should be initialized with all values = false,
        it will contain the results.
    min_dist: minimum hamming distance between two solutions to be considered compatible.
*/
__global__ void kernel_filter_min_dist(int n_sol, int n_vars, bool* sol_var, bool* sol_var_min_dist, int min_dist) {
    __shared__ int counter;
    
    for (int i = 0; i < n_sol * n_sol; i += gridDim.x) {
        int curr_sol_row = (blockIdx.x + i) / n_sol;
        if (curr_sol_row >= n_sol) return;
        int curr_sol_col = (blockIdx.x + i) % n_sol;
        if (curr_sol_col >= n_sol) return; // cannot happen

        if (curr_sol_row < curr_sol_col) return; // skip symmetric pairs
        if (curr_sol_row == curr_sol_col) return; // skip diagonal

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

void solutions_hamming_dist(int max_threads, int max_blocks, int n_sol, int n_vars, int min_dist, int ins_sol_size,
                            bool* out_results, bool* h_sol_var, bool** h_sol_var_min_dist) {
    // Append new solutions to the results of last iteration
    int n_sol_var_to_filter = n_sol + ins_sol_size;

    size_t sol_var_to_filter_size = n_sol_var_to_filter * n_vars * sizeof(bool);
    size_t sol_var_min_dist_size = n_sol_var_to_filter * n_sol_var_to_filter * sizeof(bool);
    size_t max_heap_size = sol_var_to_filter_size + sol_var_min_dist_size;
    set_heap_size(max_heap_size);

    bool* h_sol_var_to_filter = (bool*)malloc(n_sol_var_to_filter * n_vars * sizeof(bool));
    memcpy(h_sol_var_to_filter, out_results, ins_sol_size * n_vars * sizeof(bool));
    memcpy(h_sol_var_to_filter + ins_sol_size * n_vars, h_sol_var, n_sol * n_vars * sizeof(bool));

    bool* d_sol_var_to_filter;
    HANDLE_ERROR(cudaMalloc((void**)&d_sol_var_to_filter, sol_var_to_filter_size));
    HANDLE_ERROR(cudaMemcpy(d_sol_var_to_filter, h_sol_var_to_filter, sol_var_to_filter_size, cudaMemcpyHostToDevice));

    int threads_per_block = std::min(max_threads, n_vars);
    int n_blocks = std::min(max_blocks, n_sol_var_to_filter * n_sol_var_to_filter);

    bool* d_sol_var_min_dist;
    HANDLE_ERROR(cudaMalloc((void**)&d_sol_var_min_dist, sol_var_min_dist_size));

    // printf("Computing compatibility between solutions...\n");
    kernel_filter_min_dist<<<n_blocks, threads_per_block>>>(n_sol_var_to_filter, n_vars, d_sol_var_to_filter, d_sol_var_min_dist, min_dist);
    cudaDeviceSynchronize();
    checkCUDAError("filtered sccs solutions by min dist");
    // printf("Computing compatibility between solutions done.\n");

    // Copy results back to host
    *h_sol_var_min_dist = (bool*)malloc(n_sol_var_to_filter * n_sol_var_to_filter * sizeof(bool));
    HANDLE_ERROR(cudaMemcpy(*h_sol_var_min_dist, d_sol_var_min_dist, n_sol_var_to_filter * n_sol_var_to_filter * sizeof(bool), cudaMemcpyDeviceToHost));
    // print_array(*h_sol_var_min_dist, n_sol_var_to_filter * n_sol_var_to_filter, n_sol_var_to_filter);

    // Free device memory
    HANDLE_ERROR(cudaFree(d_sol_var_to_filter));
    HANDLE_ERROR(cudaFree(d_sol_var_min_dist));

    // Free host memory
    free(h_sol_var_to_filter);
}


/*
Serial solution.
Builds a vector of indices of the candidate solutions already inserted in the output pool. The indices refer to the h_sol_var_min_dist matrix.
Then, for each candidate solution in the new pool, it checks if it is compatible with all the solutions already inserted in the output pool.
If it is compatible, it is inserted in the output pool.

Args:
    init: true if this is the first iteration, false otherwise.
    n_sol: number of candidate solutions.
    n_vars: number of variables.
    n: maximum number of solutions to output.
    h_sol_var: matrix of new solutions on variables, it should contain the results already computed.
    h_sol_var_min_dist: matrix of compatibility between solutions, contains the compatibility between all candidates in the output pool and in the new pool.
    out_results: matrix of final results.
    n_out_results: number of candidate solutions already in out_results.
*/
int insert_new_solution(bool init, int n_sol, int n_vars, int n, bool* h_sol_var, bool* h_sol_var_min_dist,
    bool* out_results, int n_out_results) {
    // printf("Building final results...\n");
    int n_sol_var_to_filter = n_sol + n_out_results;
    std::vector<int> inserted_solutions(n_out_results);
    std::iota(inserted_solutions.begin(), inserted_solutions.end(), 0);

    // insert the first result
    if (init) {
    memcpy(out_results, h_sol_var, n_vars * sizeof(bool));
    inserted_solutions.push_back(0);
    }

    // insert the rest of the results
    for (int i = inserted_solutions.size(); i < n_sol_var_to_filter && inserted_solutions.size() < n; ++i) {
        bool valid = true;
        for (int val : inserted_solutions) {
            if (!h_sol_var_min_dist[i * n_sol_var_to_filter + val]) {
                valid = false;
                break;
            }
        }
        if (valid) {
            memcpy(out_results + inserted_solutions.size() * n_vars,
            h_sol_var + (i-n_out_results) * n_vars, n_vars * sizeof(bool));
            inserted_solutions.push_back(i);
        }
    }

    // Free host memory
    free(h_sol_var);
    free(h_sol_var_min_dist);

    // printf("Building final results done.\n");
    return inserted_solutions.size();
}
