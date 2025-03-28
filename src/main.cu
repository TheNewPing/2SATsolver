#include <cassert>

#include "2sat_solver_linear.cu"
#include "2sat_solver_parallel.cu"

#include "../include/cuda_error.cu"
#include "../include/cuda_utilities.cu"

void linear_usage(std::string filename, int n, int min_dist) {
    TwoSatSolverLinear solver_lin = TwoSatSolverLinear(filename);
    if (!solver_lin.solve_2SAT()) {
        std::cout << "No solution" << std::endl;
        return;
    }
    solver_lin.solve_from_all_nodes(n, min_dist);
    for (const auto& sol : solver_lin.solutions) {
        std::cout << "solution: ";
        for (bool val : sol) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

bool is_transpose(const bool* matrix, const bool* transpose, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (matrix[i * n + j] != transpose[j * n + i]) {
                return false;
            }
        }
    }
    return true;
}

void parallel_usage(std::string filename, int n, int min_dist) {
    // initialize adjacency matrix
    bool *h_adj, *h_adj_t;
    bool *d_adj, *d_adj_t;
    int n_vertices = fill_adjacency_matrix(filename, &h_adj, &h_adj_t);
    int n_vars = n_vertices / 2;

    if (!is_transpose(h_adj, h_adj_t, n_vertices)) {
        std::cout << "Error: adjacency matrix is not transposed" << std::endl;
        return;
    }

    // allocate in global memory
    HANDLE_ERROR(cudaMalloc(&d_adj, n_vertices * n_vertices * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc(&d_adj_t, n_vertices * n_vertices * sizeof(bool)));
    HANDLE_ERROR(cudaMemcpy(d_adj, h_adj, n_vertices * n_vertices * sizeof(bool), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_adj_t, h_adj_t, n_vertices * n_vertices * sizeof(bool), cudaMemcpyHostToDevice));
    
    int max_threads = get_device_prop(0).maxThreadsPerBlock;
    int threads_per_block = std::min(max_threads, n_vertices);
    int n_blocks = (n_vertices + threads_per_block - 1) / threads_per_block;
    printf("Threads per block: %d\n", threads_per_block);
    printf("blocks: %d\n", n_blocks);
    printf("n_vertices: %d\n", n_vertices);

    // initialize results matrix
    bool *d_results, *d_solvable;
    HANDLE_ERROR(cudaMalloc(&d_solvable, n_vertices * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc(&d_results, n_vertices * n_vars * sizeof(bool)));

    // cudaDeviceSetLimit(cudaLimitStackSize, 1024 * sizeof(int));
    // checkCUDAError("setting stack size");
    size_t d_adj_size = n_vertices * n_vertices * sizeof(bool);
    size_t d_adj_t_size = n_vertices * n_vertices * sizeof(bool);
    size_t d_solvable_size = n_vertices * sizeof(bool);
    size_t d_results_size = n_vertices * n_vars * sizeof(bool);
    size_t assignment_size = n_vars * sizeof(bool);
    size_t order_size = n_vertices * sizeof(int);
    size_t comp_size = n_vertices * sizeof(int);
    size_t used_size = n_vertices * sizeof(bool);
    size_t dfs_stack_size = n_vertices * sizeof(int);
    size_t byte_per_thread = assignment_size + order_size + comp_size + used_size + dfs_stack_size;
    size_t total_bytes = byte_per_thread * threads_per_block * n_blocks + d_adj_size + d_adj_t_size + d_solvable_size + d_results_size;
    printf("Total bytes: %lu\n", total_bytes);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, total_bytes);

    // solve 2-SAT problem in parallel
    kernel_solve_2SAT<<<n_blocks, threads_per_block>>>(d_results, d_solvable, 0, n_vars, d_adj, d_adj_t);
    cudaDeviceSynchronize();
    checkCUDAError("parallel 2SAT solver");

    // copy results back to host memory
    bool *h_results = (bool*)malloc(n_vertices * n_vars * sizeof(bool));
    bool *h_solvable = (bool*)malloc(n_vertices * sizeof(bool));
    HANDLE_ERROR(cudaMemcpy(h_results, d_results, n_vertices * n_vars * sizeof(bool), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_solvable, d_solvable, n_vertices * sizeof(bool), cudaMemcpyDeviceToHost));

    // check if the 2-SAT problem is solvable for all starting nodes
    // it should be solvable for all starting nodes, or from none of them
    for (int i = 0; i < n_vertices; ++i) {
        assert(h_solvable[i] == true);
    }

    // prepare output
    int results_count = 0;
    bool *out_results = (bool*)malloc(n * n_vars * sizeof(bool));
    std::fill(out_results, out_results + n * n_vars, false);

    // insert the first result
    memcpy(out_results, h_results, n_vars * sizeof(bool));
    ++results_count;

    // insert the rest of the results
    for (int i = 1; i < n_vertices && results_count < n; ++i) {
        bool valid = true;
        for (int k = 0; k < results_count; ++k) {
            int hamming_dist = 0;
            for (int j = 0; j < n_vars; ++j) {
                if (h_results[i * n_vars + j] != out_results[k * n_vars + j])
                    ++hamming_dist;
            }
            if (hamming_dist < min_dist) {
                valid = false;
                break;
            }
        }
        if (valid) {
            memcpy(out_results + results_count * n_vars, h_results + i * n_vars, n_vars * sizeof(bool));
            ++results_count;
        }
    }

    // print output
    printf("Parallel solutions:\n");
    print_array(out_results, results_count * n_vars, n_vars);

    free(h_adj);
    free(h_adj_t);
    free(h_results);
    free(h_solvable);
    free(out_results);
    cudaFree(d_adj);
    cudaFree(d_adj_t);
    cudaFree(d_results);
    cudaFree(d_solvable);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }
    const char* filename = argv[1];
    
    linear_usage(filename, 10, 1);
    parallel_usage(filename, 10, 1);

    return 0;
}
