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

void print_array(bool* array, int length, int values_per_row) {
    for (int i = 0; i < length; ++i) {
        std::cout << array[i] << " ";
        if ((i + 1) % values_per_row == 0) {
            std::cout << std::endl;
        }
    }
    if (length % values_per_row != 0) {
        std::cout << std::endl;
    }
}

void parallel_usage(std::string filename, int n, int min_dist) {
    // initialize adjacency matrix
    bool *h_adj, *h_adj_t;
    bool *d_adj, *d_adj_t;
    int n_vertices = fill_adjacency_matrix(filename, &h_adj, &h_adj_t);
    int n_vars = n_vertices / 2;

    // allocate in global memory
    HANDLE_ERROR(cudaMalloc(&d_adj, n_vertices * n_vertices * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc(&d_adj_t, n_vertices * n_vertices * sizeof(bool)));
    HANDLE_ERROR(cudaMemcpy(d_adj, h_adj, n_vertices * n_vertices * sizeof(bool), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_adj_t, h_adj_t, n_vertices * n_vertices * sizeof(bool), cudaMemcpyHostToDevice));
    
    int threads_per_block = get_device_prop(0).maxThreadsPerBlock;

    // initialize results matrix
    bool *d_results, *d_solvable;
    HANDLE_ERROR(cudaMalloc(&d_solvable, n_vertices * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc(&d_results, n_vertices * n_vars * sizeof(bool)));

    // solve 2-SAT problem in parallel
    kernel_solve_2SAT<<<(n_vertices + threads_per_block - 1) / threads_per_block, threads_per_block>>>(d_results, d_solvable, 0, n_vars, d_adj, d_adj_t);
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
    
    linear_usage(filename, 10, 100);
    parallel_usage(filename, 10, 1);

    return 0;
}
