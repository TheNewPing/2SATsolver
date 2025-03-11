#include "2sat_solver_linear.cu"
#include "2sat_solver_parallel.cu"

#include "../include/cuda_error.cu"
#include "../include/cuda_utilities.cu"

void linear_usage(std::string filename) {
    TwoSatSolverLinear solver_lin = TwoSatSolverLinear(filename);
    if (!solver_lin.solve_2SAT()) {
        std::cout << "No solution" << std::endl;
        return;
    }
    solver_lin.solve_from_all_nodes(1000, 10);
    for (const auto& sol : solver_lin.solutions) {
        std::cout << "solution: ";
        for (bool val : sol) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void parallel_usage(std::string filename) {
    // initialize adjacency matrix
    bool **h_adj, **h_adj_t;
    bool **d_adj, **d_adj_t;
    int n_vertices = fill_adjacency_matrix(filename, &h_adj, &h_adj_t);
    int n_vars = n_vertices / 2;


    std::cout << "Number of vertices: " << n_vertices << std::endl;
    // allocate in global memory
    HANDLE_ERROR(cudaMalloc(&d_adj, n_vertices * sizeof(bool*)));
    HANDLE_ERROR(cudaMalloc(&d_adj_t, n_vertices * sizeof(bool*)));
    for (int i = 0; i < n_vertices; ++i) {
        std::cout << "Allocating " << i << std::endl;
        HANDLE_ERROR(cudaMalloc((void**)&d_adj[i], n_vertices * sizeof(bool)));
        std::cout << "Allocating " << i << std::endl;
        HANDLE_ERROR(cudaMemcpy(d_adj[i], h_adj[i], n_vertices * sizeof(bool), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMalloc(&d_adj_t[i], n_vertices * sizeof(bool)));
        HANDLE_ERROR(cudaMemcpy(d_adj_t[i], h_adj_t[i], n_vertices * sizeof(bool), cudaMemcpyHostToDevice));
    }
    
    int threads_per_block = get_device_prop(0).maxThreadsPerBlock;
    int n = 10;

    std::cout << "Solving 2SAT in linear" << std::endl;

    // initialize results matrix
    bool **d_results, *d_solvable;
    HANDLE_ERROR(cudaMalloc(&d_solvable, n * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc(&d_results, n * sizeof(bool*)));
    for (int i = 0; i < n; ++i) {
        HANDLE_ERROR(cudaMalloc(&d_results[i], n_vars * sizeof(bool)));
    }

    printf("Solving 2SAT in parallel\n");
    kernel_solve_2SAT<<<(n + threads_per_block - 1) / threads_per_block, threads_per_block>>>(d_results, d_solvable, 0, n_vars, d_adj, d_adj_t);
    cudaDeviceSynchronize();
    checkCUDAError("parallel 2SAT solver");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }
    const char* filename = argv[1];
    
    linear_usage(filename);
    parallel_usage(filename);

    return 0;
}
