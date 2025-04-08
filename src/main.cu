#include <cassert>

#include "2sat2scc.cu"
#include "2sat_solver_parallel.cu"

#include "../include/cuda_error.cu"
#include "../include/cuda_utilities.cu"

void parallel_usage(std::string filename, int n, int min_dist) {
    TwoSat2SCC sccs = TwoSat2SCC(filename);
    int n_vars = sccs.n_vars;
    int n_vertices = sccs.n_vertices;

    printf("building SCC...\n");
    sccs.build_SCC();
    printf("building SCC done.\n");
    printf("sccs.comp:\n");
    for (const auto& c : sccs.comp) {
        printf("%d ", c);
    }
    printf("\n");
    printf("building candidates...\n");
    sccs.build_candidates(n);
    printf("building candidates done.\n");
    int n_sol = sccs.candidates.size();
    int n_comp = sccs.infl_comp.size();

    // Prepare data for kernel
    printf("Preparing data for kernel...\n");
    printf("candidates...\n");
    int* candidates = sccs.arrayify_candidates();
    printf("candidates done.\n");

    print_vv(sccs.candidates);
    print_vv(sccs.infl_comp);

    printf("infl_comp...\n");
    bool* infl_comp = sccs.arrayify_infl_comp();
    printf("infl_comp done.\n");
    print_array(infl_comp, n_comp * n_comp, n_comp);

    printf("comp...\n");
    int* comp = sccs.arrayify_comp();
    printf("comp done.\n");
    printf("Preparing data for kernel done.\n");

    bool* h_sol_comp = new bool[n_sol * n_comp];
    std::fill(h_sol_comp, h_sol_comp + n_sol * n_comp, true);
    bool* h_sol_var = new bool[n_sol * n_vars];
    std::fill(h_sol_var, h_sol_var + n_sol * n_vars, true);

    // Set max heap size
    size_t candidates_size = n_sol * n_comp * sizeof(int);
    size_t infl_comp_size = n_comp * n_comp * sizeof(bool);
    size_t comp_size = n_vertices * sizeof(int);
    size_t sol_comp_size = n_sol * n_comp * sizeof(bool);
    size_t sol_var_size = n_sol * n_vars * sizeof(bool);
    size_t max_heap_size = candidates_size + infl_comp_size + comp_size + sol_comp_size + sol_var_size;
    printf("Max heap size: %zu bytes\n", max_heap_size);
    printf("n_sol: %d\n", n_sol);
    printf("n_comp: %d\n", n_comp);
    printf("n_vars: %d\n", n_vars);
    printf("candidates_size: %zu bytes\n", candidates_size);
    printf("infl_comp_size: %zu bytes\n", infl_comp_size);
    printf("comp_size: %zu bytes\n", comp_size);
    printf("sol_comp_size: %zu bytes\n", sol_comp_size);
    printf("sol_var_size: %zu bytes\n", sol_var_size);
    size_t free_mem, total_mem;
    HANDLE_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    if (max_heap_size > free_mem) {
        std::cerr << "Not enough memory on device. Required: " << max_heap_size / (1024 * 1024) << " MB, Available: " << free_mem / (1024 * 1024) << " MB" << std::endl;
        return;
    }
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, max_heap_size);
    checkCUDAError("Heap size set");
    
    // Allocate device memory
    int* d_candidates;
    bool* d_infl_comp;
    int* d_comp;
    bool* d_sol_comp;
    bool* d_sol_var;
    HANDLE_ERROR(cudaMalloc((void**)&d_candidates, n_sol * n_comp * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_infl_comp, n_comp * n_comp * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc((void**)&d_comp, n_vertices * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_sol_comp, n_sol * n_comp * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc((void**)&d_sol_var, n_sol * n_vars * sizeof(bool)));
    HANDLE_ERROR(cudaMemcpy(d_candidates, candidates, n_sol * n_comp * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_infl_comp, infl_comp, n_comp * n_comp * sizeof(bool), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_comp, comp, n_vertices * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_sol_comp, h_sol_comp, n_sol * n_comp * sizeof(bool), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_sol_var, h_sol_var, n_sol * n_vars * sizeof(bool), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int max_threads = get_device_prop(0).maxThreadsPerBlock;
    int threads_per_block = std::min(max_threads, n_comp);
    int n_blocks = n_sol;
    printf("Threads per block: %d\n", threads_per_block);
    printf("blocks: %d\n", n_blocks);
    kernel_solve_2SAT<<<n_blocks, threads_per_block>>>(n_comp, n_sol, n_vars, d_candidates, d_infl_comp, d_comp, d_sol_comp, d_sol_var);
    cudaDeviceSynchronize();
    checkCUDAError("parallel 2SAT solver");

    // Copy results back to host
    HANDLE_ERROR(cudaMemcpy(h_sol_comp, d_sol_comp, n_sol * n_comp * sizeof(bool), cudaMemcpyDeviceToHost));

    print_array(h_sol_comp, n_sol * n_comp, n_comp);

    // // prepare output
    // int results_count = 0;
    // bool *out_results = (bool*)malloc(n * n_vars * sizeof(bool));
    // std::fill(out_results, out_results + n * n_vars, false);

    // // insert the first result
    // memcpy(out_results, h_results, n_vars * sizeof(bool));
    // ++results_count;

    // // insert the rest of the results
    // for (int i = 1; i < n_vertices && results_count < n; ++i) {
    //     bool valid = true;
    //     for (int k = 0; k < results_count; ++k) {
    //         int hamming_dist = 0;
    //         for (int j = 0; j < n_vars; ++j) {
    //             if (h_results[i * n_vars + j] != out_results[k * n_vars + j])
    //                 ++hamming_dist;
    //         }
    //         if (hamming_dist < min_dist) {
    //             valid = false;
    //             break;
    //         }
    //     }
    //     if (valid) {
    //         memcpy(out_results + results_count * n_vars, h_results + i * n_vars, n_vars * sizeof(bool));
    //         ++results_count;
    //     }
    // }

    // // print output
    // printf("Parallel solutions:\n");
    // print_array(out_results, results_count * n_vars, n_vars);

}
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <filename> <n>" << std::endl;
        return 1;
    }
    const char* filename = argv[1];
    int n = std::stoi(argv[2]);

    parallel_usage(filename, n, 1);

    return 0;
}
