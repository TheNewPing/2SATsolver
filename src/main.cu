#include <cassert>

#include "2sat2scc.cu"
#include "2sat_solver_parallel.cu"

#include "../include/cuda_error.cu"
#include "../include/cuda_utilities.cu"

void arrayify_sccs(TwoSat2SCC *sccs, int n, bool init, int** h_candidates, bool** h_infl_comp, int** h_comp) {
    printf("building candidates...\n");
    sccs->build_candidates(n, init);
    printf("building candidates done.\n");

    // Prepare data for kernel
    printf("Preparing data for kernel...\n");
    printf("candidates...\n");
    *h_candidates = sccs->arrayify_candidates();
    printf("candidates done.\n");

    // print_vv(sccs.candidates);
    // print_vv(sccs.infl_comp);

    printf("infl_comp...\n");
    *h_infl_comp = sccs->arrayify_infl_comp();
    printf("infl_comp done.\n");
    // print_array(h_infl_comp, n_comp * n_comp, n_comp);

    printf("comp...\n");
    *h_comp = sccs->arrayify_comp();
    printf("comp done.\n");
    printf("Preparing data for kernel done.\n");
}

void compute_sccs_solutions(int max_threads, int max_blocks, int n_comp, int n_sol, int n_vars, int n_vertices,
                            int* h_candidates, bool* h_infl_comp, int* h_comp,
                            int** d_comp, bool** d_sol_comp) {
    int threads_per_block = std::min(max_threads, n_comp);
    int n_blocks = std::min(max_blocks, n_sol);

    bool* h_sol_comp = (bool*)malloc(n_sol * n_comp * sizeof(bool));
    std::fill(h_sol_comp, h_sol_comp + n_sol * n_comp, true);
    
    size_t candidates_size = n_sol * n_comp * sizeof(int);
    size_t infl_comp_size = n_comp * n_comp * sizeof(bool);
    size_t comp_size = n_vertices * sizeof(int);
    size_t sol_comp_size = n_sol * n_comp * sizeof(bool);
    size_t max_heap_size = candidates_size + infl_comp_size + comp_size + sol_comp_size;
    set_heap_size(max_heap_size);

    int* d_candidates;
    bool* d_infl_comp;
    HANDLE_ERROR(cudaMalloc((void**)&d_candidates, candidates_size));
    HANDLE_ERROR(cudaMemcpy(d_candidates, h_candidates, candidates_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void**)&d_infl_comp, infl_comp_size));
    HANDLE_ERROR(cudaMemcpy(d_infl_comp, h_infl_comp, infl_comp_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void**)d_comp, comp_size));
    HANDLE_ERROR(cudaMemcpy(*d_comp, h_comp, comp_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void**)d_sol_comp, sol_comp_size));
    HANDLE_ERROR(cudaMemcpy(*d_sol_comp, h_sol_comp, sol_comp_size, cudaMemcpyHostToDevice));

    printf("Computing sccs solutions...\n");
    printf("n_blocks: %d, threads_per_block: %d\n", n_blocks, threads_per_block);
    kernel_solve_2SAT<<<n_blocks, threads_per_block>>>(n_comp, n_sol, n_vars, d_candidates, d_infl_comp, *d_comp, *d_sol_comp);
    cudaDeviceSynchronize();
    checkCUDAError("computed sccs solutions");
    printf("Computing sccs solutions done.\n");

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

void solutions_sccs_to_vars(int max_threads, int max_blocks, int n_comp, int n_sol, int n_vars, int n_vertices,
                            int* d_comp, bool* d_sol_comp, bool** h_sol_var) {
    int threads_per_block = std::min(max_threads, n_sol);
    int n_blocks = std::min(max_blocks, n_vars);

    *h_sol_var = (bool*)malloc(n_sol * n_vars * sizeof(bool));
    
    size_t sol_var_size = n_sol * n_vars * sizeof(bool);
    size_t comp_size = n_vertices * sizeof(int);
    size_t sol_comp_size = n_sol * n_comp * sizeof(bool);
    size_t max_heap_size = comp_size + sol_comp_size + sol_var_size;
    set_heap_size(max_heap_size);

    bool* d_sol_var;
    HANDLE_ERROR(cudaMalloc((void**)&d_sol_var, sol_var_size));

    printf("Converting sccs solutions to variable solutions...\n");
    kernel_comp_to_var<<<n_blocks, threads_per_block>>>(n_comp, n_vars, n_sol, d_comp, d_sol_comp, d_sol_var);
    cudaDeviceSynchronize();
    checkCUDAError("converted sccs solutions to variable solutions");
    printf("Converting sccs solutions to variable solutions done.\n");

    // Copy results back to host
    HANDLE_ERROR(cudaMemcpy(*h_sol_var, d_sol_var, n_sol * n_vars * sizeof(bool), cudaMemcpyDeviceToHost));
    // print_array(*h_sol_var, n_sol * n_vars, n_vars);

    // Free device memory
    HANDLE_ERROR(cudaFree(d_comp));
    HANDLE_ERROR(cudaFree(d_sol_comp));
    HANDLE_ERROR(cudaFree(d_sol_var));
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

    printf("Computing compatibility between solutions...\n");
    kernel_filter_min_dist<<<n_blocks, threads_per_block>>>(n_sol_var_to_filter, n_vars, d_sol_var_to_filter, d_sol_var_min_dist, min_dist);
    cudaDeviceSynchronize();
    checkCUDAError("filtered sccs solutions by min dist");
    printf("Computing compatibility between solutions done.\n");

    // Copy results back to host
    *h_sol_var_min_dist = (bool*)malloc(n_sol_var_to_filter * n_sol_var_to_filter * sizeof(bool));
    HANDLE_ERROR(cudaMemcpy(*h_sol_var_min_dist, d_sol_var_min_dist, n_sol_var_to_filter * n_sol_var_to_filter * sizeof(bool), cudaMemcpyDeviceToHost));
    // print_array(h_sol_var_min_dist, n_sol_var_to_filter * n_sol_var_to_filter, n_sol_var_to_filter);

    // Free device memory
    HANDLE_ERROR(cudaFree(d_sol_var_to_filter));
    HANDLE_ERROR(cudaFree(d_sol_var_min_dist));

    // Free host memory
    free(h_sol_var_to_filter);
}

void insert_new_solution(bool init, int n_sol, int n_vars, int n, bool* h_sol_var, bool* h_sol_var_min_dist,
                         bool* out_results, std::vector<int>& inserted_solutions) {
    printf("Building final results...\n");
    // insert the first result
    if (init) {
        memcpy(out_results, h_sol_var, n_vars * sizeof(bool));
        inserted_solutions.push_back(0);
    }

    // insert the rest of the results
    for (int i = 1; i < n_sol && inserted_solutions.size() < n; ++i) {
        bool valid = true;
        for (int val : inserted_solutions) {
            if (!h_sol_var_min_dist[i * n_sol + val]) {
                valid = false;
                break;
            }
        }
        if (valid) {
            memcpy(out_results + inserted_solutions.size() * n_vars,
                h_sol_var + i * n_vars, n_vars * sizeof(bool));
            inserted_solutions.push_back(i);
        }
    }

    // Free host memory
    free(h_sol_var);
    free(h_sol_var_min_dist);

    printf("Building final results done.\n");
}

void parallel_usage(std::string filename, int n, int min_dist) {
    TwoSat2SCC sccs = TwoSat2SCC(filename);
    int n_vars = sccs.n_vars;
    int n_vertices = sccs.n_vertices;

    printf("building SCC...\n");
    if (!sccs.build_SCC()) {
        printf("No solution found.\n");
        return;
    }
    printf("building SCC done.\n");
    // printf("sccs.comp:\n");
    // for (const auto& c : sccs.comp) {
        // printf("%d ", c);
    // }
    // printf("\n");

    // prepare output
    std::vector<int> inserted_solutions;
    bool *out_results = (bool*)malloc(n * n_vars * sizeof(bool));
    bool init = true;

    while (inserted_solutions.size() < n) {
        printf("Current number of solutions: %zu\n", inserted_solutions.size());

        int *h_candidates;
        bool *h_infl_comp;
        int *h_comp;
        arrayify_sccs(&sccs, n, init, &h_candidates, &h_infl_comp, &h_comp);
        int n_sol = sccs.candidates.size();
        int n_comp = sccs.infl_comp.size();
        printf("n_sol: %d, n_comp: %d\n", n_sol, n_comp);
        
        int max_threads = get_device_prop(0).maxThreadsPerBlock;
        int max_blocks = get_device_prop(0).maxGridSize[0];
        printf("max_threads: %d, max_blocks: %d\n", max_threads, max_blocks);

        // ----------- Compute solutions based on sccs ----------- 
        int *d_comp;
        bool *d_sol_comp;
        compute_sccs_solutions(max_threads, max_blocks, n_comp, n_sol, n_vars, n_vertices,
                               h_candidates, h_infl_comp, h_comp,
                               &d_comp, &d_sol_comp);

        // ----------- Transfer sccs solutions to variable solutions ----------- 
        bool *h_sol_var;
        solutions_sccs_to_vars(max_threads, max_blocks, n_comp, n_sol, n_vars, n_vertices,
                               d_comp, d_sol_comp, &h_sol_var);

        // ----------- Compute compatibility between solutions based on min dist ----------- 
        bool *h_sol_var_min_dist;
        solutions_hamming_dist(max_threads, max_blocks, n_sol, n_vars, min_dist,
                               inserted_solutions.size(), out_results, h_sol_var, &h_sol_var_min_dist);

        // ----------- Build the final results -----------
        insert_new_solution(init, n_sol, n_vars, n, h_sol_var, h_sol_var_min_dist,
                            out_results, inserted_solutions);

        init = false;
    }

    // print output
    printf("Parallel solutions:\n");
    print_array(out_results, inserted_solutions.size() * n_vars, n_vars);

}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <filename> <number of solutions> <min hamming dist>" << std::endl;
        return 1;
    }
    const char* filename = argv[1];
    int n = std::stoi(argv[2]);
    int min_dist = std::stoi(argv[3]);

    parallel_usage(filename, n, min_dist);

    return 0;
}
