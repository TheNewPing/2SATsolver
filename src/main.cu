#include <cassert>

#include "2sat2scc.cu"
#include "2sat_solver_parallel.cu"

#include "../include/cuda_error.cu"
#include "../include/cuda_utilities.cu"

void parallel_usage(std::string filename, int n, int min_dist) {
    TwoSat2SCC sccs = TwoSat2SCC(filename);
    int n_vars = sccs.n_vars;
    int n_vertices = sccs.n_vertices;

    // printf("building SCC...\n");
    sccs.build_SCC();
    // printf("building SCC done.\n");
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
        // printf("building candidates...\n");
        sccs.build_candidates(n, init);
        // printf("building candidates done.\n");
        int n_sol = sccs.candidates.size();
        int n_comp = sccs.infl_comp.size();

        // Prepare data for kernel
        // printf("Preparing data for kernel...\n");
        // printf("candidates...\n");
        int* h_candidates = sccs.arrayify_candidates();
        // printf("candidates done.\n");

        // print_vv(sccs.candidates);
        // print_vv(sccs.infl_comp);

        // printf("infl_comp...\n");
        bool* h_infl_comp = sccs.arrayify_infl_comp();
        // printf("infl_comp done.\n");
        // print_array(h_infl_comp, n_comp * n_comp, n_comp);

        // printf("comp...\n");
        int* h_comp = sccs.arrayify_comp();
        // printf("comp done.\n");
        // printf("Preparing data for kernel done.\n");
        
        int max_threads = get_device_prop(0).maxThreadsPerBlock;
        int max_blocks = get_device_prop(0).maxGridSize[0];

        // ----------- Compute solutions based on sccs ----------- 
        int threads_per_block = std::min(max_threads, n_comp);
        int n_blocks = std::min(max_blocks, n_sol);

        bool* h_sol_comp = new bool[n_sol * n_comp];
        std::fill(h_sol_comp, h_sol_comp + n_sol * n_comp, true);

        
        size_t candidates_size = n_sol * n_comp * sizeof(int);
        size_t infl_comp_size = n_comp * n_comp * sizeof(bool);
        size_t comp_size = n_vertices * sizeof(int);
        size_t sol_comp_size = n_sol * n_comp * sizeof(bool);
        size_t max_heap_size = candidates_size + infl_comp_size + comp_size + sol_comp_size;
        set_heap_size(max_heap_size);

        int* d_candidates;
        bool* d_infl_comp;
        int* d_comp;
        bool* d_sol_comp;
        HANDLE_ERROR(cudaMalloc((void**)&d_candidates, candidates_size));
        HANDLE_ERROR(cudaMemcpy(d_candidates, h_candidates, candidates_size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMalloc((void**)&d_infl_comp, infl_comp_size));
        HANDLE_ERROR(cudaMemcpy(d_infl_comp, h_infl_comp, infl_comp_size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMalloc((void**)&d_comp, comp_size));
        HANDLE_ERROR(cudaMemcpy(d_comp, h_comp, comp_size, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMalloc((void**)&d_sol_comp, sol_comp_size));
        HANDLE_ERROR(cudaMemcpy(d_sol_comp, h_sol_comp, sol_comp_size, cudaMemcpyHostToDevice));

        kernel_solve_2SAT<<<n_blocks, threads_per_block>>>(n_comp, n_sol, n_vars, d_candidates, d_infl_comp, d_comp, d_sol_comp);
        cudaDeviceSynchronize();
        checkCUDAError("computed sccs solutions");

        // Copy results back to host
        HANDLE_ERROR(cudaMemcpy(h_sol_comp, d_sol_comp, n_sol * n_comp * sizeof(bool), cudaMemcpyDeviceToHost));
        // print_array(h_sol_comp, n_sol * n_comp, n_comp);

        // Free device memory
        HANDLE_ERROR(cudaFree(d_candidates));
        HANDLE_ERROR(cudaFree(d_infl_comp));
        // ----------- 

        // ----------- Transfer sccs solutions to variable solutions ----------- 
        threads_per_block = std::min(max_threads, n_sol);
        n_blocks = std::min(max_blocks, n_vars);

        bool* h_sol_var = new bool[n_sol * n_vars];
        
        size_t sol_var_size = n_sol * n_vars * sizeof(bool);
        max_heap_size = comp_size + sol_comp_size + sol_var_size;
        set_heap_size(max_heap_size);

        bool* d_sol_var;
        HANDLE_ERROR(cudaMalloc((void**)&d_sol_var, sol_var_size));

        kernel_comp_to_var<<<n_blocks, threads_per_block>>>(n_comp, n_vars, n_sol, d_comp, d_sol_comp, d_sol_var);
        cudaDeviceSynchronize();
        checkCUDAError("converted sccs solutions to variable solutions");

        // Copy results back to host
        HANDLE_ERROR(cudaMemcpy(h_sol_var, d_sol_var, n_sol * n_vars * sizeof(bool), cudaMemcpyDeviceToHost));
        // print_array(h_sol_var, n_sol * n_vars, n_vars);

        // Free device memory
        HANDLE_ERROR(cudaFree(d_comp));
        HANDLE_ERROR(cudaFree(d_sol_comp));
        HANDLE_ERROR(cudaFree(d_sol_var));
        // ----------- 

        // ----------- Compute compatibility between solutions based on min dist ----------- 
        // Append new solutions to the results of last iteration
        int n_sol_var_to_filter = n_sol + inserted_solutions.size();

        size_t sol_var_to_filter_size = n_sol_var_to_filter * n_vars * sizeof(bool);
        size_t sol_var_min_dist_size = n_sol_var_to_filter * n_sol_var_to_filter * sizeof(bool);
        max_heap_size = sol_var_to_filter_size + sol_var_min_dist_size;
        set_heap_size(max_heap_size);

        bool* h_sol_var_to_filter = new bool[n_sol_var_to_filter * n_vars];
        memcpy(h_sol_var_to_filter, out_results, inserted_solutions.size() * n_vars * sizeof(bool));
        memcpy(h_sol_var_to_filter + inserted_solutions.size() * n_vars, h_sol_var, n_sol * n_vars * sizeof(bool));

        bool* d_sol_var_to_filter;
        HANDLE_ERROR(cudaMalloc((void**)&d_sol_var_to_filter, sol_var_to_filter_size));
        HANDLE_ERROR(cudaMemcpy(d_sol_var_to_filter, h_sol_var_to_filter, sol_var_to_filter_size, cudaMemcpyHostToDevice));

        threads_per_block = std::min(max_threads, n_vars);
        n_blocks = std::min(max_blocks, n_sol_var_to_filter * n_sol_var_to_filter);

        bool* d_sol_var_min_dist;
        HANDLE_ERROR(cudaMalloc((void**)&d_sol_var_min_dist, sol_var_min_dist_size));

        kernel_filter_min_dist<<<n_blocks, threads_per_block>>>(n_sol_var_to_filter, n_vars, d_sol_var_to_filter, d_sol_var_min_dist, min_dist);
        cudaDeviceSynchronize();
        checkCUDAError("filtered sccs solutions by min dist");

        // Copy results back to host
        bool* h_sol_var_min_dist = new bool[n_sol_var_to_filter * n_sol_var_to_filter];
        HANDLE_ERROR(cudaMemcpy(h_sol_var_min_dist, d_sol_var_min_dist, n_sol_var_to_filter * n_sol_var_to_filter * sizeof(bool), cudaMemcpyDeviceToHost));
        // print_array(h_sol_var_min_dist, n_sol_var_to_filter * n_sol_var_to_filter, n_sol_var_to_filter);

        // Free device memory
        HANDLE_ERROR(cudaFree(d_sol_var_to_filter));
        HANDLE_ERROR(cudaFree(d_sol_var_min_dist));
        // ----------- 

        // ----------- Build the final results -----------
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
        // ----------- 
        init = false;
    }

    // print output
    printf("Parallel solutions:\n");
    print_array(out_results, inserted_solutions.size() * n_vars, n_vars);

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
