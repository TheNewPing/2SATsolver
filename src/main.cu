#include <cassert>
#include <chrono>

#include "2sat2scc.cu"
#include "2sat_solver_parallel.cu"
#include "2sat_solver_serial.cu"

int parallel_usage(std::string filename, int n, int min_dist, bool print_sol=false) {
    TwoSat2SCC sccs = TwoSat2SCC(filename);
    int n_vars = sccs.n_vars;
    int n_vertices = sccs.n_vertices;

    // printf("building SCC...\n");
    if (!sccs.build_SCC()) {
        printf("No solution found.\n");
        return -1;
    }
    // printf("building SCC done.\n");
    // printf("sccs.comp:\n");
    // for (const auto& c : sccs.comp) {
        // printf("%d ", c);
    // }
    // printf("\n");

    // prepare output
    bool *out_results = (bool*)malloc(n * n_vars * sizeof(bool));
    int n_out_results = 0;
    bool init = true;
        
    int max_threads = get_device_prop(0).maxThreadsPerBlock;
    int max_blocks = get_device_prop(0).maxGridSize[0];

    // approximate number of solutions to the closest multiple of the number of multiprocessors
    int sm_count = get_device_prop(0).multiProcessorCount;
    int n_sol = ((n + sm_count - 1) / sm_count) * sm_count;

    // int *h_infl_comp;
    // int *h_infl_comp_end_idx;
    // int *h_comp = sccs.arrayify_comp();
    // size_t infl_comp_bytes = sccs.arrayify_infl_comp(&h_infl_comp, &h_infl_comp_end_idx);
    // int n_comp = sccs.infl_comp.size();

    int *h_infl_comp;
    int *h_infl_comp_end_idx;
    int *h_comp = sccs.arrayify_comp();
    size_t infl_comp_bytes = sccs.arrayify_infl_comp(&h_infl_comp, &h_infl_comp_end_idx);
    int n_comp = sccs.infl_comp.size();

    while (n_out_results < n) {
        printf("Current number of solutions: %d\n", n_out_results);

        sccs.build_candidates(n_sol, init);
        int *h_candidates = sccs.arrayify_candidates();
        // print_vv(sccs.candidates);

        // ----------- Compute solutions based on sccs ----------- 
        int *d_comp;
        bool *d_sol_comp;
        compute_sccs_solutions(max_threads, max_blocks, n_comp, n_sol, n_vars, n_vertices,
                               h_candidates, h_infl_comp, h_infl_comp_end_idx, infl_comp_bytes, h_comp,
                               &d_comp, &d_sol_comp);
        free(h_candidates);

        // ----------- Transfer sccs solutions to variable solutions ----------- 
        bool *h_sol_var;
        solutions_sccs_to_vars(max_threads, max_blocks, n_comp, n_sol, n_vars, n_vertices,
                               d_comp, d_sol_comp, &h_sol_var);
        HANDLE_ERROR(cudaFree(d_comp));
        HANDLE_ERROR(cudaFree(d_sol_comp));

        // ----------- Compute compatibility between solutions based on min dist ----------- 
        bool *h_sol_var_min_dist;
        solutions_hamming_dist(max_threads, max_blocks, n_sol, n_vars, min_dist,
                               n_out_results, out_results, h_sol_var, &h_sol_var_min_dist);

        // ----------- Build the final results -----------
        n_out_results = insert_new_solution(init, n_sol, n_vars, n, h_sol_var, h_sol_var_min_dist,
                                            out_results, n_out_results);
        free(h_sol_var);
        free(h_sol_var_min_dist);

        init = false;
        // printf("PARTIAL solutions:\n");
        // print_array(out_results, n_out_results * n_vars, n_vars, "sol: ");
    }

    free(h_comp);
    free(h_infl_comp);

    // print output
    if (print_sol) {
        printf("Parallel solutions:\n");
        print_array(out_results, n_out_results * n_vars, n_vars, "sol: ");
    }

    // verify output
    if (verify_solutions(out_results, n_out_results, sccs.vars, n_vars)) {
        printf("All solutions are valid.\n");
        return n_out_results;
    } else {
        printf("Some solutions are invalid.\n");
        return -1;
    }
}

int serial_usage(std::string filename, int n, int min_dist, bool print_sol=false) {
    TwoSatSolverSerial solver_ser = TwoSatSolverSerial(filename);
    if (!solver_ser.solve_2SAT()) {
        std::cout << "No solution" << std::endl;
        return -1;
    }
    solver_ser.solve_from_all_nodes(n, min_dist);

    if (print_sol) {
        std::cout << "Serial solutions:" << std::endl;
        for (const auto& sol : solver_ser.solutions) {
            std::cout << "sol: ";
            for (bool val : sol) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    if (verify_solutions(solver_ser.solutions, solver_ser.vars)) {
        std::cout << "All solutions are valid." << std::endl;
        return solver_ser.solutions.size();
    } else {
        std::cout << "Some solutions are invalid." << std::endl;
        return -1;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " [-c] [-v] <filename> <number of solutions> <min hamming dist>" << std::endl;
        return 1;
    }

    bool run_serial = false;
    bool print_sol = false;
    int arg_offset = 0;

    if (std::string(argv[1]) == "-c") {
        run_serial = true;
        arg_offset++;
    }
    if (std::string(argv[1 + arg_offset]) == "-v") {
        print_sol = true;
        arg_offset++;
    }

    const char* filename = argv[1 + arg_offset];
    int n = std::stoi(argv[2 + arg_offset]);
    int min_dist = std::stoi(argv[3 + arg_offset]);

    auto start_par = std::chrono::high_resolution_clock::now();
    int par_n_sol = parallel_usage(filename, n, min_dist, print_sol);
    auto end_par = std::chrono::high_resolution_clock::now();
    auto ms_par = std::chrono::duration_cast<std::chrono::milliseconds>(end_par - start_par);

    std::cout << "Parallel time: " << ms_par.count() << " ms" << std::endl;

    if (run_serial) {
        std::cout << std::endl;
        auto start_ser = std::chrono::high_resolution_clock::now();
        int ser_n_sol = serial_usage(filename, n, min_dist, print_sol);
        auto end_ser = std::chrono::high_resolution_clock::now();
        auto ms_ser = std::chrono::duration_cast<std::chrono::milliseconds>(end_ser - start_ser);
        std::cout << "Serial time: " << ms_ser.count() << " ms" << std::endl;
        std::cout << std::endl << "Speedup: " << (double)ms_ser.count() / ms_par.count() << "x" << std::endl;
        std::cout << "Solutions amount difference: " << par_n_sol - ser_n_sol << std::endl;
    }

    return 0;
}
