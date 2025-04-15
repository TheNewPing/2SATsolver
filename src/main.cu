#include <cassert>
#include <chrono>

#include "2sat2scc.cu"
#include "2sat_solver_parallel.cu"
#include "2sat_solver_serial.cu"

void parallel_usage(std::string filename, int n, int min_dist) {
    TwoSat2SCC sccs = TwoSat2SCC(filename);
    int n_vars = sccs.n_vars;
    int n_vertices = sccs.n_vertices;

    // printf("building SCC...\n");
    if (!sccs.build_SCC()) {
        printf("No solution found.\n");
        return;
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

    while (n_out_results < n) {
        printf("Current number of solutions: %d\n", n_out_results);

        int *h_candidates;
        bool *h_infl_comp;
        int *h_comp;
        arrayify_sccs(&sccs, n, init, &h_candidates, &h_infl_comp, &h_comp);
        int n_sol = sccs.candidates.size();
        int n_comp = sccs.infl_comp.size();
        // printf("n_sol: %d, n_comp: %d\n", n_sol, n_comp);
        
        int max_threads = get_device_prop(0).maxThreadsPerBlock;
        int max_blocks = get_device_prop(0).maxGridSize[0];
        // printf("max_threads: %d, max_blocks: %d\n", max_threads, max_blocks);

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
                               n_out_results, out_results, h_sol_var, &h_sol_var_min_dist);

        // ----------- Build the final results -----------
        n_out_results = insert_new_solution(init, n_sol, n_vars, n, h_sol_var, h_sol_var_min_dist,
                                            out_results, n_out_results);

        init = false;
    }

    // print output
    printf("Parallel solutions:\n");
    print_array(out_results, n_out_results * n_vars, n_vars, "sol: ");

}

void serial_usage(std::string filename, int n, int min_dist) {
    TwoSatSolverSerial solver_ser = TwoSatSolverSerial(filename);
    if (!solver_ser.solve_2SAT()) {
        std::cout << "No solution" << std::endl;
        return;
    }
    solver_ser.solve_from_all_nodes(n, min_dist);
    std::cout << "Serial solutions:" << std::endl;
    for (const auto& sol : solver_ser.solutions) {
        std::cout << "solution: ";
        for (bool val : sol) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <filename> <number of solutions> <min hamming dist>" << std::endl;
        return 1;
    }
    const char* filename = argv[1];
    int n = std::stoi(argv[2]);
    int min_dist = std::stoi(argv[3]);

    auto start_ser = std::chrono::high_resolution_clock::now();
    serial_usage(filename, n, min_dist);
    auto end_ser = std::chrono::high_resolution_clock::now();
    auto ms_ser = std::chrono::duration_cast<std::chrono::milliseconds>(end_ser - start_ser);
    
    auto start_par = std::chrono::high_resolution_clock::now();
    parallel_usage(filename, n, min_dist);
    auto end_par = std::chrono::high_resolution_clock::now();
    auto ms_par = std::chrono::duration_cast<std::chrono::milliseconds>(end_par - start_par);
    
    std::cout << "Serial time: " << ms_ser.count() << " ms" << std::endl;
    std::cout << "Parallel time: " << ms_par.count() << " ms" << std::endl;
    std::cout << "Speedup: " << (double)ms_ser.count() / ms_par.count() << "x" << std::endl;

    return 0;
}
