#include <chrono>

#include "2sat2scc.cu"
#include "2sat_solver_parallel.cu"
#include "2sat_solver_serial.cu"

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

    TwoSat2SCC sccs = TwoSat2SCC(filename);

    bool *par_out_results = (bool*)malloc(n * sccs.n_vars * sizeof(bool));
    int n_out_results = -1;

    int max_threads = get_device_prop(0).maxThreadsPerBlock;
    int max_blocks = get_device_prop(0).maxGridSize[0];
    int sm_count = get_device_prop(0).multiProcessorCount;

    auto start_par = std::chrono::high_resolution_clock::now();
    int par_n_sol = parallel_usage(&sccs, n, min_dist, par_out_results,
                                    max_threads, max_blocks, sm_count);
    auto end_par = std::chrono::high_resolution_clock::now();
    auto ms_par = std::chrono::duration_cast<std::chrono::milliseconds>(end_par - start_par);

    std::cout << "Parallel time: " << ms_par.count() << " ms" << std::endl;

    if (print_sol) {
        printf("Parallel solutions:\n");
        print_array(par_out_results, n_out_results * sccs.n_vars, sccs.n_vars, "sol: ");
    }

    if (!verify_solutions(par_out_results, n_out_results, sccs.vars, sccs.n_vars)) {
        printf("PARALLEL ERROR: Some solutions are invalid.\n");
        return -1;
    } else {
        printf("PARALLEL: All solutions are valid.\n");
    }
    free(par_out_results);

    if (run_serial) {
        bool *ser_out_results = (bool*)malloc(n * sccs.n_vars * sizeof(bool));
        std::cout << std::endl;
        auto start_ser = std::chrono::high_resolution_clock::now();
        int ser_n_sol = serial_usage(&sccs, n, min_dist, ser_out_results);
        auto end_ser = std::chrono::high_resolution_clock::now();
        auto ms_ser = std::chrono::duration_cast<std::chrono::milliseconds>(end_ser - start_ser);
        std::cout << "Serial time: " << ms_ser.count() << " ms" << std::endl;
        std::cout << std::endl << "Speedup: " << (double)ms_ser.count() / ms_par.count() << "x" << std::endl;
        std::cout << "Solutions amount difference: " << par_n_sol - ser_n_sol << std::endl;

        if (print_sol) {
            printf("Serial solutions:\n");
            print_array(par_out_results, ser_n_sol * sccs.n_vars, sccs.n_vars, "sol: ");
        }

        if (!verify_solutions(par_out_results, ser_n_sol, sccs.vars, sccs.n_vars)) {
            printf("SERIAL ERROR: Some solutions are invalid.\n");
            return -1;
        } else {
            printf("SERIAL: All solutions are valid.\n");
        }
        free(ser_out_results);
    }

    return 0;
}
