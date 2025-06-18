#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <cstring>

#include "../include/cuda_utilities.cu"
#include "../include/literal.cu"
#include "../src/2sat2scc.cu"
#include "../src/2sat_solver_parallel.cu"
#include "../src/2sat_solver_serial.cu"

int parallel_usage(TwoSat2SCC* sccs, int n, int min_dist, bool **out_results, int max_threads, int max_blocks, int sm_count) {
    if (!sccs->build_SCC()) {
        printf("No solution found.\n");
        return -1;
    }

    int n_vars = sccs->n_vars;
    int n_vertices = sccs->n_vertices;

    // prepare output
    *out_results = (bool*)malloc(n * n_vars * sizeof(bool));
    int n_out_results = 0;
    bool init = true;

    // approximate number of solutions to the closest multiple of the number of multiprocessors
    int n_sol = ((n + sm_count - 1) / sm_count) * sm_count;

    int *h_infl_comp;
    int *h_infl_comp_end_idx;
    int *h_comp = sccs->arrayify_comp();
    size_t infl_comp_bytes = sccs->arrayify_infl_comp(&h_infl_comp, &h_infl_comp_end_idx);
    int n_comp = sccs->infl_comp.size();

    while (n_out_results < n) {
        sccs->build_candidates(n_sol, init, min_dist);
        int *h_candidates = sccs->arrayify_candidates();

        // ----------- Compute solutions based on sccs ----------- 
        bool *d_sol_comp;
        compute_sccs_solutions(max_threads, sm_count, n_comp, n_sol, n_vars, n_vertices,
                               h_candidates, h_infl_comp, h_infl_comp_end_idx, infl_comp_bytes,
                               &d_sol_comp);
        free(h_candidates);

        // ----------- Transfer sccs solutions to variable solutions ----------- 
        bool *h_sol_var;
        solutions_sccs_to_vars(max_threads, max_blocks, n_comp, n_sol, n_vars, n_vertices,
                               h_comp, d_sol_comp, &h_sol_var);
        HANDLE_ERROR(cudaFree(d_sol_comp));

        // ----------- Compute compatibility between solutions based on min dist ----------- 
        bool *h_sol_var_min_dist;
        solutions_hamming_dist(max_threads, max_blocks, n_sol, n_vars, min_dist,
                               n_out_results, *out_results, h_sol_var, &h_sol_var_min_dist);

        // ----------- Build the final results -----------
        n_out_results = insert_new_solution(init, n_sol, n_vars, n, h_sol_var, h_sol_var_min_dist,
                                            *out_results, n_out_results);
        free(h_sol_var);
        free(h_sol_var_min_dist);

        init = false;
    }

    free(h_comp);
    free(h_infl_comp);
    free(h_infl_comp_end_idx);

    return n_out_results;
}

int main(int argc, char** argv) {
    if (argc < 8) {
        printf("Usage: %s <repetitions> <n_sol> <2SAT formulas> <min_dist> <new_var_prob> <parallel=1, serial=0> <logfile> [formulafile1 formulafile2 ...]\n", argv[0]);
        return -1;
    }
    int repetitions = atoi(argv[1]);
    int n_sol = atoi(argv[2]);
    int n_2sat_formulas = atoi(argv[3]);
    int min_dist = atoi(argv[4]);
    float new_var_prob = atof(argv[5]);
    int parallel = atoi(argv[6]);
    const char* logfile = argv[7];

    // Collect multiple formulafiles if provided
    std::vector<const char*> formulafiles;

    if (argc > 8) {
        for (int i = 8; i < argc; ++i) {
            struct stat path_stat;
            if (stat(argv[i], &path_stat) == 0 && S_ISDIR(path_stat.st_mode)) {
                // It's a directory, add all .txt files inside
                DIR *dir = opendir(argv[i]);
                if (dir) {
                    struct dirent *entry;
                    while ((entry = readdir(dir)) != NULL) {
                        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
                            const char *dot = strrchr(entry->d_name, '.');
                            if (dot && strcmp(dot, ".txt") == 0) {
                                std::string fullpath = std::string(argv[i]) + "/" + entry->d_name;
                                formulafiles.push_back(strdup(fullpath.c_str()));
                            }
                        }
                    }
                    closedir(dir);
                }
            } else {
                // It's a file, add directly
                formulafiles.push_back(argv[i]);
            }
        }
    }

    // Sort formulafiles by the number before ".txt" if present, otherwise lexicographically
    std::sort(formulafiles.begin(), formulafiles.end(), [](const char* a, const char* b) {
        auto extract_number = [](const char* filename) -> long {
            const char* dot = strrchr(filename, '.');
            if (!dot || strcmp(dot, ".txt") != 0) return -1;
            const char* num_start = dot;
            while (num_start > filename && isdigit(*(num_start - 1))) --num_start;
            if (num_start == dot) return -1;
            char buf[32];
            size_t len = dot - num_start;
            if (len >= sizeof(buf)) return -1;
            strncpy(buf, num_start, len);
            buf[len] = '\0';
            char* endptr;
            long val = strtol(buf, &endptr, 10);
            if (*endptr != '\0') return -1;
            return val;
        };
        long na = extract_number(a);
        long nb = extract_number(b);
        if (na != -1 && nb != -1) return na < nb;
        if (na != -1) return true;
        if (nb != -1) return false;
        return strcmp(a, b) < 0;
    });

    if (repetitions <= 0 || n_sol <= 0 || n_2sat_formulas <= 0 || min_dist < 0 || new_var_prob < 0.0 || new_var_prob > 1.0) {
        printf("Invalid arguments.\n");
        return -1;
    }
    if (min_dist > n_sol) {
        printf("Minimum distance cannot be greater than the number of variables.\n");
        return -1;
    }
    if (parallel != 0 && parallel != 1) {
        printf("Invalid parallel argument. Use 1 for parallel and 0 for serial.\n");
        return -1;
    }
    if (min_dist == 0) {
        printf("Minimum distance cannot be 0.\n");
        return -1;
    }

    // Initialize log.csv with header
    FILE *log_file = fopen(logfile, "r");
    bool file_exists = (log_file != NULL);
    if (log_file) fclose(log_file);

    log_file = fopen(logfile, "a");
    if (log_file == NULL) {
        printf("Failed to open log file for writing.\n");
        return -1;
    }
    if (!file_exists) {
        fprintf(log_file, "n_sol,n_found_sol,n_2sat_formulas,n_vars,min_dist,new_var_prob,repetitions,parallel,duration\n");
    }

    printf("Running benchmark with parameters:\n");
    printf("Repetitions: %d\n", repetitions);
    printf("Number of solutions: %d\n", n_sol);
    printf("Number of 2SAT formulas: %d\n", n_2sat_formulas);
    printf("Minimum distance: %d\n", min_dist);
    printf("New variable probability: %.2f\n", new_var_prob);
    printf("Parallel: %d\n", parallel);
    if (!formulafiles.empty()) {
        printf("Formula files:\n");
        for (int i = 0; i < formulafiles.size(); ++i) {
            printf(" - %s\n", formulafiles[i]);
        }
    }
    printf("Starting benchmark...\n");

    int max_threads = get_device_prop(0).maxThreadsPerBlock;
    int max_blocks = get_device_prop(0).maxGridSize[0];
    int sm_count = get_device_prop(0).multiProcessorCount;
    
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<TwoSat2SCC> sccs_history;

    for (int i = 0; i < repetitions; i++) {
        TwoSat2SCC sccs = formulafiles.empty() ? TwoSat2SCC(&gen, n_2sat_formulas, new_var_prob) : TwoSat2SCC(formulafiles[i % formulafiles.size()]);
        sccs_history.push_back(sccs);
        bool *out_results;
        int n_out_results = -1;
        std::chrono::_V2::system_clock::time_point start_par;
        std::chrono::_V2::system_clock::time_point end_par;

        if (parallel) {
            start_par = std::chrono::high_resolution_clock::now();
            n_out_results = parallel_usage(&sccs, n_sol, min_dist, &out_results,
                                              max_threads, max_blocks, sm_count);
            end_par = std::chrono::high_resolution_clock::now();
        } else {
            // not implemented
        }

        if (!verify_solutions(out_results, n_out_results, sccs.vars, sccs.n_vars)) {
            printf("Some solutions are invalid.\n");
            free(out_results);
            fclose(log_file);
            for (int i = 0; i < sccs_history.size(); ++i) {
                std::ostringstream filename;
                filename << "debug_out/error_formula_" << i << ".txt";
                formulas_to_file(sccs_history[i].vars, filename.str());
            }
            return -1;
        }
        auto ms_par = std::chrono::duration_cast<std::chrono::milliseconds>(end_par - start_par);
        fprintf(log_file, "%d,%d,%d,%d,%d,%.6f,%d,%d,%ld\n",
            n_sol, n_out_results, n_2sat_formulas, sccs.n_vars, min_dist, new_var_prob, repetitions, parallel, ms_par.count());
        fflush(log_file);
        free(out_results);
    }
    fclose(log_file);
    printf("Benchmark completed.\n");

    return 0;
}