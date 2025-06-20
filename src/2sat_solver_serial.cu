#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <fstream>

#include "../include/literal.cu"
#include "./2sat2scc.cu"

struct TwoSatSolverSerial {
    TwoSat2SCC *scc_solver;
    std::vector<bool> assignment;
    std::unordered_set<std::vector<bool>> solutions;

    TwoSatSolverSerial(const std::string& filepath) {
        scc_solver = new TwoSat2SCC(filepath);
        assignment.resize(scc_solver->n_vars, false);
    }

    TwoSatSolverSerial(std::mt19937 *gen, int n, float new_variable_probability=0.9) {
        scc_solver = new TwoSat2SCC(gen, n, new_variable_probability);
        assignment.resize(scc_solver->n_vars, false);
    }

    TwoSatSolverSerial(TwoSat2SCC *scc_solver) {
        this->scc_solver = scc_solver;
        assignment.resize(scc_solver->n_vars, false);
    }
    

    bool solve_2SAT(int start_node = 0) {
        if (!scc_solver->build_SCC(start_node)) {
            return false; // if the 2-SAT problem is unsatisfiable
        }

        assignment.assign(scc_solver->n_vars, false);
        // check if the 2-SAT problem is satisfiable
        for (int i = 0; i < scc_solver->n_vertices; i += 2) {
            assignment[i / 2] = scc_solver->comp[i] > scc_solver->comp[i + 1];
        }
        return true;
    }

    void solve_from_all_nodes(int k = 1, int min_dist = 0) {
        solutions.clear();
        for (int i = 0; i < scc_solver->n_vertices && solutions.size() < k; ++i) {
            solve_2SAT(i);
            bool valid = true;
            for (const auto& sol : solutions) {
                int hamming_dist = 0;
                for (int j = 0; j < scc_solver->n_vars; ++j) {
                    if (sol[j] != assignment[j])
                        ++hamming_dist;
                }
                if (hamming_dist < min_dist) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                solutions.insert(assignment);
            }
        }
    }
};

int serial_usage(TwoSat2SCC* sccs, int max_sol, int min_dist, bool *out_results) {
    if (!sccs->build_SCC()) {
        printf("No solution found.\n");
        return -1;
    }

    TwoSatSolverSerial solver = TwoSatSolverSerial(sccs);
    solver.solve_from_all_nodes(max_sol, min_dist);

    int offset = 0;
    for (const auto& sol : solver.solutions) {
        std::copy(sol.begin(), sol.end(), out_results + offset);
        offset += sol.size();
    }

    return solver.solutions.size();
}
