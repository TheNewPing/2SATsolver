#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <unordered_set>
#include <fstream>

#include "../include/literal.cu"

struct TwoSatSolverLinear {
    int n_vars;
    int n_vertices;
    std::vector<std::vector<int>> adj, adj_t;
    std::vector<bool> used;
    std::vector<int> order, comp;
    std::vector<bool> assignment;
    std::unordered_set<std::vector<bool>> solutions;

    TwoSatSolverLinear(int _n_vars) : n_vars(_n_vars), n_vertices(2 * n_vars), adj(n_vertices), adj_t(n_vertices), used(n_vertices), order(), comp(n_vertices, -1), assignment(n_vars) {
        order.reserve(n_vertices);
    }

    TwoSatSolverLinear(const std::string& filepath) {
        std::ifstream file(filepath);
        std::vector<Literal> vars;
        std::string var1, var2;
        while (file >> var1 >> var2) {
            vars.push_back(Literal(var1));
            vars.push_back(Literal(var2));
        }

        Literal max_var = *max_element(vars.begin(), vars.end());
        n_vars = max_var.value + 1;
        n_vertices = 2 * n_vars;
        adj.resize(n_vertices);
        adj_t.resize(n_vertices);
        used.resize(n_vertices);
        comp.resize(n_vertices, -1);
        assignment.resize(max_var.value);
        order.reserve(n_vertices);
        for (size_t i = 0; i < vars.size(); i += 2) {
            add_disjunction(vars[i], vars[i + 1]);
        }
    }

    void dfs1(int v) {
        used[v] = true;
        for (int u : adj[v]) {
            if (!used[u])
                dfs1(u);
        }
        order.push_back(v);
    }

    void dfs2(int v, int cl) {
        comp[v] = cl;
        for (int u : adj_t[v]) {
            if (comp[u] == -1)
                dfs2(u, cl);
        }
    }

    bool solve_2SAT(int start_node = 0) {
        order.clear();
        used.assign(n_vertices, false);
        // prepare the dfs order starting from the specified node
        dfs1(start_node);
        for (int i = 0; i < n_vertices; ++i) {
            if (!used[i]) // handle the case where the graph is not connected
                dfs1(i);
        }

        comp.assign(n_vertices, -1);
        // identify the strongly connected components and create a topological order
        for (int i = 0, j = 0; i < n_vertices; ++i) {
            int v = order[n_vertices - i - 1];
            if (comp[v] == -1)
                dfs2(v, j++);
        }

        assignment.assign(n_vars, false);
        // check if the 2-SAT problem is satisfiable
        for (int i = 0; i < n_vertices; i += 2) {
            if (comp[i] == comp[i + 1]) // if a variable and its negation are in the same strongly connected component
                return false;
            assignment[i / 2] = comp[i] > comp[i + 1];
        }
        return true;
    }

    void solve_from_all_nodes(int k = 1, int min_dist = 0) {
        solutions.clear();
        for (int i = 0; i < n_vertices && solutions.size() < k; ++i) {
            assert(solve_2SAT(i));
            bool valid = true;
            for (const auto& sol : solutions) {
                int hamming_dist = 0;
                for (int j = 0; j < n_vars; ++j) {
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

    void add_disjunction(Literal var1, Literal var2) {
        unsigned int a = var1.value;
        unsigned int b = var2.value;
        bool na = !var1.isPositive;
        bool nb = !var2.isPositive;
        // Note: remember, the k-th variable is represented by 2 * k and 2 * k + 1 (its negation) 
        // an even number has always the LSB set to 0, so the XOR with 1 is equivalent to adding 1
        a = 2 * a ^ na;
        b = 2 * b ^ nb;
        // Note2: if a variable is negated, then its LSB is set to 1, so the XOR with 1 is equivalent to subtracting 1
        // otherwise, the XOR with 1 is equivalent to adding 1
        int neg_a = a ^ 1;
        int neg_b = b ^ 1;
        adj[neg_a].push_back(b);
        adj[neg_b].push_back(a);
        adj_t[b].push_back(neg_a);
        adj_t[a].push_back(neg_b);
    }

    static void example_usage() {
        TwoSatSolverLinear solver(3); // a, b, c
        solver.add_disjunction(Literal(0, true), Literal(1, false));  //     a  v  not b
        solver.add_disjunction(Literal(0, false), Literal(1, false));   // not a  v  not b
        solver.add_disjunction(Literal(1, true), Literal(2, true)); //     b  v      c
        //solver.add_disjunction(Literal(0, true), Literal(0, true)); //     a  v      a
        solver.solve_from_all_nodes(3, 2);
        for (const auto& sol : solver.solutions) {
            for (bool val : sol) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
};