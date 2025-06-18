#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <random>
#include <omp.h>

#include "../include/literal.cu"
#include "../include/cuda_utilities.cu"

struct TwoSat2SCC {
    int n_vars;
    int n_vertices;
    std::vector<std::vector<int>> adj, adj_t, infl_comp, candidates;
    std::unordered_map<int, std::unordered_set<int>> adj_comp_t;
    std::vector<bool> used;
    std::vector<int> order, comp;
    std::vector<Literal> vars;
    std::uniform_int_distribution<> distrib;

    void init() {
        Literal max_var = *max_element(vars.begin(), vars.end());
        n_vars = max_var.value + 1;
        n_vertices = 2 * n_vars;
        adj.resize(n_vertices);
        adj_t.resize(n_vertices);
        used.resize(n_vertices);
        comp.resize(n_vertices, -1);
        order.resize(n_vertices);
        for (size_t i = 0; i < vars.size(); i += 2) {
            add_disjunction(vars[i], vars[i + 1]);
        }
    }

    void clear_all() {
        adj.clear();
        adj_t.clear();
        infl_comp.clear();
        candidates.clear();
        adj_comp_t.clear();
        used.clear();
        order.clear();
        comp.clear();
        vars.clear();
    }

    TwoSat2SCC(Literal* formulas, int n) {
        for (int i = 0; i < n * 2; ++i) {
            vars.push_back(formulas[i]);
        }
        init();
    }

    TwoSat2SCC(std::mt19937 *gen, int n, float new_variable_probability=0.9) {
        Literal *formulas = nullptr;
        do {
            if (formulas) {
                free(formulas);
                clear_all();
            }
            formulas = generate_2cnf(gen, n, new_variable_probability);
            for (int i = 0; i < n * 2; ++i) {
                vars.push_back(formulas[i]);
            }
            init();
        } while (!build_SCC());
    }

    TwoSat2SCC(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::in);
        std::string var1, var2;
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filepath);
        }
        while (file >> var1 >> var2) {
            vars.push_back(Literal(var1));
            vars.push_back(Literal(var2));
        }
        init();
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
            else if (comp[u] != cl)
                // if u is already in a component, cl is adjacent to the component of u
                adj_comp_t[cl].insert(comp[u]);
        }
    }

    bool build_SCC(int start_node = 0) {
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
            if (comp[v] == -1) {
                adj_comp_t[j] = std::unordered_set<int>();
                dfs2(v, j++);
            }
        }

        infl_comp.resize(adj_comp_t.size());
        for (int i = 0; i < n_vertices; i += 2) {
            if (comp[i] == comp[i + 1]) // if a variable and its negation are in the same strongly connected component
                return false;
            infl_comp[comp[i]].push_back(comp[i + 1]);
            infl_comp[comp[i + 1]].push_back(comp[i]);
        }

        distrib = std::uniform_int_distribution<>(0, infl_comp.size() - 1);
        return true;
    }

    // Precondition: build_SCC() has been called and returned true
    void build_candidates(int max_solutions, bool init, int iterations = 1) {
        std::vector<int> c;
        int init_ctr = 0;
        if (init) {
            candidates.resize(max_solutions);
            c.resize(infl_comp.size());
            std::iota(c.begin(), c.end(), 0);
            candidates[init_ctr++] = c;
        } else {
            c = std::vector<int>(candidates[max_solutions - 1]);
        }
        // Generate all candidates by swapping elements starting from the first one
        #pragma omp parallel for \
            schedule(auto) \
            firstprivate(c, distrib) \
            shared(candidates, adj_comp_t)
        for (int ctr = init_ctr; ctr < max_solutions; ++ctr) {
            for (int itr = 0; itr < iterations; ++itr) {
                std::random_device rd;
                std::mt19937 gen(rd());
                bool valid = false;
                while (!valid) {
                    int i, j;
                    do {
                        i = distrib(gen);
                        j = distrib(gen);
                    } while (i == j);
                    if (i > j) {
                        // this is needed later to check if the candidate is valid
                        std::swap(i, j);
                    }
                    std::swap(c[i], c[j]);

                    // Check if the new candidate is valid
                    valid = true;
                    // the component moved forward (c[j]) should not be adjacent to any component in the range [i,j-1]
                    for (int k = i; k < j; ++k) {
                        if (adj_comp_t[c[k]].find(c[j]) != adj_comp_t[c[k]].end()) {
                            valid = false;
                            break;
                        }
                    }
                    // any component in the range [i+1,j] should not be adjacent to the component moved backward (c[i])
                    if (valid) {
                        for (int k = j; k > i; --k) {
                            if (adj_comp_t[c[i]].find(c[k]) != adj_comp_t[c[i]].end()) {
                                valid = false;
                                break;
                            }
                        }
                    }
                    if (!valid){
                        // If the candidate is not valid, swap back
                        std::swap(c[i], c[j]);
                    }
                }
            }
            candidates[ctr] = c;
        }
    }

    int* arrayify_candidates() {
        int* candidates_array = (int*)malloc(candidates.size() * candidates[0].size() * sizeof(int));
        if (!candidates_array) {
            std::cerr << "Memory allocation failed for candidates_array." << std::endl;
            return nullptr;
        }
        for (size_t i = 0; i < candidates.size(); ++i) {
            std::copy(candidates[i].begin(), candidates[i].end(), candidates_array + i * candidates[0].size());
        }
        return candidates_array;
    }

    size_t arrayify_infl_comp(int** infl_comp_out, int** end_indexes) {
        size_t tot_size = 0;
        for (auto val : infl_comp) {
            tot_size += val.size() * sizeof(int);
        }
        *infl_comp_out = (int*)malloc(tot_size);
        *end_indexes = (int*)malloc(infl_comp.size() * sizeof(int));
        int offset = 0;
        for (size_t i = 0; i < infl_comp.size(); ++i) {
            std::copy(infl_comp[i].begin(), infl_comp[i].end(), (*infl_comp_out) + offset);
            offset += infl_comp[i].size();
            (*end_indexes)[i] = offset;
        }
        return tot_size;
    }

    int* arrayify_comp() {
        int* comp_array = (int*)malloc(comp.size() * sizeof(int));
        if (!comp_array) {
            std::cerr << "Memory allocation failed for comp_array." << std::endl;
            return nullptr;
        }
        std::copy(comp.begin(), comp.end(), comp_array);
        return comp_array;
    }
};
