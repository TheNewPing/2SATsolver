#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <fstream>

#include "../include/literal.cu"

void print_vv(const std::vector<std::vector<int>>& vv) {
    std::cout << "vv:" << std::endl;
    for (const auto& v : vv) {
        for (int val : v) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void print_umap(const std::unordered_map<int, std::unordered_set<int>>& umap) {
    std::cout << "unordered_map:" << std::endl;
    for (const auto& pair : umap) {
        std::cout << pair.first << ": { ";
        for (const int val : pair.second) {
            std::cout << val << " ";
        }
        std::cout << "}" << std::endl;
    }
}


struct TwoSat2SCC {
    int n_vars;
    int n_vertices;
    std::vector<std::vector<int>> adj, adj_t, infl_comp, candidates;
    std::unordered_map<int, std::unordered_set<int>> adj_comp_t;
    std::vector<bool> used;
    std::vector<int> order, comp;

    TwoSat2SCC(int _n_vars) : n_vars(_n_vars), n_vertices(2 * n_vars), adj(n_vertices), adj_t(n_vertices), used(n_vertices), order(), comp(n_vertices, -1) {
        order.reserve(n_vertices);
    }

    TwoSat2SCC(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::in);
        std::vector<Literal> vars;
        std::string var1, var2;
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filepath);
        }
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
        order.reserve(n_vertices);
        for (size_t i = 0; i < vars.size(); i += 2) {
            add_disjunction(vars[i], vars[i + 1]);
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
        return true;
    }

    void build_candidates(int max_solutions) {
        candidates.clear();
        std::unordered_map<int, std::unordered_set<int>> adj_comp_t_copy = adj_comp_t;
        std::vector<int> c;
        // Create the first candidate
        while (!adj_comp_t_copy.empty()) {
            for (auto it = adj_comp_t_copy.cbegin(); it != adj_comp_t_copy.cend();) {
                if (it->second.empty()) {
                    int val = it->first;
                    // Remove the element at the same index from adj_comp_t_copy
                    adj_comp_t_copy.erase(it++);
                    for (auto jt = adj_comp_t_copy.begin(); jt != adj_comp_t_copy.end(); ++jt) {
                        jt->second.erase(val);
                    }
                    c.push_back(val);
                }
                else {
                    ++it;
                }
            }
        }
        candidates.push_back(c);
        // Generate all candidates by swapping elements starting from the first one
        while (candidates.size() < max_solutions) {
            for (int i = 0; i < c.size() - 1; ++i) {
                for (int j = i + 1; j < c.size(); ++j) {
                    std::swap(c[i], c[j]);
                    bool valid = true;
                    // Check if the new candidate is valid
                    for (int k = 0; k < j; ++k) {
                        if (adj_comp_t[c[k]].find(c[j]) != adj_comp_t[c[k]].end()) {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) {
                        candidates.push_back(c);
                        // Check if we have reached the maximum number of solutions
                        if (candidates.size() >= max_solutions) {
                            return;
                        }
                    } else {
                        // If the candidate is not valid, swap back
                        std::swap(c[i], c[j]);
                    }
                }
            }
        }
    }

    std::vector<std::vector<int>> _build_candidates_rec(const std::unordered_map<int, std::unordered_set<int>>& adj_comp_tr, int max_solutions) {
        print_umap(adj_comp_tr);
        std::vector<std::vector<int>> out;
        for (auto pair : adj_comp_tr) {
            if (pair.second.empty()) {
                // righe dovbrebbero essere named?
                int val = pair.first;
                std::unordered_map<int, std::unordered_set<int>> adj_comp_t_copy = adj_comp_tr;
                // Remove the element at the same index from adj_comp_t_copy
                adj_comp_t_copy.erase(val);
                for (auto& vec : adj_comp_t_copy) {
                    vec.second.erase(val);
                }
                std::vector<std::vector<int>> candidates_rec;
                if (adj_comp_t_copy.size() > 1) {
                    candidates_rec = _build_candidates_rec(adj_comp_t_copy, max_solutions);
                } else {
                    candidates_rec.push_back({adj_comp_tr.begin()->first});
                }
                for (const auto& candidate : candidates_rec) {
                    std::vector<int> new_candidate = {val};
                    new_candidate.insert(new_candidate.end(), candidate.begin(), candidate.end());
                    out.push_back(new_candidate);
                    if (out.size() >= max_solutions) {
                        return out;
                    }
                }
            }
        }
        return out;
    }

    int* arrayify_candidates() {
        int* candidates_array = new int[candidates.size() * candidates[0].size()];
        for (size_t i = 0; i < candidates.size(); ++i) {
            std::copy(candidates[i].begin(), candidates[i].end(), candidates_array + i * candidates[0].size());
        }
        return candidates_array;
    }

    bool* arrayify_infl_comp() {
        bool* infl_comp_array = new bool[infl_comp.size() * infl_comp.size()];
        std::fill(infl_comp_array, infl_comp_array + infl_comp.size() * infl_comp.size(), false);
        for (size_t i = 0; i < infl_comp.size(); ++i) {
            for (int val : infl_comp[i]) {
                infl_comp_array[i * infl_comp.size() + val] = true;
            }
        }
        return infl_comp_array;
    }

    int* arrayify_comp() {
        int* comp_array = new int[comp.size()];
        std::copy(comp.begin(), comp.end(), comp_array);
        return comp_array;
    }
};
