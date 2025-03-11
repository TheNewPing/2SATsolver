#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <fstream>

#include "../include/literal.cu"
#include "../include/cuda_error.cu"

void add_disjunction(Literal var1, Literal var2, bool** adj, bool** adj_t) {
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
    adj[neg_a][b] = true;
    adj[neg_b][a] = true;
    adj_t[b][neg_a] = true;
    adj_t[a][neg_b] = true;
}

int fill_adjacency_matrix(std::string filepath, bool*** adj, bool*** adj_t) {
    std::ifstream file(filepath);
    std::vector<Literal> vars;
    std::string var1, var2;
    while (file >> var1 >> var2) {
        vars.push_back(Literal(var1));
        vars.push_back(Literal(var2));
    }

    Literal max_var = *max_element(vars.begin(), vars.end());
    size_t n_vars = max_var.value + 1;
    size_t n_vertices = 2 * n_vars;
    *adj = (bool**)malloc(n_vertices * sizeof(bool*));
    *adj_t = (bool**)malloc(n_vertices * sizeof(bool*));
    for (size_t i = 0; i < n_vertices; ++i) {
        (*adj)[i] = (bool*)malloc(n_vertices * sizeof(bool));
        (*adj_t)[i] = (bool*)malloc(n_vertices * sizeof(bool));
        for (size_t j = 0; j < n_vertices; ++j) {
            (*adj)[i][j] = false;
            (*adj_t)[i][j] = false;
        }
    }
    for (size_t i = 0; i < vars.size(); i += 2) {
        add_disjunction(vars[i], vars[i + 1], *adj, *adj_t);
    }
    return n_vertices;
}

__device__ void dfs1(int v, int n_vertices, bool* used, int* order, int order_count, bool** adj) {
    used[v] = true;
    for (int u = 0; u < n_vertices; ++u) {
        if (adj[v][u] && !used[u])
            dfs1(u, n_vertices, used, order, order_count, adj);
    }
    order[order_count++] = v;
}

__device__ void dfs2(int v, int cl, int n_vertices, int* comp, bool** adj_t) {
    comp[v] = cl;
    for (int u = 0; u < n_vertices; ++u) {
        if (adj_t[v][u] && comp[u] == -1)
            dfs2(u, cl, n_vertices, comp, adj_t);
    }
}

__device__ bool solve_2SAT(int n_vars, int n_vertices, bool* used, int* order, int* comp, bool** adj, bool** adj_t, bool* assignment, int start_node = 0) {
    for (int i = 0; i < n_vertices; ++i) {
        order[i] = -1;
        used[i] = false;
        comp[i] = -1;
    }
    // prepare the dfs order starting from the specified node
    dfs1(start_node, n_vertices, used, order, 0, adj);
    for (int i = 0; i < n_vertices; ++i) {
        if (!used[i]) // handle the case where the graph is not connected
            dfs1(i, n_vertices, used, order, 0, adj);
    }

    // identify the strongly connected components and create a topological order
    for (int i = 0, j = 0; i < n_vertices; ++i) {
        int v = order[n_vertices - i - 1];
        if (comp[v] == -1)
            dfs2(v, j++, n_vertices, comp, adj_t);
    }

    for (int i = 0; i < n_vars; ++i) {
        assignment[i] = false;
    }
    // check if the 2-SAT problem is satisfiable
    for (int i = 0; i < n_vertices; i += 2) {
        if (comp[i] == comp[i + 1]) // if a variable and its negation are in the same strongly connected component
            return false;
        assignment[i / 2] = comp[i] > comp[i + 1];
    }
    return true;
}


__global__ void kernel_solve_2SAT(bool** results, bool* solvable, int start_node, int n_vars, bool** adj, bool** adj_t) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int current_vertex = tid + start_node;
    int n_vertices = 2 * n_vars;
    if (current_vertex < n_vertices) {
        bool* assignment = (bool*)malloc(n_vars * sizeof(bool));
        int* order = (int*)malloc(n_vertices * sizeof(int));
        int* comp = (int*)malloc(n_vertices * sizeof(int));
        bool* used = (bool*)malloc(n_vertices * sizeof(bool));
        if (solve_2SAT(n_vars, n_vertices, used, order, comp, adj, adj_t, assignment, current_vertex)) {
            solvable[current_vertex] = true;
            for (int i = 0; i < n_vars; ++i) {
                results[current_vertex][i] = assignment[i];
            }
        } else {
            solvable[current_vertex] = false;
        }
        free(assignment);
        free(order);
        free(comp);
        free(used);
    }
}