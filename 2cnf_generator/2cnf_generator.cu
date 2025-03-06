#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include <ctime>
#include <cassert>

#include "../include/cuda_error.cu"
#include "../include/cuda_utilities.cu"

__global__ void generate2CNF(int* formulas, int n, int num_vars, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        int var1 = (curand(&state) % num_vars) + 1;
        int var2 = (curand(&state) % num_vars) + 1;
        int sign1 = (curand(&state) % 2) ? 1 : -1;
        int sign2 = (curand(&state) % 2) ? 1 : -1;
        formulas[idx * 2] = sign1 * var1;
        formulas[idx * 2 + 1] = sign2 * var2;
    }
}

bool is_used(int* formulas, int n, int var_name) {
    for (int i = 0; i < n*2; ++i) {
        if (formulas[i] == var_name || formulas[i] == -var_name) {
            return true;
        }
    }
    return false;
}

void compact_formulas(int* formulas, int n, int num_vars) {
    int var_name = 1;
    bool modified = true;
    while (var_name <= num_vars && modified) {
        if (!is_used(formulas, n, var_name)) {
            modified = false;
            for (int i = 0; i < n*2; ++i) {
                if (formulas[i] > var_name) {
                    formulas[i] -= 1;
                    modified = true;
                } else if (formulas[i] < -var_name) {
                    formulas[i] += 1;
                    modified = true;
                }
            }
        } else {
            var_name += 1;
        }
    }
}

void assert_compactness(int* formulas, int n, int num_vars) {
    int max_var = 0;
    int min_var = num_vars + 1;
    for (int i = 0; i < n * 2; ++i) {
        if (formulas[i] > max_var) {
            max_var = formulas[i];
        }
        if (formulas[i] < min_var) {
            min_var = formulas[i];
        }
    }
    int limit_var = max_var > -min_var ? max_var : -min_var;
    for (int var_name = 1; var_name <= limit_var; ++var_name) {
        if(!is_used(formulas, n, var_name)) {
            printf("ERROR: Variable %d is STILL not used\n", var_name);
            assert(false);
        }
    }
}

void print_array(int* array, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    int n, num_vars;
    std::cout << "Enter the number of 2-CNF formulas: ";
    std::cin >> n;
    std::cout << "Enter the maximum number of variables: ";
    std::cin >> num_vars;

    int* d_formulas;
    int* h_formulas = new int[n * 2];
    int threads_per_block = get_device_prop(0).maxThreadsPerBlock;

    HANDLE_ERROR(cudaMalloc(&d_formulas, n * 2 * sizeof(int)));
    unsigned long long seed = time(0);
    generate2CNF<<<(n + threads_per_block - 1) / threads_per_block, threads_per_block>>>(d_formulas, n, num_vars, seed);
    cudaDeviceSynchronize();
    checkCUDAError("2CNF generation");
    HANDLE_ERROR(cudaMemcpy(h_formulas, d_formulas, n * 2 * sizeof(int), cudaMemcpyDeviceToHost));

    compact_formulas(h_formulas, n, num_vars);
    assert_compactness(h_formulas, n, num_vars);

    std::ofstream outfile("2cnf_formulas.txt");
    for (int i = 0; i < n; ++i) {
        outfile << h_formulas[i * 2] << " " << h_formulas[i * 2 + 1] << "\n";
    }
    outfile.close();

    HANDLE_ERROR(cudaFree(d_formulas));
    delete[] h_formulas;

    return 0;
}
