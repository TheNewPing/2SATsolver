#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include <ctime>
#include <cassert>

#include "../include/cuda_error.cu"
#include "../include/cuda_utilities.cu"
#include "../include/literal.cu"

__global__ void generate2CNF(Literal* formulas, int n, int num_vars, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        unsigned int var1 = (curand(&state) % num_vars);
        unsigned int var2 = (curand(&state) % num_vars);
        bool sign1 = (curand(&state) % 2);
        bool sign2 = (curand(&state) % 2);
        formulas[idx * 2] = Literal(var1, sign1);
        formulas[idx * 2 + 1] = Literal(var2, sign2);
    }
}

bool is_used(Literal* formulas, int n, unsigned int var_name) {
    for (int i = 0; i < n*2; ++i) {
        if (formulas[i].value == var_name) {
            return true;
        }
    }
    return false;
}

void compact_formulas(Literal* formulas, int n, int num_vars) {
    unsigned int var_name = 0;
    bool modified = true;
    while (var_name < num_vars && modified) {
        if (!is_used(formulas, n, var_name)) {
            modified = false;
            for (int i = 0; i < n*2; ++i) {
                if (formulas[i].value > var_name) {
                    formulas[i].value -= 1;
                    modified = true;
                }
            }
        } else {
            var_name += 1;
        }
    }
}

void assert_compactness(Literal* formulas, int n, int num_vars) {
    unsigned int max_var = 0;
    for (int i = 0; i < n * 2; ++i) {
        if (formulas[i].value > max_var) {
            max_var = formulas[i].value;
        }
    }
    for (int var_name = 0; var_name <= max_var; ++var_name) {
        if(!is_used(formulas, n, var_name)) {
            printf("ERROR: Variable %d is STILL not used\n", var_name);
            assert(false);
        }
    }
}

void print_array(Literal* array, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << array[i].to_string() << " ";
    }
    std::cout << std::endl;
}

int main() {
    int n, num_vars;
    std::cout << "Enter the number of 2-CNF formulas: ";
    std::cin >> n;
    std::cout << "Enter the maximum number of variables: ";
    std::cin >> num_vars;
    std::string filename;
    std::cout << "Enter the output filename: ";
    std::cin >> filename;

    Literal* d_formulas;
    Literal* h_formulas = (Literal*)malloc(sizeof(Literal) * n * 2);
    int threads_per_block = get_device_prop(0).maxThreadsPerBlock;

    HANDLE_ERROR(cudaMalloc(&d_formulas, n * 2 * sizeof(Literal)));
    unsigned long long seed = time(0);
    generate2CNF<<<(n + threads_per_block - 1) / threads_per_block, threads_per_block>>>(d_formulas, n, num_vars, seed);
    cudaDeviceSynchronize();
    checkCUDAError("2CNF generation");
    HANDLE_ERROR(cudaMemcpy(h_formulas, d_formulas, n * 2 * sizeof(Literal), cudaMemcpyDeviceToHost));

    compact_formulas(h_formulas, n, num_vars);
    assert_compactness(h_formulas, n, num_vars);

    std::ofstream outfile(filename);
    for (int i = 0; i < n; ++i) {
        outfile << h_formulas[i * 2].to_string() << " " << h_formulas[i * 2 + 1].to_string() << "\n";
    }
    outfile.close();

    HANDLE_ERROR(cudaFree(d_formulas));
    delete[] h_formulas;

    return 0;
}
