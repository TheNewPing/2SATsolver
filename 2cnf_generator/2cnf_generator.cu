#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include <ctime>
#include <cassert>
#include <random>

#include "../include/cuda_error.cu"
#include "../include/cuda_utilities.cu"
#include "../include/literal.cu"

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

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <number_of_2CNF_formulas> <max_number_of_variables> <output_filename>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    int num_vars = std::stoi(argv[2]);
    std::string filename = argv[3];

    Literal* h_formulas = (Literal*)malloc(sizeof(Literal) * n * 2);

    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib_n(0, n-1);
    std::uniform_int_distribution<> distrib_sign(0, 1);
    
    for (int i = 0; i < n * 2; ++i) {
        unsigned int var_name = distrib_n(gen);
        bool sign = distrib_sign(gen);
        h_formulas[i] = Literal(var_name, sign);
    }

    compact_formulas(h_formulas, n, num_vars);
    assert_compactness(h_formulas, n, num_vars);

    std::ofstream outfile(filename);
    for (int i = 0; i < n; ++i) {
        outfile << h_formulas[i * 2].to_string() << " " << h_formulas[i * 2 + 1].to_string() << "\n";
    }
    outfile.close();

    delete[] h_formulas;

    return 0;
}
