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

void assert_compactness(Literal* formulas, int n) {
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
        std::cerr << "Usage: " << argv[0] << " <number_of_2CNF_formulas> <new_variable_probability> <output_filename>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Error: number_of_2CNF_formulas must be a positive integer.\n";
        return 1;
    }
    float new_variable_probability = std::stof(argv[2]);
    if (new_variable_probability < 0.0f || new_variable_probability > 1.0f) {
        std::cerr << "Error: new_variable_probability must be between 0 and 1.\n";
        return 1;
    }
    std::string filename = argv[3];
    if (filename.empty()) {
        std::cerr << "Error: output_filename cannot be empty.\n";
        return 1;
    }

    Literal* h_formulas = (Literal*)malloc(sizeof(Literal) * n * 2);

    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::bernoulli_distribution distrib_sign(0.5);
    std::bernoulli_distribution distrib_new_var(new_variable_probability);
    
    int max_var = 0;
    unsigned int var_name = 0;
    bool sign = distrib_sign(gen);
    h_formulas[0] = Literal(var_name, sign);
    std::uniform_int_distribution<unsigned int> distrib_n(0, max_var);
    for (int i = 1; i < n * 2; ++i) {
        bool new_var = distrib_new_var(gen);
        if (new_var) {
            max_var++;
            distrib_n = std::uniform_int_distribution<unsigned int>(0, max_var);
            var_name = max_var;
        } else {
            var_name = distrib_n(gen);
        }
        sign = distrib_sign(gen);
        h_formulas[i] = Literal(var_name, sign);
    }

    std::ofstream outfile(filename);
    for (int i = 0; i < n; ++i) {
        outfile << h_formulas[i * 2].to_string() << " " << h_formulas[i * 2 + 1].to_string() << "\n";
    }
    outfile.close();

    delete[] h_formulas;

    return 0;
}
