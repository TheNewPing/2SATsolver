#ifndef __LITERAL_CU__
#define __LITERAL_CU__

// literal.cu

#include <iostream>
#include <random>
#include <string>
#include <fstream>

struct Literal {
    unsigned int value;
    bool isPositive;

    // Constructor
    __host__ __device__ Literal(unsigned int v, bool isPos) : value(v), isPositive(isPos) {}

    // Constructor from string
    Literal(const std::string& str) {
        if (str[0] == '_') {
            isPositive = false;
            value = static_cast<unsigned int>(std::stoi(str.substr(1)));
        } else {
            isPositive = true;
            value = static_cast<unsigned int>(std::stoi(str));
        }
    }

    // to_string method
    __host__ std::string to_string() const {
        return isPositive ? std::to_string(value) : "_" + std::to_string(value);
    }

    // LessThanComparableConcept
    __host__ __device__ bool operator<(const Literal& other) const {
        return value < other.value;
    }
};


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

bool assert_compactness(Literal* formulas, int n) {
    unsigned int max_var = 0;
    for (int i = 0; i < n * 2; ++i) {
        if (formulas[i].value > max_var) {
            max_var = formulas[i].value;
        }
    }
    for (int var_name = 0; var_name <= max_var; ++var_name) {
        if(!is_used(formulas, n, var_name)) {
            printf("ERROR: Variable %d is STILL not used\n", var_name);
            return false;
        }
    }
    return true;
}

void print_array(Literal* array, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << array[i].to_string() << " ";
    }
    std::cout << std::endl;
}

Literal* generate_2cnf(std::mt19937 *gen, int n, float new_variable_probability) {
    Literal* h_formulas = (Literal*)malloc(sizeof(Literal) * n * 2);

    std::bernoulli_distribution distrib_sign(0.5);
    std::bernoulli_distribution distrib_new_var(new_variable_probability);
    
    int max_var = 0;
    unsigned int var_name = 0;
    bool sign = distrib_sign(*gen);
    h_formulas[0] = Literal(var_name, sign);
    std::uniform_int_distribution<unsigned int> distrib_n(0, max_var);
    for (int i = 1; i < n * 2; ++i) {
        bool new_var = distrib_new_var(*gen);
        if (new_var) {
            max_var++;
            distrib_n = std::uniform_int_distribution<unsigned int>(0, max_var);
            var_name = max_var;
        } else {
            var_name = distrib_n(*gen);
        }
        sign = distrib_sign(*gen);
        h_formulas[i] = Literal(var_name, sign);
    }
    
    return h_formulas;
}

void formulas_to_file(Literal* h_formulas, int n, const std::string& filename) {
    std::ofstream outfile(filename);
    for (int i = 0; i < n; ++i) {
        outfile << h_formulas[i * 2].to_string() << " " << h_formulas[i * 2 + 1].to_string() << "\n";
    }
    outfile.close();
}

void formulas_to_file(std::vector<Literal>& formulas, const std::string& filename) {
    std::ofstream outfile(filename);
    for (int i = 0; i < formulas.size()/2; ++i) {
        outfile << formulas[i * 2].to_string() << " " << formulas[i * 2 + 1].to_string() << "\n";
    }
    outfile.close();
}

#endif // __LITERAL_CU__
