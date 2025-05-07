#include <iostream>

#include "../include/literal.cu"

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

    Literal* h_formulas = generate_2cnf(n, new_variable_probability);
    if (h_formulas == nullptr) {
        std::cerr << "Error: Failed to generate 2CNF formulas.\n";
        return 1;
    }
    
    formulas_to_file(h_formulas, n, filename);

    return 0;
}
