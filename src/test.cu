#include "../include/cuda_utilities.cu"
#include "../include/literal.cu"
#include "./2sat2scc.cu"

int main(int argc, char** argv) {
    std::vector<const char*> formulafiles;
    formulafiles.push_back("t_0.txt");
    formulafiles.push_back("t_1.txt");
    formulafiles.push_back("t_3.txt");
    formulafiles.push_back("t_2.txt");
    for (const auto& f : formulafiles) {
        printf(" - %s\n", f);
    }
}