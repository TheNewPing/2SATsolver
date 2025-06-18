#include "../include/cuda_utilities.cu"
#include "../include/literal.cu"
#include "./2sat2scc.cu"

int main(int argc, char** argv) {
    TwoSat2SCC sccs = TwoSat2SCC("error_formulas.txt");
    printf("%d %d", sccs.vars[784*2].value, sccs.vars[784*2+1].value);
    formulas_to_file(sccs.vars, "error_formulas_copy.txt");
}