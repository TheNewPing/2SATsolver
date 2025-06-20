NVCC        = nvcc
CFLAGS      = -Xcompiler -fopenmp -lgomp -g -G -Iinclude

SRC_MAIN    = src/main.cu
SRC_BENCH   = src/benchmark.cu
SRC_GEN     = src/2cnf_generator.cu

BIN_MAIN    = bin/main
BIN_BENCH   = bin/benchmark
BIN_GEN     = bin/2cnf_generator

all: $(BIN_MAIN) $(BIN_BENCH) $(BIN_GEN)

$(BIN_MAIN): $(SRC_MAIN)
	$(NVCC) $(CFLAGS) $< -o $@

$(BIN_BENCH): $(SRC_BENCH)
	$(NVCC) $(CFLAGS) $< -o $@

$(BIN_GEN): $(SRC_GEN)
	$(NVCC) $(CFLAGS) $< -o $@

test: $(BIN_MAIN)
	./test_all.sh

clean:
	rm -f $(BIN_MAIN) $(BIN_BENCH) $(BIN_GEN)

.PHONY: all clean test
