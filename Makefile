NVCC        = nvcc
CFLAGS      = -Xcompiler -fopenmp -lgomp -g -G -Iinclude

SRC_MAIN    = src/main.cu
SRC_BENCH   = src/benchmark.cu
SRC_GEN     = src/2cnf_generator.cu

BIN_DIR     = bin
BIN_MAIN    = $(BIN_DIR)/main
BIN_BENCH   = $(BIN_DIR)/benchmark
BIN_GEN     = $(BIN_DIR)/2cnf_generator

all: $(BIN_MAIN) $(BIN_BENCH) $(BIN_GEN)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_MAIN): $(SRC_MAIN) | $(BIN_DIR)
	$(NVCC) $(CFLAGS) $< -o $@

$(BIN_BENCH): $(SRC_BENCH) | $(BIN_DIR)
	$(NVCC) $(CFLAGS) $< -o $@

$(BIN_GEN): $(SRC_GEN) | $(BIN_DIR)
	$(NVCC) $(CFLAGS) $< -o $@

test: $(BIN_MAIN)
	./test_all.sh

clean:
	rm -f $(BIN_MAIN) $(BIN_BENCH) $(BIN_GEN)

.PHONY: all clean test
