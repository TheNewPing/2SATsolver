NVCC := nvcc
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
GEN_DIR := 2cnf_generator
TARGET := $(BIN_DIR)/main
GENERATOR_TARGET := $(BIN_DIR)/2cnf_generator

SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRC_FILES))

CFLAGS := -O3 -Iinclude

all: $(TARGET)

generator: $(GENERATOR_TARGET)

$(TARGET): $(OBJ_FILES) | $(BIN_DIR)
	$(NVCC) $(CFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

$(GENERATOR_TARGET): $(OBJ_DIR)/2cnf_generator.o | $(BIN_DIR)
	$(NVCC) $(CFLAGS) -o $@ $<

$(OBJ_DIR)/2cnf_generator.o: $(GEN_DIR)/2cnf_generator.cu | $(OBJ_DIR)
	$(NVCC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean generator
