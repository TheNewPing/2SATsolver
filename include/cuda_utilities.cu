#ifndef __CUDA_UTILITIES_CU__
#define __CUDA_UTILITIES_CU__

#include <stdio.h>
#include <iostream>
#include <vector>
#include <unordered_set>
#include "./cuda_error.cu"    

cudaDeviceProp get_device_prop(int i) {
    cudaDeviceProp prop;
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
    return prop;
}

void print_prop_summary(int i) {
    cudaDeviceProp prop = get_device_prop(i);
    printf( " --- General Information for device %d ---\n", i );
    printf( "Name: %s\n", prop.name );
    printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
    printf( "Clock rate: %d\n", prop.clockRate );
    printf( "Device copy overlap: " );
    if (prop.deviceOverlap) printf( "Enabled\n" );
    else printf( "Disabled\n");
    printf( "Kernel execution timeout : " ); 
    if (prop.kernelExecTimeoutEnabled) printf( "Enabled\n" );
    else printf( "Disabled\n" );
    printf( " --- Memory Information for device %d ---\n", i );
    printf( "Total global mem: %ld\n", prop.totalGlobalMem );
    printf( "Total constant Mem: %ld\n", prop.totalConstMem );
    printf( "Max mem pitch: %ld\n", prop.memPitch );
    printf( "Texture Alignment: %ld\n", prop.textureAlignment );
    printf( " --- MP Information for device %d ---\n", i );
    printf( "Multiprocessor count: %d\n", prop.multiProcessorCount );
    printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
    printf( "Registers per mp: %d\n", prop.regsPerBlock );
    printf( "Threads in warp: %d\n", prop.warpSize );
    printf( "Max threads per block: %d\n", prop.maxThreadsPerBlock );
    printf( "Max thread dimensions: (%d, %d, %d)\n",
    prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
    printf( "Max grid dimensions: (%d, %d, %d)\n",
    prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
    printf( "\n" );
}

void set_heap_size(size_t max_heap_size) {
    size_t free_mem, total_mem;
    HANDLE_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    if (max_heap_size > free_mem) {
        std::cerr << "Not enough memory on device. Required: " << max_heap_size / (1024 * 1024) << " MB, Available: " << free_mem / (1024 * 1024) << " MB" << std::endl;
        return;
    }
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, max_heap_size);
    checkCUDAError("Heap size set");
}

__device__ __host__ void print_array(int* array, int length, int values_per_row, const char* row_prefix) {
    for (int i = 0; i < length; ++i) {
        if (i % values_per_row == 0) {
            printf("%s", row_prefix);
        }
        printf("%d ", array[i]);
        if ((i + 1) % values_per_row == 0) {
            printf("\n");
        }
    }
    if (length % values_per_row != 0) {
        printf("\n");
    }
}
__device__ __host__ void print_array(bool* array, int length, int values_per_row, const char* row_prefix = "") {
    for (int i = 0; i < length; ++i) {
        if (i % values_per_row == 0) {
            printf("%s", row_prefix);
        }
        printf("%d ", array[i]);
        if ((i + 1) % values_per_row == 0) {
            printf("\n");
        }
    }
    if (length % values_per_row != 0) {
        printf("\n");
    }
}

__device__ __host__ void print_array(int* array, int num_rows, int* row_end_indices, const char* row_prefix = "") {
    int start_index = 0;
    for (int row = 0; row < num_rows; ++row) {
        printf("%s", row_prefix);
        for (int index = start_index; index < row_end_indices[row]; ++index) {
            printf("%d ", array[index]);
        }
        printf("\n");
        start_index = row_end_indices[row];
    }
}

bool verify_solutions(bool* out_results, int n_out_results, std::vector<Literal>& vars, int n_vars) {
    bool valid = true;
    for (int i = 0; i < n_out_results; ++i) {
        for (int j = 0; j < vars.size(); j+=2) {
            bool sign1 = vars[j].isPositive;
            bool sign2 = vars[j + 1].isPositive;
            int var1 = vars[j].value;
            int var2 = vars[j + 1].value;
            bool val1 = out_results[i * n_vars + var1] == sign1;
            bool val2 = out_results[i * n_vars + var2] == sign2;
            if (!(val1 || val2)) {
                printf("ERROR: solution %d is not valid.", i);
                print_array(out_results + i * n_vars, n_vars, n_vars, "sol: ");
                valid = false;
            }
        }
    }
    return valid;
}

bool verify_solutions(std::unordered_set<std::vector<bool>> solutions, std::vector<Literal>& vars) {
    bool valid = true;
    int i = 0;
    for (const auto& sol : solutions) {
        for (int j = 0; j < vars.size(); j+=2) {
            bool sign1 = vars[j].isPositive;
            bool sign2 = vars[j + 1].isPositive;
            int var1 = vars[j].value;
            int var2 = vars[j + 1].value;
            bool val1 = sol[var1] == sign1;
            bool val2 = sol[var2] == sign2;
            if (!(val1 || val2)) {
                printf("ERROR: solution %d is not valid.", i);
                std::cout << "sol: ";
                for (bool val : sol) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
                valid = false;
            }
        }
        ++i;
    }
    return valid;
}

#endif // __CUDA_UTILITIES_CU__
