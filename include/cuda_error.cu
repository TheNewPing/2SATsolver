#ifndef __CUDA_ERROR_CU__
#define __CUDA_ERROR_CU__

#include <stdio.h>

static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line ); 
        exit( EXIT_FAILURE );
    }
}
       
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError(); 
    if( cudaSuccess != err) {
fprintf(stderr, "CUDA ERROR: >%s<: >%s<. Executing: EXIT\n", msg, cudaGetErrorString(err) ); exit(-1);
    }
}

#endif // __CUDA_ERROR_CU__