#pragma once
#include <cstdio>

#define CHECKED_OPERATION(function) \
{ \
    cudaError_t cudaStatus = function; \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(cudaStatus)); \
        return 1; \
    } \
}