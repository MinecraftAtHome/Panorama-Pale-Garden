#pragma once
#include <cstdio>

#define DEBUG 0


#define CHECKED_OPERATION(function) \
{ \
    cudaError_t cudaStatus = function; \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR (%s, line %d) : %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus)); \
        return 1; \
    } \
}

#define HOST_ERROR(msg) \
{ \
	fprintf(stderr, "HOST ERROR (%s, line %d) : %s\n", __FILE__, __LINE__, msg); \
	return 1; \
}

#define DEBUG_PRINT(...) \
{ \
	if (DEBUG) printf(__VA_ARGS__); \
}