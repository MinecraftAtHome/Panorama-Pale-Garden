#include "flowergen.cuh"
#include "cudawrapper.cuh"
#include <cstdio>
#include <chrono>

constexpr uint64_t TEXT_SEEDS_TOTAL = 1ULL << 32;
__managed__ uint64_t seedCounter;


__device__ void testWorldseed(uint64_t worldseed)
{
    Xoroshiro xrand = { 0ULL, 0ULL };

    if (!testFlowerInChunkUnconditional(&xrand, worldseed, { 630, 478 }, { 7, 8 }))
        return;

    // conditional filters

    const ChunkPos chunks1[] = { {626, 479},   {627, 479},   {627, 480},   {626, 480} };
    const BlockPos2D flower1 = { 11, 5 };
    if (!testFlowerInChunkConditional(&xrand, worldseed, chunks1, flower1))
        return;

    const ChunkPos chunks2[] = { {627, 477},   {627, 477},   {627, 477},   {627, 478} };
    const BlockPos2D flower2 = { 7, 2 };
    if (!testFlowerInChunkConditional(&xrand, worldseed, chunks2, flower2))
        return;

    const ChunkPos chunks3[] = { {632, 472},   {632, 472},   {632, 472},   {631, 472} };
    const BlockPos2D flower3 = { 15, 7 };
    if (!testFlowerInChunkConditional(&xrand, worldseed, chunks3, flower3))
        return;

    // TODO add full check

    atomicAdd(&seedCounter, 1);

    if (worldseed == (uint64_t)(-15378126LL))
    {
        printf("Found good seed!!!\n");
    }
}

__global__ void crackTextSeed()
{
    uint64_t tid = threadIdx.x + (uint64_t)blockDim.x * blockIdx.x;
    if (tid >= TEXT_SEEDS_TOTAL) return;

    // extend the sign bit if necessary
    uint64_t worldseed = tid;
    if ((worldseed & 0x80000000ULL) != 0ULL)
        worldseed |= 0xffffffff00000000;

    testWorldseed(worldseed);
}



int main()
{
    int err = 1;

    CHECKED_OPERATION(cudaSetDevice(0));

    auto start = std::chrono::high_resolution_clock::now();

    seedCounter = 0;

    const int THREADS_PER_BLOCK = 512;
    const int NUM_BLOCKS = (TEXT_SEEDS_TOTAL + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    crackTextSeed << < NUM_BLOCKS, THREADS_PER_BLOCK >> > ();

    CHECKED_OPERATION(cudaGetLastError());
    CHECKED_OPERATION(cudaDeviceSynchronize());

    printf("Total candidates: %llu\n", seedCounter);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    double ms = (double)elapsed.count() / 1000000.0;
    printf("Kernel took %lf ms\n", ms);

    CHECKED_OPERATION(cudaDeviceReset());

    return 0;
}
