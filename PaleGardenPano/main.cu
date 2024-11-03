#include "flowergen.cuh"
#include "cudawrapper.cuh"
#include <cstdio>
#include <chrono>

constexpr uint64_t TEXT_SEEDS_TOTAL = 1ULL << 32;
__managed__ uint64_t seedCounter;


__device__ void testWorldseed(uint64_t worldseed)
{
    Xoroshiro xrand = { 0ULL, 0ULL };

    // first filters - boolean flower generation parameters
    // hardcoded for efficiency, keep in mind while debugging

    // in chunk (191, 20): flower patch (or unlikely single flower at 7,8)
    if (!testFlowerInChunkUnconditional(&xrand, worldseed, { 191, 20 }, { 7, 8 }))
        return;

    // conditional filters

    // if no flower patch in chunks (190,22), (191,21), (191,22) then 
    // in chunk (190, 21) there must be a flower patch (or unlikely single flower at 9, 9)
    const ChunkPos chunks1[] = { {190, 22},   {191, 21},   {191, 22},   {190, 21} };
    const BlockPos2D flower1 = { 9, 9 };
    if (!testFlowerInChunkConditional(&xrand, worldseed, chunks1, flower1))
        return;
        
    // if no flower patch in chunks (190,19), (190,18), (191,18) then 
    // in chunk(191, 19) : flower patch (or unlikely single flower at 5, 4)
    const ChunkPos chunks2[] = { {190, 19},   {190, 18},   {191, 18},   {191, 19} };
    const BlockPos2D flower2 = { 5, 4 };
    if (!testFlowerInChunkConditional(&xrand, worldseed, chunks2, flower2))
        return;

    // if no flower patch in chunks(188, 20), (188, 19), (189, 19) then
    // in chunk(189, 20) : flower patch (or unlikely single flower at 0, 11)
    const ChunkPos chunks3[] = { 188, 20,   188, 19,   189, 19,   189, 20 };
    const BlockPos2D flower3 = { 0, 11 };
    if (!testFlowerInChunkConditional(&xrand, worldseed, chunks3, flower3))
        return;

    // TODO add full check

    atomicAdd(&seedCounter, 1);
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

    CHECKED_OPERATION( cudaSetDevice(0) );

    auto start = std::chrono::high_resolution_clock::now();

    seedCounter = 0;

    const int THREADS_PER_BLOCK = 512;
    const int NUM_BLOCKS = (TEXT_SEEDS_TOTAL + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    crackTextSeed <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> ();

    CHECKED_OPERATION( cudaGetLastError() );
    CHECKED_OPERATION( cudaDeviceSynchronize() );

    printf("Total candidates: %llu\n", seedCounter);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    double ms = (double)elapsed.count() / 1000000.0;
    printf("Kernel took %lf ms\n", ms);

    CHECKED_OPERATION( cudaDeviceReset() );

    return 0;
}
