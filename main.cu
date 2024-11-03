#include "flowergen.cuh"
#include "cudawrapper.cuh"
#include <cstdio>
#include <chrono>

constexpr uint32_t MIN_X = 0;
constexpr uint32_t MAX_X = 31;
constexpr uint32_t MIN_Z = 0;
constexpr uint32_t MAX_Z = 31;
constexpr uint32_t MIN_Y = 111;
constexpr uint32_t MAX_Y = 112;


constexpr uint32_t SIZE_I = (MAX_X - MIN_X + 32) / 32;  // x
constexpr uint32_t SIZE_J = MAX_Y - MIN_Y + 1;          // y
constexpr uint32_t SIZE_K = MAX_Z - MIN_Z + 1;          // z
__constant__ uint32_t targetPatternFlowers[SIZE_I][SIZE_J][SIZE_K];
__constant__ uint32_t targetPatternAirblocks[SIZE_I][SIZE_J][SIZE_K];

constexpr int MAX_RESULTS_1 = 1024 * 1024;
__device__ uint64_t results1[MAX_RESULTS_1];
__managed__ int resultID1;



__device__ void initialFilter(uint64_t worldseed)
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
    const ChunkPos chunks3[] = { {188, 20},   {188, 19},   {189, 19},   {189, 20} };
    const BlockPos2D flower3 = { 0, 11 };
    if (!testFlowerInChunkConditional(&xrand, worldseed, chunks3, flower3))
        return;

    // TODO add full check

    const int i = atomicAdd(&resultID1, 1);
    if (i >= MAX_RESULTS_1)
        fprintf(stderr, "ERR: too many results from initial filter!\n");

    results1[i] = worldseed;
}

__device__ void secondFilter(uint64_t worldseed)
{
    // the target flower arrangement, as well as the target known air block arrangement,
    // are stored in gpu constant memory as 3D integer arrays. Each bit maps to a certain
    // block coordinate, as shown below:

    // (block Y is the third dimension)
    // -------------------------------------> block X
    // | [0100...10][0100...10][0100...10]
    // | [0100...10][0100...10][0100...10]
    // | ...
    // | [0100...10][0100...10][0100...10]
    // | [0100...10][0100...10][0100...10]
    // v
    // block Z

    // the coordinates are not mapped directly, to access the block at (X, Y, Z), we need
    // to subtract (Xmin, Ymin, Zmin) first and only then access the correct integer's bit.

    // The second filter's job is to generate the flower positions for all chunks that could generate
    // the visible flowers, match the y-heights of the generated flowers to minimize the error rate,
    // and ultimately only leave in seeds for which the minimum error is below an arbitrary threashold.

    // TODO how the fuck do i even start
}

// --------------------------------------------------------------------------------------------

constexpr uint64_t TEXT_SEEDS_TOTAL = 1ULL << 32;

__global__ void crackTextSeedPart1()
{
    uint64_t tid = threadIdx.x + (uint64_t)blockDim.x * blockIdx.x;
    if (tid >= TEXT_SEEDS_TOTAL) return;

    // extend the sign bit if necessary
    uint64_t worldseed = tid;
    if ((worldseed & 0x80000000ULL) != 0ULL)
        worldseed |= 0xffffffff00000000;

    initialFilter(worldseed);
}

__global__ void crackTextSeedPart2()
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= resultID1) return;

    uint64_t worldseed = results1[tid];

    secondFilter(worldseed);
}



int main()
{
    CHECKED_OPERATION( cudaSetDevice(0) );

    auto start = std::chrono::high_resolution_clock::now();

    resultID1 = 0;

    const int THREADS_PER_BLOCK = 512;
    const int NUM_BLOCKS = (TEXT_SEEDS_TOTAL + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    crackTextSeed <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> ();

    CHECKED_OPERATION( cudaGetLastError() );
    CHECKED_OPERATION( cudaDeviceSynchronize() );

    printf("Total candidates: %llu\n", resultID1);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    double ms = (double)elapsed.count() / 1000000.0;
    printf("Kernel took %lf ms\n", ms);

    CHECKED_OPERATION( cudaDeviceReset() );

    return 0;
}
