#include "jrand.cuh"
#include "cudawrapper.cuh"
#include <cstdio>
#include <chrono>


constexpr int FLOWER_PATCH_SALT = 90003;
constexpr int SINGLE_FLOWER_SALT = 90004;

constexpr uint64_t TEXT_SEEDS_TOTAL = 1ULL << 32;

__managed__ uint64_t seedCounter;


__device__ inline bool singleFlowerGenerates(Xoroshiro* xrand, int relativeX, int relativeZ)
{
    // rarity filter (chance = 8)
    float r = xNextFloat(xrand);
    if (!(r < 1.0F / 32.0F)) 
        return false;

    // inSquarePlacement -- test coords
    if (xNextIntJPO2(xrand, 16) != relativeX)
        return false;

    return xNextIntJPO2(xrand, 16) == relativeZ;
}

__device__ inline bool patchGenerates(Xoroshiro* xrand)
{
    // rarity filter (chance = 8)
    float r = xNextFloat(xrand);
    return r < 1.0F / 8.0F;

    // don't care about anything else here
}

__device__ bool conditionalCheck(Xoroshiro* xrand, uint64_t worldseed, const int chunkCoords[])
{
    // if no flower patch in chunks (0), (1), (2) then 
    // there must be a patch in chunk (3)

    #pragma unroll
    for (int i = 0; i < 6; i += 2)
    {
        xSetDecoratorSeed(xrand, worldseed, chunkCoords[i] << 4, chunkCoords[i+1] << 4, FLOWER_PATCH_SALT);
        if (patchGenerates(xrand)) return true; // false -> any <=> true
    }
    
    xSetDecoratorSeed(xrand, worldseed, chunkCoords[6] << 4, chunkCoords[7] << 4, FLOWER_PATCH_SALT);
    return patchGenerates(xrand);
}

__device__ void testWorldseed(uint64_t worldseed)
{
    Xoroshiro xrand = { 0ULL, 0ULL };

    // first filters - boolean flower generation parameters
    // hardcoded for efficiency, keep in mind while debugging

    // in chunk (191, 20): flower patch (or unlikely single flower at 7,8)
    uint64_t popseed = xGetPopulationSeed(&xrand, worldseed, 191 << 4, 20 << 4);
    xSetSeed(&xrand, popseed + FLOWER_PATCH_SALT);

    if (!patchGenerates(&xrand))
    {
        xSetSeed(&xrand, popseed + SINGLE_FLOWER_SALT);
        if (!singleFlowerGenerates(&xrand, 7, 8))
            return;
    }

    // conditional filters

    // if no flower patch in chunks (190,21), (191,21), (191,22) then 
    // in chunk(190, 22) there must be a flower patch
    const int chunks1[] = { 190, 21,   191, 21,   191, 22,   190, 22 };
    if (!conditionalCheck(&xrand, worldseed, chunks1)) return;

    // if no flower patch in chunks (190,19), (190,18), (191,18) then 
    // in chunk(191, 19) : flower patch (or unlikely single flower at 5, 4)
    const int chunks2[] = { 190, 19,   190, 18,   191, 18,   191, 19 };
    if (!conditionalCheck(&xrand, worldseed, chunks2))
    {
        // try single flower (pretty hopeless but whatever)
        xSetDecoratorSeed(&xrand, worldseed, 191 << 4, 19 << 4, SINGLE_FLOWER_SALT);
        if (!singleFlowerGenerates(&xrand, 5, 4)) 
            return;
    }

    // if no flower patch in chunks(188, 20), (188, 19), (189, 19) then
    // in chunk(189, 20) : flower patch (or unlikely single flower at 0, 11)
    const int chunks3[] = { 188, 20,   188, 19,   189, 19,   189, 20 };
    if (!conditionalCheck(&xrand, worldseed, chunks3))
    {
        // try single flower (pretty hopeless but whatever)
        xSetDecoratorSeed(&xrand, worldseed, 189 << 4, 20 << 4, SINGLE_FLOWER_SALT);
        if (!singleFlowerGenerates(&xrand, 0, 11))
            return;
    }

    // if (worldseed % 100000 == 42069)
    //    printSignedSeed(worldseed);

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
