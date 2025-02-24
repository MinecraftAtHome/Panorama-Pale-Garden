#include "flowergen.h"
#include "treegen.h"
#include "cudawrapper.h"
#include <cstdio>
#include <chrono>
#include <cmath>

// test units
//#define TREE_FILE "data/tests-trees/t2.txt"
#define TREE_FILE "data/tests-trees/t3.txt"
//#define TREE_FILE "data/tests-trees/t4.txt"

// filter data
constexpr int MAX_PALE_OAKS = 32;
__constant__ PaleOakTree targetTrees[MAX_PALE_OAKS];
__constant__ int targetTreeCount;

// for initial filter
constexpr int MAX_RESULTS_1 = 1024 * 1024;
__managed__ uint64_t results1[MAX_RESULTS_1];
__managed__ int resultID1;

// --------------------------------------------------------------------------------------------

__device__ static inline bool testFlowers(const SeedConstants& sc)
{
    Xoroshiro xrand = { 0ULL, 0ULL };

    // test 2
    //{
    //    const ChunkPos chunks[] = QUAD_CHUNK(-50, 88, -1, -1);
    //    const BlockPos2D flower = { 3, 0 };
    //    if (!testFlowerInChunkConditional(&xrand, sc, chunks, flower))
    //        return false;
    //}
    //{
    //    const ChunkPos chunks[] = QUAD_CHUNK(-52, 90, 1, 1);
    //    const BlockPos2D flower = { 15, 10 };
    //    if (!testFlowerInChunkConditional(&xrand, sc, chunks, flower))
    //        return false;
    //}
    //{
    //    const ChunkPos chunks[] = QUAD_CHUNK(-46, 90, 1, -1);
    //    const BlockPos2D flower = { 10, 2 };
    //    if (!testFlowerInChunkConditional(&xrand, sc, chunks, flower))
    //        return false;
    //}

    // test 3
    {
        const ChunkPos chunks[] = QUAD_CHUNK(181, -112, 1, 1);
        const BlockPos2D flower = { 13, 9 };
        if (!testFlowerInChunkConditional(&xrand, sc, chunks, flower))
            return false;
    }
    {
        const ChunkPos chunks[] = QUAD_CHUNK(180, -114, -1, -1);
        const BlockPos2D flower = { 4, 5 };
        if (!testFlowerInChunkConditional(&xrand, sc, chunks, flower))
            return false;
    }
    {
        const ChunkPos chunks[] = QUAD_CHUNK(171, -119, 1, -1);
        const BlockPos2D flower = { 11, 0 };
        if (!testFlowerInChunkConditional(&xrand, sc, chunks, flower))
            return false;
    }

	// test 4
	/*{
		const ChunkPos chunks[] = QUAD_CHUNK(-149, -160, -1, -1);
		const BlockPos2D flower = { 3, 2 };
		if (!testFlowerInChunkConditional(&xrand, sc, chunks, flower))
			return false;
	}
    {
        const ChunkPos chunks[] = QUAD_CHUNK(-149, -161, 1, 1);
        const BlockPos2D flower = { 15, 15 };
        if (!testFlowerInChunkConditional(&xrand, sc, chunks, flower))
            return false;
    }
    {
        const ChunkPos chunks[] = QUAD_CHUNK(-149, -161, 1, 1);
        const BlockPos2D flower = { 14, 12 };
        if (!testFlowerInChunkConditional(&xrand, sc, chunks, flower))
            return false;
    }*/
    
    return true;
}

__device__ static inline bool testMushroom(const SeedConstants& sc)
{
    // missing test 2 mushrooms but dont care

    // test3
    return 
        canMushroomGenerate(sc, { 2695, -1925 })
        && canMushroomGenerate(sc, { 2700, -1927 })
        && canMushroomGenerate(sc, { 2700, -1932 })
        && canMushroomGenerate(sc, { 2709, -1949 })
        && canMushroomGenerate(sc, { 2696, -1969 })
        && canMushroomGenerate(sc, { 2718, -1974 })
    ;

  //  return
		//canMushroomGenerate(sc, { -2344, -2511 })
		//&& canMushroomGenerate(sc, { -2385, -2506 })
		//&& canMushroomGenerate(sc, { -2400, -2456 })
  //  ;
}

__device__ static inline void randomBullshitFilter(const uint64_t worldseed)
{
    // filters seeds based on obstructed flowers, then mushroom, then pale oaks

    // calculate shared constants
    Xoroshiro xrand = { 0ULL, 0ULL };
    SeedConstants sc = { worldseed, 0ULL, 0ULL };
    xSetSeed(&xrand, worldseed);
    sc.A = (xNextLongJ(&xrand) | 1ULL) << 4;
    sc.B = (xNextLongJ(&xrand) | 1ULL) << 4;

    // check flower generation (decently fast filter)
    if (!testFlowers(sc))
    	return;

    // check mushroom generation (slow filter)
    if (!testMushroom(sc))
        return;

    // check tree generation (very, very slow filter)
    for (int i = 0; i < targetTreeCount; i++)
    {
        if (!canTreeGenerate(sc, targetTrees[i]))
            return;
    }


    const int i = atomicAdd(&resultID1, 1);
    if (i >= MAX_RESULTS_1)
    {
        printf("DEVICE ERROR: too many results!\n");
        return;
    }

    results1[i] = worldseed;
}

// --------------------------------------------------------------------------------------------

static int setupConstantMemory()
{
    PaleOakTree trees_H[MAX_PALE_OAKS];
    int treeCount = 0;

    FILE* fptr = fopen(TREE_FILE, "r");
    if (fptr == NULL)
        HOST_ERROR("couldn't open input file");

    int treeHeight;
    BlockPos genSource, branch;
    while (treeCount < MAX_PALE_OAKS && fscanf(fptr, "%d%d%d%d", &(genSource.x), &(genSource.y), &(genSource.z), &treeHeight) == 4)
    {
        PaleOakTree* treePtr = &(trees_H[treeCount]);
        initTreeData(treePtr, genSource, treeHeight);
        treeCount++;

        // read branch data
        int branches = 0;
        (void)fscanf(fptr, "%d", &branches);
        for (int i = 0; i < branches; i++)
        {
            (void)fscanf(fptr, "%d%d%d", &(branch.x), &(branch.y), &(branch.z));
            addBranch(treePtr, branch);
        }
    }
    fclose(fptr);

    printf("Read %d trees from file\n", treeCount);

    CHECKED_OPERATION(cudaMemcpyToSymbol(targetTrees, trees_H, sizeof(PaleOakTree) * treeCount));
    CHECKED_OPERATION(cudaMemcpyToSymbol(targetTreeCount, &treeCount, sizeof(int)));

    return 0;
}

// ----------------------------------------------------------------

constexpr uint64_t TEXT_SEEDS_TOTAL = 1ULL << 32;

__global__ static void crackTextSeedTreesTest()
{
    uint64_t tid = threadIdx.x + (uint64_t)blockDim.x * blockIdx.x;
    if (tid >= TEXT_SEEDS_TOTAL) return;

    // extend the sign bit if necessary
    uint64_t worldseed = tid;
    if ((worldseed & 0x80000000ULL) != 0ULL)
        worldseed |= 0xffffffff00000000;

    randomBullshitFilter(worldseed);
}

static int testCrackerTextSeeds()
{
    CHECKED_OPERATION(cudaSetDevice(0));

    if (setupConstantMemory() != 0)
        return 1;

    auto start = std::chrono::high_resolution_clock::now();

    resultID1 = 0;

    const int THREADS_PER_BLOCK = 256;
    const int NUM_BLOCKS_1 = (TEXT_SEEDS_TOTAL + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    crackTextSeedTreesTest << < NUM_BLOCKS_1, THREADS_PER_BLOCK >> > ();
    CHECKED_OPERATION(cudaGetLastError());
    CHECKED_OPERATION(cudaDeviceSynchronize());

    for (int i = 0; i < resultID1; i++)
    {
        printSignedSeed(results1[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    double ms = (double)elapsed.count() / 1000000.0;
    printf("\nKernel took %lf ms\n", ms);

    CHECKED_OPERATION(cudaDeviceReset());

    return 0;
}

// ------------------------------------------------------

int testTreeKernelTextSeeds()
{
    return testCrackerTextSeeds();
}