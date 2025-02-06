#include "flowergen.h"
#include "treegen.h"
#include "cudawrapper.h"
#include <cstdio>
#include <chrono>
#include <cmath>

#define STATS
#define TREE_FILE "data/tests-trees/t1.txt"

// filter data
constexpr int MAX_PALE_OAKS = 32;
__constant__ PaleOakTree targetTrees[MAX_PALE_OAKS];
__constant__ int targetTreeCount;

// for initial filter
constexpr int MAX_RESULTS_1 = 1024 * 1024;
__managed__ uint64_t results1[MAX_RESULTS_1];
__managed__ int resultID1;

// --------------------------------------------------------------------------------------------

__device__ inline bool testFlowers(const SeedConstants& sc)
{
    // TODO add all the flower stuff here
    return true;
}

__device__ inline bool testMushroom(const SeedConstants& sc)
{
    // TODO add the mushroom filter
    return true;
}

__device__ inline void randomBullshitFilter(const uint64_t worldseed)
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
        // DEBUG-ONLY
        if (sc.worldseed == 44441ULL)
        {
            printf("Checking %d ...\n", i);
        }

        if (!canTreeGenerate(sc, targetTrees[i]))
            return;

        // DEBUG-ONLY
		if (sc.worldseed == 44441ULL)
		{
			printf("Tree %d can generate!\n\n", i);
		}
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

    BlockPos genSource, branch;
    while (treeCount < MAX_PALE_OAKS && fscanf(fptr, "%d%d%d", &(genSource.x), &(genSource.y), &(genSource.z)) == 3)
    {
        PaleOakTree* treePtr = &(trees_H[treeCount]);
		initTreeData(treePtr, genSource);
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

    CHECKED_OPERATION(cudaMemcpyToSymbol(targetTrees, trees_H, sizeof(PaleOakTree) * treeCount));
    CHECKED_OPERATION(cudaMemcpyToSymbol(targetTreeCount, &treeCount, sizeof(int)));

    return 0;
}

// ----------------------------------------------------------------

constexpr uint64_t TEXT_SEEDS_TOTAL = 1ULL << 16; // TODO REPLACE WITH 1<<32

__global__ void crackTextSeedTrees()
{
    uint64_t tid = threadIdx.x + (uint64_t)blockDim.x * blockIdx.x;
    if (tid >= TEXT_SEEDS_TOTAL) return;

    // extend the sign bit if necessary
    uint64_t worldseed = tid;
    if ((worldseed & 0x80000000ULL) != 0ULL)
        worldseed |= 0xffffffff00000000;

    randomBullshitFilter(worldseed);
}

static int runCrackerTextSeeds()
{
    CHECKED_OPERATION(cudaSetDevice(0));

    if (setupConstantMemory() != 0)
        return 1;

    auto start = std::chrono::high_resolution_clock::now();

    resultID1 = 0;

    const int THREADS_PER_BLOCK = 512;
    const int NUM_BLOCKS_1 = (TEXT_SEEDS_TOTAL + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    crackTextSeedTrees <<< NUM_BLOCKS_1, THREADS_PER_BLOCK >>> ();
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

// ----------------------------------------------------------------

constexpr uint64_t RANDOM_SEEDS_TOTAL = 1ULL << 48;
constexpr uint64_t THREADS_LAUNCHED_PER_RUN = 1ULL << 31;
//constexpr uint8_t BITS_PER_THREAD = 0;
//constexpr int RANDOM_SEEDS_PER_THREAD = 1 << BITS_PER_THREAD;
constexpr uint64_t RANDOM_SEEDS_PER_RUN = THREADS_LAUNCHED_PER_RUN;
constexpr int NUM_RUNS_RANDOM_SEEDS = (RANDOM_SEEDS_TOTAL + RANDOM_SEEDS_PER_RUN - 1) / RANDOM_SEEDS_PER_RUN;
constexpr int RUNS_PER_PRINT = 100;

__global__ void crackRandomSeedTrees(const uint64_t offset)
{
    // tid is the first state of java random used in the nextLong()
    const uint64_t tid = threadIdx.x + (uint64_t)blockDim.x * blockIdx.x + offset;
    if (tid >= RANDOM_SEEDS_TOTAL) return;

    //#pragma unroll
    //for (uint32_t low = 0; low < RANDOM_SEEDS_PER_THREAD; low++)
    //{
        //const uint64_t firstState = tid | low;
    const uint64_t secondState = (tid * JRAND_MULTIPLIER + JRAND_ADDEND) & MASK48;
    const int toAdd = (int)(secondState >> 16);
    const uint64_t worldseed = ((tid >> 16) << 32) + toAdd;

    randomBullshitFilter(worldseed);
    //}
}

static int runCrackerRandomSeeds(int runStart, int runEnd, int devID)
{
    if (runStart < 0) runStart = 0;
    if (runEnd > NUM_RUNS_RANDOM_SEEDS) runEnd = NUM_RUNS_RANDOM_SEEDS;

    CHECKED_OPERATION(cudaSetDevice(devID));

    if (setupConstantMemory() != 0)
        return 1;

#ifdef STATS
    auto startGlobal = std::chrono::steady_clock::now();
    auto start = std::chrono::steady_clock::now();
    //double ms1 = 0.0, ms2 = 0.0, ms3 = 0.0;
#endif

    for (int run = runStart; run < runEnd; run++)
    {
#ifdef STATS
        if (run % RUNS_PER_PRINT == 0)
        {
            start = std::chrono::steady_clock::now();
            printf(" --- Run %d / %d\n", run + 1, NUM_RUNS_RANDOM_SEEDS);
        }
#endif

        resultID1 = 0;

        const int THREADS_PER_BLOCK = 512;
        const int NUM_BLOCKS_1 = (THREADS_LAUNCHED_PER_RUN + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        //auto s1 = std::chrono::steady_clock::now();
        crackRandomSeedTrees <<< NUM_BLOCKS_1, THREADS_PER_BLOCK >>> (run * THREADS_LAUNCHED_PER_RUN);
        CHECKED_OPERATION(cudaGetLastError());
        CHECKED_OPERATION(cudaDeviceSynchronize());
        //auto e1 = std::chrono::steady_clock::now();
        //ms1 += (e1 - s1).count() / 1000000.0;

        for (int i = 0; i < resultID1; i++)
        {
            printSignedSeed(results1[i]);
        }

#ifdef STATS
        if (run % RUNS_PER_PRINT == RUNS_PER_PRINT - 1)
        {
            auto end = std::chrono::steady_clock::now();
            auto elapsed = end - start;
            double ms = (double)elapsed.count() / 1000000.0 / (double)RUNS_PER_PRINT;

            // calc eta based on this run
            double eta_s = ms * (runEnd - run) / 1000.0;
            int sec = (int)floor(eta_s) % 60;
            double eta_min = eta_s / 60.0;
            int min = (int)floor(eta_min) % 60;
            double eta_h = eta_min / 60.0;
            int hrs = (int)floor(eta_h);
            fprintf(stderr, "ETA: %d HRS %d MIN %d SEC\n", hrs, min, sec);
        }
#endif
    }

    CHECKED_OPERATION(cudaDeviceReset());

#ifdef STATS
    auto endGlobal = std::chrono::steady_clock::now();
    auto elapsedGlobal = endGlobal - startGlobal;
    double seconds = (double)elapsedGlobal.count() / 1000000.0 / 1000.0;
    printf("Runs took %lf seconds in total:\n", seconds);

    //printf("Filter 1: %lf sec\n", ms1 / 1000.0);
    //printf("Filter 2: %lf sec\n", ms2 / 1000.0);
    //printf("Filter 3: %lf sec\n", ms3 / 1000.0);

    double minutesFull = seconds / 60.0 * NUM_RUNS_RANDOM_SEEDS / (runEnd - runStart);
    int min = (int)floor(minutesFull) % 60;
    int hrs = (int)floor(minutesFull / 60.0);
    printf("\nEstimated runtime for full seedspace: %d hours %d minutes\n", hrs, min);
#endif

    return 0;
}

// ------------------------------------------------------

int runTreeKernel(int argc, char** argv)
{
    if (argc <= 2)
        HOST_ERROR("usage: ./executable rangeStartInclusive rangeEndExclusive [otherArgs]");

    int rangeStart = atoi(argv[1]);
    int rangeEnd = atoi(argv[2]);

    int devID = 0;
    for (int i = 0; i < argc; i++)
    {
        if (argv[i][0] == 'd' && i != argc - 1)
            devID = atoi(argv[i + 1]);
    }

    return runCrackerRandomSeeds(rangeStart, rangeEnd, devID);
}

int runTreeKernelTextSeeds()
{
	return runCrackerTextSeeds();
}