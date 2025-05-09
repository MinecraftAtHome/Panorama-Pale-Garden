#include "treegen.h"
#include "flowergen.h"


#define __STDC_FORMAT_MACROS 1

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <inttypes.h>

#include <chrono>
typedef std::chrono::steady_clock::time_point time_point;

#ifdef BOINC
constexpr int RUNS_PER_CHECKPOINT = 16;
#include "boinc/boinc_api.h"
#if defined _WIN32 || defined _WIN64
#include "boinc/boinc_win.h"
#endif
#endif

#define GPU_ASSERT(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "[ERROR] GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
        exit(code);
    }
}

#define HOST_ERROR(msg) hostError((msg), __FILE__, __LINE__)
inline void hostError(const char* msg, const char* file, int line) {
	fprintf(stderr, "[ERROR] %s, line %d: %s \n", file, line, msg);
	exit(1);
}

#define HOST_LOG(...) \
{ \
	fprintf(stderr, "[LOG] %s, line %d: ", __FILE__, __LINE__); \
	fprintf(stderr, __VA_ARGS__); \
	fprintf(stderr, "\n"); \
}


// crucial data
#define TREE_FILE "pale_oak_trees.txt"
#define FLOWER_POS { 5, 4 }
#define FLOWER_CHUNKS QUAD_CHUNK(191, 19, -1, -1)
#define MUSHROOM_POS { 3049, 382 }

// ---------------------------------------------------------------------------------------------

//#define STATS

// filter data
constexpr int MAX_PALE_OAKS = 32;
__constant__ PaleOakTree targetTrees[MAX_PALE_OAKS];
__constant__ int targetTreeCount;
__constant__ PaleOakTree targetOredTrees[MAX_PALE_OAKS];
__constant__ int targetOredTreeCount;

// for initial filter
constexpr int MAX_RESULTS_1 = 1024;
__managed__ uint64_t results1[MAX_RESULTS_1];
__managed__ int resultID1;

// flower cluster positions, hardcoded because i'm lazy
constexpr int CLUSTER_THRESHOLD = 2;
constexpr int FLOWER_CLUSTER_COUNT = 5;
__constant__ BlockPos2D flowerClusters[FLOWER_CLUSTER_COUNT] = {
    { 3051, 331 },
    { 3062, 328 },
    { 3067, 341 },
    { 3050, 355 },
    { 3037, 330 }
};

// for boinc quorum validation
__managed__ uint32_t checksum = 0;

// --------------------------------------------------------------------------------------------

// 3s per 2^32
//__device__ inline bool testFlowers(const SeedConstants& sc)
//{
//    Xoroshiro xrand = { 0ULL, 0ULL };
//
//    // the most likely naturally generated flower
//    const ChunkPos chunks1[] = QUAD_CHUNK(191, 19, -1, -1);
//    const BlockPos2D flower1 = { 5, 4 };
//    if (!testFlowerInChunkConditional(&xrand, sc, chunks1, flower1))
//        return false;
//
//    // i feel like this has to be natural too, just a hunch though
//    const ChunkPos chunks2[] = QUAD_CHUNK(190, 22, 1, -1);
//    const BlockPos2D flower2 = { 10, 2 };
//    if (!testFlowerInChunkConditional(&xrand, sc, chunks2, flower2))
//        return false;
//
//    // for these two we'll just say that it's likely that at least one is correct
//    bool anyGood = false;
//
//    const ChunkPos chunks3[] = QUAD_CHUNK(191, 21, 1, -1);
//    const BlockPos2D flower3 = { 11, 5 };
//    anyGood = testFlowerInChunkConditional(&xrand, sc, chunks3, flower3);
//
//    const ChunkPos chunks4[] = QUAD_CHUNK(189, 20, -1, 1);
//    const BlockPos2D flower4 = { 0, 11 };
//    if (!anyGood)
//        anyGood = testFlowerInChunkConditional(&xrand, sc, chunks4, flower4);
//
//    return anyGood;
//}

// 13s per 2^32
//__device__ inline bool testFlowers2 /*ElectricBoogaloo*/(const SeedConstants& sc)
//{
//    Xoroshiro xrand = { 0ULL, 0ULL };
//
//    // the most likely naturally generated flower
//    const ChunkPos chunks1[] = QUAD_CHUNK(191, 19, -1, -1);
//    const BlockPos2D flower1 = { 5, 4 };
//    if (!testFlowerInChunkConditional(&xrand, sc, chunks1, flower1))
//        return false;
//
//    int matchedClusters = 0;
//#pragma unroll
//    for (int i = 0; i < FLOWER_CLUSTER_COUNT; i++)
//    {
//        const BlockPos2D clusterPos = flowerClusters[i];
//        const BlockPos2D flower = { clusterPos.x & 15, clusterPos.z & 15 };
//        const BlockPos2D dir = { (flower.x < 8 ? -1 : 1), (flower.z < 8 ? -1 : 1) };
//        const ChunkPos clusterChunk = { clusterPos.x >> 4, clusterPos.z >> 4 };
//        const ChunkPos chunks[] = QUAD_CHUNK(clusterChunk.x, clusterChunk.z, dir.x, dir.z);
//
//        if (testFlowerInChunkConditional(&xrand, sc, chunks, flower))
//            matchedClusters++;
//    }
//
//    return matchedClusters >= CLUSTER_THRESHOLD;
//}

// 51s per 2^32
__device__ inline bool testFlowers3(const SeedConstants& sc)
{
    Xoroshiro xrand = { 0ULL, 0ULL };

    // the most likely naturally generated flower
    const ChunkPos chunks1[] = FLOWER_CHUNKS;
    const BlockPos2D flower1 = FLOWER_POS;
    return testFlowerInChunkConditional(&xrand, sc, chunks1, flower1);
}

__device__ inline bool testMushroom(const SeedConstants& sc)
{
    const BlockPos2D mushroomStem = MUSHROOM_POS;
    return canMushroomGenerate(sc, mushroomStem);
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
    //if (!testFlowers3(sc))
    //    return;

    // check mushroom generation (slow filter)
    //if (!testMushroom(sc))
    //    return;

    // check tree generation (very, very slow filter)
    for (int i = 0; i < targetTreeCount; i++)
    {
        if (!canTreeGenerate(sc, targetTrees[i]))
            return;
        if (i >= 1) // introduced this to limit the number of managed atomic ops performed
            atomicAdd(reinterpret_cast<uint32_t*>(&checksum), static_cast<uint32_t>(sc.B)); 
    }

    // check or-ed tree generation (very, very slow filter)
    // for now we're hardcoding groups of 2
    for (int i = 0; i < targetOredTreeCount; i += 2)
    {
        const bool canAnyGen = canTreeGenerate(sc, targetOredTrees[i]) || canTreeGenerate(sc, targetOredTrees[i + 1]);
        if (!canAnyGen)
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
    PaleOakTree oredTrees_H[MAX_PALE_OAKS];
    int treeCount = 0, oredTreeCount = 0;

    FILE* fptr = fopen(TREE_FILE, "r");
    if (fptr == NULL)
        HOST_ERROR("couldn't open input file");

    int treeHeight;
    BlockPos genSource, branch;
    int treesInGroup = 1;

    while (treeCount < MAX_PALE_OAKS && fscanf(fptr, "%d", &treesInGroup) > 0)
    {
        //printf("trees in group: %d\n", treesInGroup);
        int* countptr = treesInGroup > 1 ? &oredTreeCount : &treeCount;

        for (int i = 0; i < treesInGroup; i++)
        {
            //printf("reading tree\n");
            PaleOakTree* treePtr = treesInGroup > 1 ? &(oredTrees_H[*countptr]) : &(trees_H[*countptr]);
            (*countptr)++;

            int branches = 0;
            if (fscanf(fptr, "%d%d%d%d%d", &(genSource.x), &(genSource.y), &(genSource.z), &treeHeight, &branches) != 5)
                HOST_ERROR("incorrect input data format");

            //printf("data: %d %d %d %d\n", genSource.x, genSource.y, genSource.z, treeHeight);
            initTreeData(treePtr, genSource, treeHeight);

            // read branch data
            for (int i = 0; i < branches; i++)
            {
                if (fscanf(fptr, "%d%d%d", &(branch.x), &(branch.y), &(branch.z)) != 3)
					HOST_ERROR("incorrect input data format");
                addBranch(treePtr, branch);
            }
        }

    }
    fclose(fptr);

    //printf("Read %d trees, %d or-ed trees from file\n", treeCount, oredTreeCount);

    GPU_ASSERT(cudaMemcpyToSymbol(targetTrees, trees_H, sizeof(PaleOakTree) * treeCount));
    GPU_ASSERT(cudaMemcpyToSymbol(targetTreeCount, &treeCount, sizeof(int)));
    GPU_ASSERT(cudaMemcpyToSymbol(targetOredTrees, oredTrees_H, sizeof(PaleOakTree) * oredTreeCount));
    GPU_ASSERT(cudaMemcpyToSymbol(targetOredTreeCount, &oredTreeCount, sizeof(int)));

    return 0;
}

// ----------------------------------------------------------------

//constexpr uint64_t TEXT_SEEDS_TOTAL = 1ULL << 32;
//
//__global__ void crackTextSeedTrees()
//{
//    uint64_t tid = threadIdx.x + (uint64_t)blockDim.x * blockIdx.x;
//    if (tid >= TEXT_SEEDS_TOTAL) return;
//
//    // extend the sign bit if necessary
//    uint64_t worldseed = tid;
//    if ((worldseed & 0x80000000ULL) != 0ULL)
//        worldseed |= 0xffffffff00000000;
//
//    randomBullshitFilter(worldseed);
//}
//
//static int runCrackerTextSeeds()
//{
//    GPU_ASSERT(cudaSetDevice(0));
//    if (setupConstantMemory() != 0)
//        return 1;
//
//    auto start = std::chrono::steady_clock::now();
//
//    resultID1 = 0;
//    const int THREADS_PER_BLOCK = 256;
//    const int NUM_BLOCKS_1 = (TEXT_SEEDS_TOTAL + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//    crackTextSeedTrees <<< NUM_BLOCKS_1, THREADS_PER_BLOCK >>> ();
//    GPU_ASSERT(cudaGetLastError());
//    GPU_ASSERT(cudaDeviceSynchronize());
//
//    for (int i = 0; i < resultID1; i++)
//    {
//        printSignedSeed(results1[i]);
//    }
//
//    auto end = std::chrono::high_resolution_clock::now();
//    auto elapsed = end - start;
//    double ms = (double)elapsed.count() / 1000000.0;
//    printf("Kernel took %f ms\n");
//
//    GPU_ASSERT(cudaDeviceReset());
//    return 0;
//}

// ----------------------------------------------------------------

constexpr uint64_t RANDOM_SEEDS_TOTAL = 1ULL << 48;
constexpr uint64_t THREADS_LAUNCHED_PER_RUN = 1ULL << 28; // each run should now take 1-5 seconds
constexpr uint64_t RANDOM_SEEDS_PER_RUN = THREADS_LAUNCHED_PER_RUN;
constexpr int NUM_RUNS_RANDOM_SEEDS = (RANDOM_SEEDS_TOTAL + RANDOM_SEEDS_PER_RUN - 1) / RANDOM_SEEDS_PER_RUN;

// for boinc checkpointing
struct checkpoint_vars {
    int32_t range_min;
    int32_t range_max;
    uint32_t stored_checksum;
    uint64_t elapsed_chkpoint;
};
int32_t global_range_min = 0;
int32_t global_range_max = 0;


__global__ void crackRandomSeedTrees(const uint64_t offset)
{
    // tid is the first state of java random used in the nextLong()
    const uint64_t tid = threadIdx.x + (uint64_t)blockDim.x * blockIdx.x + offset;
    if (tid >= RANDOM_SEEDS_TOTAL) return;

    const uint64_t secondState = (tid * JRAND_MULTIPLIER + JRAND_ADDEND) & MASK48;
    const int toAdd = (int)(secondState >> 16);
    const uint64_t worldseed = ((tid >> 16) << 32) + toAdd;
    randomBullshitFilter(worldseed);
}

static int runCrackerRandomSeeds(int32_t runStart, int32_t runEnd, uint64_t time_elapsed, int32_t devID)
{
    if (runStart < 0)
    {
        runStart = 0;
        HOST_LOG("runStart (%d) was negative, set to 0", runStart);
    }
    if (runEnd > NUM_RUNS_RANDOM_SEEDS)
    {
        runEnd = NUM_RUNS_RANDOM_SEEDS;
        HOST_LOG("runEnd (%d) was above max value (%d), set to %d", runEnd, NUM_RUNS_RANDOM_SEEDS, NUM_RUNS_RANDOM_SEEDS);
    }

    GPU_ASSERT(cudaSetDevice(devID));
    if (setupConstantMemory() != 0)
		HOST_ERROR("CRITICAL: setupConstantMemory failed\n");

    uint64_t checkpointTemp = 0;
    FILE* seedsout = fopen("seeds.txt", "w+");

    for (int32_t run = runStart; run < runEnd; run++)
    {
        resultID1 = 0;

        const int THREADS_PER_BLOCK = 256;
        const int NUM_BLOCKS_1 = (THREADS_LAUNCHED_PER_RUN + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        time_point t0 = std::chrono::steady_clock::now();
        crackRandomSeedTrees <<< NUM_BLOCKS_1, THREADS_PER_BLOCK >>> (run * THREADS_LAUNCHED_PER_RUN);
        GPU_ASSERT(cudaGetLastError());
        GPU_ASSERT(cudaDeviceSynchronize());
		time_point t1 = std::chrono::steady_clock::now();
		uint64_t run_duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
		time_elapsed += run_duration;
        checkpointTemp++;

		// write the results before doing checkpoint
		for (int i = 0; i < resultID1; i++)
		{
			const int64_t worldseed = (int64_t)(results1[i]);
			fprintf(seedsout, "%" PRId64 "\n", worldseed);
		}
        fflush(seedsout);

#ifdef BOINC
        if (checkpointTemp >= RUNS_PER_CHECKPOINT-1 || boinc_time_to_checkpoint()) {
            //Checkpointing for BOINC
            boinc_begin_critical_section(); // Boinc should not interrupt this

            checkpointTemp = 0;
            boinc_delete_file("checkpoint.txt"); // Don't touch, same func as normal fdel
            FILE* checkpoint_data = boinc_fopen("checkpoint.txt", "wb");

            struct checkpoint_vars data_store;
			data_store.range_min = run + 1; // this run was already completed, processing can resume from next run
			data_store.range_max = runEnd;
            data_store.elapsed_chkpoint = time_elapsed;
			data_store.stored_checksum = checksum;
            fwrite(&data_store, sizeof(data_store), 1, checkpoint_data);
            fclose(checkpoint_data);

            boinc_end_critical_section();
            boinc_checkpoint_completed(); // Checkpointing completed
        }
        // Update boinc client with percentage
        double frac = (double)(run - global_range_min + 1) / (double)(global_range_max - global_range_min);
        boinc_fraction_done(frac);
#endif // BOINC
    }
    GPU_ASSERT(cudaDeviceReset());

    // append checksum to result file
	fprintf(seedsout, "##%" PRIu32 "\n", checksum);
	fclose(seedsout);

    // write performance stats to stderr
    uint64_t seeds_checked = (global_range_max - global_range_min) * RANDOM_SEEDS_PER_RUN;
	double elapsed_s = (double)time_elapsed * 1e-6;
	double sps = (double)seeds_checked / elapsed_s;
    fprintf(stderr, "[stats] completed tasks = [%d, %d)\n", global_range_min, global_range_max);
    fprintf(stderr, "[stats] seeds checked = %llu\n", seeds_checked);
    fprintf(stderr, "[stats] time taken = %f (s)\n", elapsed_s);
	fprintf(stderr, "[stats] speed = %f (seeds/s)\n", sps);
	fprintf(stderr, "[checksum] %u\n", checksum);

#ifdef BOINC
    boinc_finish(0);
#endif
    return 0;
}

// ------------------------------------------------------

int runTreeKernel(int argc, char** argv)
{
    int32_t range_min = 0;
    int32_t range_max = 0;
    int32_t device = 0;

    for (int i = 0; i < argc; i++)
    {
        const char* param = argv[i];
        if (strcmp(param, "-d") == 0 || strcmp(param, "--device") == 0) {
            sscanf(argv[i + 1], "%d", &device);
            i++;
        }
        else if (strcmp(param, "-s") == 0 || strcmp(param, "--start") == 0) {
            sscanf(argv[i + 1], "%d", &range_min);
            i++;
        }
        else if (strcmp(param, "-e") == 0 || strcmp(param, "--end") == 0) {
            sscanf(argv[i + 1], "%d", &range_max);
            i++;
        }
        else {
			HOST_LOG("Unknown parameter: %s", param);
        }
    }
    if (range_min == 0 && range_max == 0)
    {
		HOST_LOG("range might not have been specified (was 0:0).");
    }

    // range_min, range_max will get updated by the checkpoint, their
	// initial values need to be stored for performance measurement
    global_range_min = range_min;
	global_range_max = range_max;
    checksum = 0;
    uint64_t time_elapsed = 0;

#ifdef BOINC
    BOINC_OPTIONS options;
    boinc_options_defaults(options);
    options.normal_thread_priority = true;
    boinc_init_options(&options);
    APP_INIT_DATA aid;
    boinc_get_init_data(aid);
    if (aid.gpu_device_num >= 0) {
        //If BOINC client provided us a device ID
        device = aid.gpu_device_num;
        fprintf(stderr, "boinc gpu %i gpuindex: %i \n", aid.gpu_device_num, device);
    }
    else {
        //If BOINC client did not provide us a device ID
        device = -5;
        for (int i = 1; i < argc; i += 2) {
            //Check for a --device flag, just in case we missed it earlier, use it if it's available. For older clients primarily.
            if (strcmp(argv[i], "--device") == 0) {
                sscanf(argv[i + 1], "%i", &device);
            }

        }
        if (device == -5) {
            //Something has gone wrong. It pulled from BOINC, got -1. No --device parameter present.
            fprintf(stderr, "Error: No --device parameter provided! Defaulting to device 0...\n");
            device = 0;
        }
        fprintf(stderr, "stndalone gpuindex %i (aid value: %i)\n", device, aid.gpu_device_num);
    }

    FILE* checkpoint_data = boinc_fopen("checkpoint.txt", "rb");
    if (!checkpoint_data) {
        //No checkpoint file was found. Proceed from the beginning.
        fprintf(stderr, "No checkpoint to load\n");
    }
    else {
        //Load from checkpoint. You can put any data in data_store that you need to keep between runs of this program.
        boinc_begin_critical_section();
        struct checkpoint_vars data_store;
        fread(&data_store, sizeof(data_store), 1, checkpoint_data);
		range_min = data_store.range_min;
		range_max = data_store.range_max;
        time_elapsed = data_store.elapsed_chkpoint;
		checksum = data_store.stored_checksum;
        fprintf(stderr, "Checkpoint loaded, task time %llu us, seed pos: %llu\n", time_elapsed, range_min);
        fclose(checkpoint_data);
        boinc_end_critical_section();
    }
#endif // BOINC

    return runCrackerRandomSeeds(range_min, range_max, time_elapsed, device);
}

int runTreeKernelTextSeeds()
{
    //return runCrackerTextSeeds();
    return 0;
}