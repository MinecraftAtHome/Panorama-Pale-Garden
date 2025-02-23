#ifndef TREEGEN_H_
#define TREEGEN_H_

#include "mccore.h"
#include "cudawrapper.h"

constexpr int PALE_OAK_SALT = 90001;
constexpr int DARK_TREES_SALT = 90001;

typedef struct PaleOakTree PaleOakTree;
struct PaleOakTree {
	BlockPos generationSource;
	int branches[4][4];		// contains branch y-positions relative to the tree's generation source
	int branchCount;	// how many extra branches are visible in the panorama
	int height;			// height of the tree trunk (will always be certain, otherwise branch data is invalid)
	/*
	[] = possible branch,
	## = standard tree trunk,
	top left square is (x=0,z=0) 
	  +---------> X
	  | [][][][]
	  | []####[]
	  | []####[]
	  V [][][][]
	  Z
	*/
};

__host__ __device__ inline void printTreeData(const PaleOakTree& treeData)
{
	DEBUG_PRINT("TreeData:\n\tgenerationSource=(%d,%d,%d)\n",
		treeData.generationSource.x, treeData.generationSource.y, treeData.generationSource.z);

	DEBUG_PRINT("\t%d branches:\n\t\t", treeData.branchCount);
	for (int z = 0; z < 4; z++)
	{
		for (int x = 0; x < 4; x++)
			DEBUG_PRINT("%d ", treeData.branches[x][z]);
		DEBUG_PRINT("\n\t\t");
	}

	DEBUG_PRINT("\n");
}


__device__ inline bool testTreePos(Xoroshiro* xrand, const PaleOakTree& target)
{
	// assume that the generator is seeded to generate the next 
	// tree-like feature in some chunk.

	const int treeX = xNextIntJPO2(xrand, 16);
	if ((target.generationSource.x & 15) != treeX) 
		return false;

	const int treeZ = xNextIntJPO2(xrand, 16);
	return (target.generationSource.z & 15) == treeZ;
}

__device__ inline bool testPaleOakTree(Xoroshiro* xrand, const PaleOakTree& target)
{
	DEBUG_PRINT("testPaleOakTree()\n");

	// we already got a good pale oak tree position here, 
	// test if the extra branches are all a match

	if (xNextFloat(xrand) >= 0.1F)
		xNextLong(xrand); // burn one xNextFloat() call if creaking heart

    int treeHeight = 6 + xNextIntJ(xrand, 3);   // baseHeight + heightRandA
	treeHeight += xNextIntJ(xrand, 2);          // + heightRandB

	if (treeHeight != target.height) // not an amazing filter but still something
		return false;

	xNextLong(xrand);		// rand.nextInt(4) for bend direction
	xNextLong(xrand);		// rand.nextInt(4) for bend starting height
	xNextIntJ(xrand, 3);	// rand.nextInt(3) for bend size

	// here the algo would place the standard trunk,
	// this doesn't affect extra branches or the prng
	// state so we're ignoring this part heres

    const int maxY = treeHeight - 1;
	int matchedBranches = 0;

    for (int deltaX = -1; deltaX <= 2; deltaX++) 
	{
        for (int deltaZ = -1; deltaZ <= 2; deltaZ++) 
		{
            if ((deltaX < 0 || deltaX > 1 || deltaZ < 0 || deltaZ > 1) && xNextIntJ(xrand, 3) == 0) 
			{
                const int extraBranchSize = xNextIntJ(xrand, 3) + 2;
				const int branchRequirement = target.branches[deltaX + 1][deltaZ + 1];
				if (branchRequirement < 0)
					continue; // don't know if a branch should be here, skip

				const int lowestExtraBranchBlock = maxY - extraBranchSize;
				if (lowestExtraBranchBlock != branchRequirement)
				{
					DEBUG_PRINT("Branch mismatch: expected %d, got %d\n", branchRequirement, lowestExtraBranchBlock);
					DEBUG_PRINT("at deltaX = %d, deltaZ = %d\n", deltaX, deltaZ);
					return false; // had branch data and it didn't match, fail
				}

				matchedBranches++;

                //for (int deltaY = 0; deltaY < extraBranchSize; ++deltaY) {
                //    this.placeLog(new BPos(startPosX + deltaX, maxY - deltaY - 1, startPosZ + deltaZ));
                //}
            }
        }
    }

	DEBUG_PRINT("Matched %d branches out of %d required\n", matchedBranches, target.branchCount);

	// if all branches matched, we're good
	return matchedBranches == target.branchCount;
}

__device__ inline bool testDarkForestMushroomTree(Xoroshiro* xrand, const BlockPos2D& treePos)
{
	// test inSquarePlacement
	const int mushX = xNextIntJPO2(xrand, 16);
	const int mushZ = xNextIntJPO2(xrand, 16);
	if (mushX != (treePos.x & 15) || mushZ != (treePos.z & 15))
		return false;

	// random selector for tree type, need specific sequence of calls
	if (!(xNextFloat(xrand) < 0.025F))
		if (!(xNextFloat(xrand) < 0.05F))
			return false; // didn't get brown mushroom or red mushroom tree

	return true;
}

// --------------------------------------
// host-side data initialization utils
// --------------------------------------

__host__ inline void initTreeData(PaleOakTree* treeData, const BlockPos& generationSource, const int height)
{
	treeData->branchCount = 0;
	treeData->generationSource = generationSource;
	treeData->height = height;

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			treeData->branches[i][j] = -1;
}

__host__ inline void addBranch(PaleOakTree* treeData, BlockPos branch)
{
	// branch position relative to the tree's generation source
	branch.x -= treeData->generationSource.x;
	branch.y -= treeData->generationSource.y;
	branch.z -= treeData->generationSource.z;

	// offset the (x,z) by (1,1) to fit the array indices
	treeData->branches[branch.x + 1][branch.z + 1] = branch.y;
	treeData->branchCount++;
}

// --------------------------------------
// complete device-side filter
// --------------------------------------

__device__ inline bool canTreeGenerate(const SeedConstants& sc, const PaleOakTree& target)
{
	//// DEBUG-ONLY
	//if (sc.worldseed != 44441ULL)
	//	return false;

	DEBUG_PRINT("canTreeGenerate() called for target\n");
	//printTreeData(target);

	Xoroshiro xrand = { 0ULL, 0ULL }, xrandStep = { 0ULL, 0ULL };
	const ChunkPos chunkPos = { target.generationSource.x >> 4, target.generationSource.z >> 4 };
	const uint64_t populationSeed = getPopulationSeedFast(sc, chunkPos);

	xSetSeed(&xrandStep, populationSeed + PALE_OAK_SALT);

	// we can expect each pale oak to make around 300-350 random calls.
	// if 8 pale oaks get generated in a chunk, that's 2400-2800 calls.
	// 3000 calls should be enough to avoid false negatives.
	for (int i = 0; i < 3000; i++)
	{
		xrand.lo = xrandStep.lo;
		xrand.hi = xrandStep.hi;
		if (testTreePos(&xrand, target))
			if (testPaleOakTree(&xrand, target))
				return true;

		xNextLong(&xrandStep);
	}
	
	return false;
}

__device__ inline bool canMushroomGenerate(const SeedConstants& sc, const BlockPos2D& mushPos)
{
	Xoroshiro xrand = { 0ULL, 0ULL }, xrandStep = { 0ULL, 0ULL };
	const ChunkPos chunkPos = { mushPos.x >> 4, mushPos.z >> 4 };
	const uint64_t populationSeed = getPopulationSeedFast(sc, chunkPos);

	xSetSeed(&xrandStep, populationSeed + DARK_TREES_SALT);

	for (int i = 0; i < 500; i++)
	{
		xrand.lo = xrandStep.lo;
		xrand.hi = xrandStep.hi;
		if (testDarkForestMushroomTree(&xrand, mushPos))
			return true;

		xNextLong(&xrandStep);
	}
}

#endif // TREEGEN_H_