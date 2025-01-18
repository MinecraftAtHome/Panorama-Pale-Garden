#ifndef TREEGEN_H_
#define TREEGEN_H_

#include "mccore.h"


typedef struct PaleOakTrunk PaleOakTrunk;
struct PaleOakTrunk {
	BlockPos generationSource;
	int branches[4][4];		// contains branch y-positions relative to the tree's generation source
	int branchCount;	// how many extra branches are visible in the panorama
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

__device__ inline bool testTreePos(Xoroshiro* xrand, const PaleOakTrunk& target)
{
	// assume that the generator is seeded to generate the next 
	// tree-like feature in some chunk.

	const int treeX = xNextIntJPO2(xrand, 16);
	if (target.generationSource.x != treeX) 
		return false;

	const int treeZ = xNextIntJPO2(xrand, 16);
	return target.generationSource.z == treeZ;
}


__device__ inline bool testPaleOakTree(Xoroshiro* xrand, const PaleOakTrunk& target)
{
	// we already got a good pale oak tree position here, 
	// test if the extra branches are all a match

	if (xNextFloat(xrand) >= 0.1F)
		xNextLong(xrand); // burn one xNextFloat() call if creaking heart

    int treeHeight = 6 + xNextIntJ(xrand, 3);   // baseHeight + heightRandA
	treeHeight += xNextIntJ(xrand, 2);          // + heightRandB

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

				const int lowestExtraBranchBlock = maxY - extraBranchSize - 2;
				if (lowestExtraBranchBlock != branchRequirement)
					return false; // had branch data and it didn't match, fail

				matchedBranches++;

                //for (int deltaY = 0; deltaY < extraBranchSize; ++deltaY) {
                //    this.placeLog(new BPos(startPosX + deltaX, maxY - deltaY - 1, startPosZ + deltaZ));
                //}
            }
        }
    }

	// if all branches matched, we're good
	return matchedBranches == target.branchCount;
}


__host__ inline void initTreeData(PaleOakTrunk* treeData, const BlockPos& generationSource)
{
	treeData->branchCount = 0;
	treeData->generationSource = generationSource;

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			treeData->branches[i][j] = -1;
}

__host__ inline void addBranch(PaleOakTrunk* treeData, int x, int y, int z)
{
	// branch position relative to the tree's generation source
	x -= treeData->generationSource.x;
	y -= treeData->generationSource.y;
	z -= treeData->generationSource.z;

	// offset the (x,z) by (1,1) to fit the array indices
	treeData->branches[x + 1][z + 1] = y;
	treeData->branchCount++;
}

#endif // TREEGEN_H_