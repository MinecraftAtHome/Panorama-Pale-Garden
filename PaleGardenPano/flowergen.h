#pragma once

#include "mccore.h"
#include <cinttypes>

constexpr int FLOWER_PATCH_SALT = 90003;
constexpr int SINGLE_FLOWER_SALT = 90004;

// --------------------------------------
// helper functions
// --------------------------------------

__device__ inline bool getFlowerGenOrigin(uint64_t populationSeed, const ChunkPos& chunkPos, BlockPos2D *resultPtr)
{
	Xoroshiro xrand = { 0ULL, 0ULL };
	xSetSeed(&xrand, populationSeed + FLOWER_PATCH_SALT);

	// rarity filter (chance = 8)
	if (xNextFloat(&xrand) >= 1.0F / 8.0F)
		return false;

	resultPtr->x = xNextIntJPO2(&xrand, 16) + (chunkPos.x << 4);
	resultPtr->z = xNextIntJPO2(&xrand, 16) + (chunkPos.z << 4);
	return true;
}

__device__ inline bool getSingleFlower(uint64_t populationSeed, const ChunkPos& chunkPos, BlockPos2D* resultPtr)
{
    Xoroshiro xrand = { 0ULL, 0ULL };
	xSetSeed(&xrand, populationSeed + SINGLE_FLOWER_SALT);

    // rarity filter (chance = 32)
    if (xNextFloat(&xrand) >= 1.0F / 32.0F)
        return false;

    resultPtr->x = xNextIntJPO2(&xrand, 16) + (chunkPos.x << 4);
    resultPtr->z = xNextIntJPO2(&xrand, 16) + (chunkPos.z << 4);
    return true;
}

// --------------------------------------
// simple checks
// --------------------------------------

__device__ inline bool singleFlowerGenerates(Xoroshiro* xrand, const BlockPos2D& relativePos)
{
    // rarity filter (chance = 32)
    const float r = xNextFloat(xrand);
    if (r >= 1.0F / 32.0F)
        return false;

    // inSquarePlacement -- test coords
    if (xNextIntJPO2(xrand, 16) != relativePos.x)
        return false;

    return xNextIntJPO2(xrand, 16) == relativePos.z;
}

__device__ inline bool patchGenerates(Xoroshiro* xrand)
{
    // rarity filter (chance = 8)
    const float r = xNextFloat(xrand);
    return r < 1.0F / 8.0F;

    // don't care about anything else here
}

__device__ inline bool conditionalCheck(Xoroshiro* xrand, uint64_t worldseed, const ChunkPos chunkCoords[])
{
    // if no flower patch in chunks (0), (1), (2) then 
    // there must be a patch in chunk (3)

#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        xSetDecoratorSeed(xrand, worldseed, chunkCoords[i].x << 4, chunkCoords[i].z << 4, FLOWER_PATCH_SALT);
        if (patchGenerates(xrand)) return true; // false -> any <=> true
    }

    xSetDecoratorSeed(xrand, worldseed, chunkCoords[3].x << 4, chunkCoords[3].z << 4, FLOWER_PATCH_SALT);
    return patchGenerates(xrand);
}


// --------------------------------------
// position-bounded checks
// --------------------------------------

__device__ inline bool patchGeneratesNearPos(Xoroshiro* xrand, const BlockPos2D& pos, const ChunkPos& cpos)
{
    // rarity filter (chance = 8)
    const float r = xNextFloat(xrand);
    if (r >= 1.0F / 8.0F)
        return false;

    // in-square placement
    const int centerX = xNextIntJPO2(xrand, 16) + (cpos.x << 4);
    const int centerZ = xNextIntJPO2(xrand, 16) + (cpos.z << 4);

    // check whether the patch could generate a flower at (x,z)
    return centerX - 7 <= pos.x && pos.x <= centerX + 7 && centerZ - 7 <= pos.z && pos.z <= centerZ + 7;
}

//__device__ inline bool conditionalCheckNearPos(Xoroshiro* xrand, uint64_t worldseed, const ChunkPos chunkCoords[], const BlockPos2D& pos)
//{
//    // if no flower patch in chunks (0), (1), (2) then 
//    // there must be a patch in chunk (3)
//
//#pragma unroll
//    for (int i = 0; i < 3; i++)
//    {
//        xSetDecoratorSeed(xrand, worldseed, chunkCoords[i].x << 4, chunkCoords[i].z << 4, FLOWER_PATCH_SALT);
//        if (patchGeneratesNearPos(xrand, pos, chunkCoords[i])) return true; // false -> any <=> true
//    }
//
//    xSetDecoratorSeed(xrand, worldseed, chunkCoords[3].x << 4, chunkCoords[3].z << 4, FLOWER_PATCH_SALT);
//    return patchGeneratesNearPos(xrand, pos, chunkCoords[3]);
//}


// --------------------------------------
// complete filters
// --------------------------------------

__device__ inline bool testFlowerInChunkUnconditional(Xoroshiro* xrand, const SeedConstants& sc, const ChunkPos& chunkPos, const BlockPos2D& flowerPosInChunk)
{
    const uint64_t popseed = getPopulationSeedFast(sc, chunkPos);

    xSetSeed(xrand, popseed + FLOWER_PATCH_SALT);
    if (patchGeneratesNearPos(xrand, { (chunkPos.x << 4) + flowerPosInChunk.x, (chunkPos.z << 4) + flowerPosInChunk.z }, chunkPos))
        return true;

    xSetSeed(xrand, popseed + SINGLE_FLOWER_SALT);
    return singleFlowerGenerates(xrand, flowerPosInChunk);
}


// helper macro
#define QUAD_CHUNK(x, z, dirX, dirZ) { { x + dirX, z + dirZ }, { x + dirX, z }, { x, z + dirZ }, { x, z } }

__device__ inline bool testFlowerInChunkConditional(Xoroshiro* xrand, const SeedConstants& sc, const ChunkPos chunks[], const BlockPos2D& flowerPosInChunk)
{
    const BlockPos2D flowerPos = { (chunks[3].x << 4) + flowerPosInChunk.x, (chunks[3].z << 4) + flowerPosInChunk.z };

    // if no flower patch in chunks (0), (1), (2) then 
    // there must be a patch in chunk (3)

    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        xSetSeed(xrand, getPopulationSeedFast(sc, chunks[i]) + FLOWER_PATCH_SALT);
        if (patchGeneratesNearPos(xrand, flowerPos, chunks[i])) return true; // false -> any <=> true
    }

    const uint64_t popseedTargetChunk = getPopulationSeedFast(sc, chunks[3]);
    
    xSetSeed(xrand, popseedTargetChunk + FLOWER_PATCH_SALT);
    if (patchGeneratesNearPos(xrand, flowerPos, chunks[3]))
        return true;

    // try single flower (pretty hopeless but whatever)
    xSetSeed(xrand, popseedTargetChunk + SINGLE_FLOWER_SALT);
    return singleFlowerGenerates(xrand, flowerPosInChunk);
}


// ---------------------------------------------------------
// the full flower simulation method
// ---------------------------------------------------------

__device__ inline int addFlowersToChunk(Xoroshiro* xrand, int flowerChunk[7][16], const BlockPos2D& patchCenter, const int dcx, const int dcz)
{
    // following mc source code here
    int counter = 0;

    for (int attempt = 0; attempt < 96 /*yes let's do 576 random calls for some fucking flowers*/; attempt++)
    {
		// xNextIntJPO2() - xNextIntJPO2() would be undefined behavior!
		int x = (dcx << 4) + patchCenter.x + xNextIntJPO2(xrand, 8); x -= xNextIntJPO2(xrand, 8);
		int y = 3 + xNextIntJPO2(xrand, 4); y -= xNextIntJPO2(xrand, 4);
		int z = (dcz << 4) + patchCenter.z + xNextIntJPO2(xrand, 8); z -= xNextIntJPO2(xrand, 8);

		if (x < 0 || x >= 16 || z < 0 || z >= 16)
			continue;

		flowerChunk[y][z] |= 1 << x; // it's a matter of choice how to arrange the bits, this should be easier
        counter++;
    }

    return counter;
}
