#pragma once

#include "jrand.cuh"
#include <cinttypes>

constexpr int FLOWER_PATCH_SALT = 90003;
constexpr int SINGLE_FLOWER_SALT = 90004;

typedef struct Pos2 ChunkPos;
typedef struct Pos2 BlockPos2D;
struct Pos2 {
    int x;
    int z;
};

typedef struct Pos3 BlockPos;
struct Pos3 {
    int x;
    int y;
    int z;
};



// --------------------------------------
// simple checks
// --------------------------------------

__device__ inline bool singleFlowerGenerates(Xoroshiro* xrand, const BlockPos2D& relativePos)
{
    // rarity filter (chance = 8)
    const float r = xNextFloat(xrand);
    if (!(r < 1.0F / 32.0F))
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
    if (!(r < 1.0F / 8.0F))
        return false;

    // in-square placement
    const int centerX = xNextIntJPO2(xrand, 16) + (cpos.x << 4);
    const int centerZ = xNextIntJPO2(xrand, 16) + (cpos.z << 4);

    // check whether the patch could generate a flower at (x,z)
    return centerX - 7 <= pos.x && pos.x <= centerX + 7 && centerZ - 7 <= pos.z && pos.z <= centerZ + 7;
}

__device__ inline bool conditionalCheckNearPos(Xoroshiro* xrand, uint64_t worldseed, const ChunkPos chunkCoords[], const BlockPos2D& pos)
{
    // if no flower patch in chunks (0), (1), (2) then 
    // there must be a patch in chunk (3)

#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        xSetDecoratorSeed(xrand, worldseed, chunkCoords[i].x << 4, chunkCoords[i].z << 4, FLOWER_PATCH_SALT);
        if (patchGeneratesNearPos(xrand, pos, chunkCoords[i])) return true; // false -> any <=> true
    }

    xSetDecoratorSeed(xrand, worldseed, chunkCoords[3].x << 4, chunkCoords[3].z << 4, FLOWER_PATCH_SALT);
    return patchGeneratesNearPos(xrand, pos, chunkCoords[3]);
}


// --------------------------------------
// complete filters
// --------------------------------------

__device__ inline bool testFlowerInChunkUnconditional(Xoroshiro* xrand, uint64_t worldseed, const ChunkPos& chunkPos, const BlockPos2D& flowerPosInChunk)
{
    uint64_t popseed = xGetPopulationSeed(xrand, worldseed, (chunkPos.x << 4), (chunkPos.z << 4));

    xSetSeed(xrand, popseed + FLOWER_PATCH_SALT);
    if (patchGeneratesNearPos(xrand, { (chunkPos.x << 4) + flowerPosInChunk.x, (chunkPos.z << 4) + flowerPosInChunk.z }, chunkPos))
        return true;

    xSetSeed(xrand, popseed + SINGLE_FLOWER_SALT);
    return singleFlowerGenerates(xrand, flowerPosInChunk);
}

__device__ inline bool testFlowerInChunkConditional(Xoroshiro* xrand, uint64_t worldseed, const ChunkPos chunks[], const BlockPos2D& flowerPosInChunk)
{
    if (conditionalCheckNearPos(xrand, worldseed, chunks, { (chunks[3].x << 4) + flowerPosInChunk.x, (chunks[3].z << 4) + flowerPosInChunk.z }))
        return true;

    // try single flower (pretty hopeless but whatever)
    xSetDecoratorSeed(xrand, worldseed, chunks[3].x << 4, chunks[3].z << 4, SINGLE_FLOWER_SALT);
    return singleFlowerGenerates(xrand, flowerPosInChunk);
}
