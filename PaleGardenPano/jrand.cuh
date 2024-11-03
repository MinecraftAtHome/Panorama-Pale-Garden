#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>
#include <cinttypes>
#include <cstdio>

#define JRAND_MULTIPLIER (0x5deece66dULL)
#define JRAND_ADDEND (11ULL)
#define MASK48 (0xffffffffffffULL)


__host__ __device__ inline void printSeed(uint64_t seed) 
{
    printf("%" PRIu64 "\n", seed);
}

__host__ __device__ inline void printSignedSeed(uint64_t seed) 
{
    const int64_t signedSeed = (int64_t)seed;
    printf("%" PRId64 "\n", signedSeed);
}



///=============================================================================
///                               Java Random
/// 
/// basic function implementations are from Cubitect's Cubiomes library:
/// https://github.com/Cubitect/cubiomes
///=============================================================================

__host__ __device__ inline int32_t next(uint64_t* rand, int bits)
{
    *rand = (*rand * 0x5deece66du + 0xBu) & MASK48;

    return (int32_t)(*rand >> (48u - bits));
}

__host__ __device__ inline uint64_t nextLong(uint64_t* rand)
{
    return ((uint64_t)next(rand, 32) << 32) + next(rand, 32);
}

// jrand state operations

__host__ __device__ inline void advance(uint64_t* rand)
{
    *rand = (*rand * 0x5deece66du + 0xBu) & MASK48;
}

__host__ __device__ inline void advance2(uint64_t* rand)
{
    *rand = (*rand * 205749139540585ULL + 277363943098ULL) & MASK48;
}


__host__ __device__ inline void goBack(uint64_t* rand)
{
    *rand = (*rand * 246154705703781ULL + 107048004364969ULL) & MASK48;
}

__host__ __device__ inline void goBack2(uint64_t* rand)
{
    *rand = (*rand * 254681119335897ULL + 120305458776662ULL) & MASK48;
}

// seeding

__host__ __device__ inline void setSeed(uint64_t* rand, uint64_t seed) 
{
    *rand = (uint64_t)(seed ^ 0x5deece66dULL) & MASK48;
}

__host__ __device__ inline void setSeedUnscrambled(uint64_t* rand, uint64_t seed) 
{
    *rand = seed & MASK48;
}

__host__ __device__ inline void setCarverSeed(uint64_t* rand, uint64_t structseed, int cx, int cz)
{
    setSeed(rand, structseed);
    uint64_t a = nextLong(rand);
    uint64_t b = nextLong(rand);
    setSeed(rand, cx * a ^ cz * b ^ structseed);
}

__host__ __device__ inline void setDecoratorSeed(uint64_t* rand, uint64_t structseed, uint64_t cx, uint64_t cz, uint64_t salt)
{
    setSeed(rand, structseed);
    uint64_t a = nextLong(rand) | 1ULL;
    uint64_t b = nextLong(rand) | 1ULL;
    uint64_t popseed = (cx * a + cz * b) ^ structseed;
    setSeed(rand, popseed + salt);
}

// rng functions

__host__ __device__ inline int nextInt(uint64_t* rand, const int n)
{
    int bits, val;
    const int m = n - 1;

    if ((m & n) == 0) {
        uint64_t x = n * (uint64_t)next(rand, 31);
        return (int)((int64_t)x >> 31);
    }

    do {
        bits = next(rand, 31);
        val = bits % n;
    } while (bits - val + m < 0);
    return val;
}

__host__ __device__ inline int fastBoundedNextInt(uint64_t* rand, const int n)
{
    return (int)next(rand, 31) % n;
}

__host__ __device__ inline int fastIntDistribution(uint64_t* rand, const int n)
{
    return fastBoundedNextInt(rand, n) - fastBoundedNextInt(rand, n);
}

__host__ __device__ inline int fastPO2NextInt(uint64_t* rand, const int n)
{
    return (int)((uint64_t)next(rand, 31) >> 31-n);
}

__host__ __device__ inline bool nextBoolean(uint64_t* rand) {
    return next(rand, 1) != 0;
}

__host__ __device__ inline double nextDouble(uint64_t* rand)
{
    return (((uint64_t)next(rand, 26) << 27u) + next(rand, 27))
        / (double)(1ULL << 53u);
}

__host__ __device__ inline float nextFloat(uint64_t* rand) {
    return next(rand, 24) / ((float)(1 << 24));
}



///=============================================================================
///                               Xoroshiro 128
/// 
/// basic function implementations are from Cubitect's Cubiomes library:
/// https://github.com/Cubitect/cubiomes
///=============================================================================

typedef struct Xoroshiro Xoroshiro;
struct Xoroshiro {
    uint64_t lo, hi;
};

__host__ __device__ inline uint64_t rotl64(uint64_t x, uint8_t b)
{
    return (x << b) | (x >> (64 - b));
}

__host__ __device__ inline uint32_t rotr32(uint32_t a, uint8_t b)
{
    return (a >> b) | (a << (32 - b));
}

__host__ __device__ inline void xSetSeed(Xoroshiro* xr, uint64_t value)
{
    const uint64_t XL = 0x9e3779b97f4a7c15ULL;
    const uint64_t XH = 0x6a09e667f3bcc909ULL;
    const uint64_t A = 0xbf58476d1ce4e5b9ULL;
    const uint64_t B = 0x94d049bb133111ebULL;
    uint64_t l = value ^ XH;
    uint64_t h = l + XL;
    l = (l ^ (l >> 30)) * A;
    h = (h ^ (h >> 30)) * A;
    l = (l ^ (l >> 27)) * B;
    h = (h ^ (h >> 27)) * B;
    l = l ^ (l >> 31);
    h = h ^ (h >> 31);
    xr->lo = l;
    xr->hi = h;
}

__host__ __device__ inline uint64_t xNextLong(Xoroshiro* xr)
{
    uint64_t l = xr->lo;
    uint64_t h = xr->hi;
    uint64_t n = rotl64(l + h, 17) + l;
    h ^= l;
    xr->lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64(h, 28);
    return n;
}

__host__ __device__ inline int xNextInt(Xoroshiro* xr, uint32_t n)
{
    uint64_t r = (xNextLong(xr) & 0xFFFFFFFF) * n;
    if ((uint32_t)r < n)
    {
        while ((uint32_t)r < (~n + 1) % n)
        {
            r = (xNextLong(xr) & 0xFFFFFFFF) * n;
        }
    }
    return r >> 32;
}

__host__ __device__ inline double xNextDouble(Xoroshiro* xr)
{
    return (xNextLong(xr) >> (64 - 53)) * 1.1102230246251565E-16;
}

__host__ __device__ inline float xNextFloat(Xoroshiro* xr)
{
    return (xNextLong(xr) >> (64 - 24)) * 5.9604645E-8F;
}

__host__ __device__ inline void xSkipN(Xoroshiro* xr, int count)
{
    while (count-- > 0)
        xNextLong(xr);
}

__host__ __device__ inline uint64_t xNextLongJ(Xoroshiro* xr)
{
    int32_t a = xNextLong(xr) >> 32;
    int32_t b = xNextLong(xr) >> 32;
    return ((uint64_t)a << 32) + b;
}

__host__ __device__ inline int xNextIntJ(Xoroshiro* xr, uint32_t n)
{
    int bits, val;
    const int m = n - 1;

    if ((m & n) == 0) {
        uint64_t x = n * (xNextLong(xr) >> 33);
        return (int)((int64_t)x >> 31);
    }

    do {
        bits = (xNextLong(xr) >> 33);
        val = bits % n;
    } while (bits - val + m < 0);
    return val;
}

__host__ __device__ inline int xNextIntJPO2(Xoroshiro* xr, uint32_t n)
{
    uint64_t x = n * (xNextLong(xr) >> 33);
    return (int)((int64_t)x >> 31);
}

__host__ __device__ inline void xSetDecoratorSeed(Xoroshiro* xr, uint64_t worldSeed, int blockX, int blockZ, int salt) 
{
    xSetSeed(xr, worldSeed);
    uint64_t l = xNextLongJ(xr) | 1ULL;
    uint64_t m = xNextLongJ(xr) | 1ULL;
    uint64_t populationSeed = (uint64_t)blockX * l + (uint64_t)blockZ * m ^ worldSeed;

    xSetSeed(xr, populationSeed + salt);
}

__host__ __device__ inline uint64_t xGetPopulationSeed(Xoroshiro* xr, uint64_t worldSeed, int blockX, int blockZ) 
{
    xSetSeed(xr, worldSeed);
    uint64_t l = xNextLongJ(xr) | 1ULL;
    uint64_t m = xNextLongJ(xr) | 1ULL;
    uint64_t populationSeed = (uint64_t)blockX * l + (uint64_t)blockZ * m ^ worldSeed;
    return populationSeed;
}