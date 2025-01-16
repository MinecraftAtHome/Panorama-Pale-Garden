#ifndef MCCORE_H_
#define MCCORE_H_

#include <cinttypes>

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

typedef struct SeedConstants SeedConstants;
struct SeedConstants {
    uint64_t worldseed;
    uint64_t A; // gets multiplied by chunk x in population seeding
    uint64_t B; // gets multiplied by chunk z in population seeding
};

#endif // MCCORE_H_