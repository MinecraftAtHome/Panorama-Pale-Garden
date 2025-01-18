#pragma once
#include "mccore.h"

#define TEST_FILE_AIRBLOCKS "data/no_air.txt"

constexpr uint64_t RANDOM_SEEDS_TOTAL = 1ULL << 48;
constexpr uint64_t THREADS_LAUNCHED_PER_RUN = 1ULL << 31;
constexpr uint64_t RANDOM_SEEDS_PER_RUN = THREADS_LAUNCHED_PER_RUN;
constexpr int NUM_RUNS_RANDOM_SEEDS = (RANDOM_SEEDS_TOTAL + RANDOM_SEEDS_PER_RUN - 1) / RANDOM_SEEDS_PER_RUN;

// Test units
// -----------------------------------------------
// random stuff
//#define TEST_FILE_FLOWERS "data/rstest/t1.txt"
//constexpr uint64_t CORRECT_SEED = 1923902813954367147LL;
// -----------------------------------------------
// random stuff
//#define TEST_FILE_FLOWERS "data/rstest/t2.txt"
//constexpr uint64_t CORRECT_SEED = -140586064374340817LL;
// -----------------------------------------------
// filter3 correctness - 1 patch
//#define TEST_FILE_FLOWERS "data/rstest/t3.txt"
//constexpr uint64_t CORRECT_SEED = -8839342471543779883LL;
// -----------------------------------------------
// filter3 correctness - 1 large patch + 1 other flower
//#define TEST_FILE_FLOWERS "data/rstest/t4.txt"
//constexpr uint64_t CORRECT_SEED = 8850313098065137364ULL;
// -----------------------------------------------
// filter2 and filter3 correctness - a lot of flowers from unrelated patches
//#define TEST_FILE_FLOWERS "data/rstest/t5.txt"
//constexpr uint64_t CORRECT_SEED = 719929476363093545LL;
// -----------------------------------------------
// supertest - a shitton of random flowers
//#define TEST_FILE_FLOWERS "data/rstest/t6_big.txt"
//constexpr uint64_t CORRECT_SEED = 4243907523119604105LL;
// -----------------------------------------------
// supertest #2 - maximum size
//#define TEST_FILE_FLOWERS "data/rstest/t7_max.txt"
//constexpr uint64_t CORRECT_SEED = -2385567111971732643LL;
// -----------------------------------------------
// supertest #3 - maximum size
//#define TEST_FILE_FLOWERS "data/rstest/t8_max.txt"
//constexpr uint64_t CORRECT_SEED = 6199487863832639085LL;
// -----------------------------------------------
// supertest #4 - maximum size, most from patch extremes
//#define TEST_FILE_FLOWERS "data/rstest/t9_max.txt"
//constexpr uint64_t CORRECT_SEED = 3151648800121342591LL;
// -----------------------------------------------


inline int findRun(uint64_t seed)
{
    const uint64_t firstNext32 = seed >> 32;
    const uint64_t state = firstNext32 << 16;
	return state / RANDOM_SEEDS_PER_RUN;
}