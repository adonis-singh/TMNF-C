/* Device transcendentals that reproduce this machine's glibc bit for bit.
 *
 * The CPU port evaluates sin, cos, exp and atan2 in glibc double precision on
 * float-valued arguments and rounds the result to float (src/tmnf_fp.h). CUDA's
 * double libm is accurate to 1-2 ulp but is not glibc, so after the float
 * rounding a handful of inputs out of 2^32 land on the other side of a
 * rounding boundary. tools/cuda_libm_hardcases.cu enumerates every float
 * input (every reachable argument pair for atan2), records the inputs where
 * the CUDA result differs from glibc into tmnf_libm_hardcases.h, and verifies
 * that the corrected functions below agree on all of them. The table is tied
 * to the glibc and CUDA toolkit versions recorded in that header:
 * TmnfCudaVecEnv_Create refuses to run under any other, and the
 * cuda_libm_hardcases ctest rescans all five domains (25 s).
 *
 * Included from src/tmnf_fp.h in device compilation only.
 */
#ifndef TMNF_DEV_LIBM_H
#define TMNF_DEV_LIBM_H

#include <stdint.h>

typedef struct {
	uint32_t input;
	uint32_t output;
} TmnfLibmHardCase1;

typedef struct {
	uint32_t y;
	uint32_t x;
	uint32_t output;
} TmnfLibmHardCase2;

#include "tmnf_libm_hardcases.h"

__device__ static inline uint32_t tmnf_dev_float_bits(float value)
{
	return __float_as_uint(value);
}

__device__ static inline float tmnf_dev_bits_float(uint32_t bits)
{
	return __uint_as_float(bits);
}

/* Tables are sorted by input bits; a binary search costs log2(n) loads. */
__device__ static inline float tmnf_dev_correct1(
	const TmnfLibmHardCase1 *table, uint32_t count, float input,
	float candidate)
{
	uint32_t key = tmnf_dev_float_bits(input);
	uint32_t lower = 0;
	uint32_t upper = count;
	while (lower < upper) {
		uint32_t middle = lower + (upper - lower) / 2;
		uint32_t probe = table[middle].input;
		if (probe == key)
			return tmnf_dev_bits_float(table[middle].output);
		if (probe < key)
			lower = middle + 1;
		else
			upper = middle;
	}
	return candidate;
}

__device__ static inline float tmnf_dev_correct2(
	const TmnfLibmHardCase2 *table, uint32_t count, float y, float x,
	float candidate)
{
	uint32_t key_y = tmnf_dev_float_bits(y);
	uint32_t key_x = tmnf_dev_float_bits(x);
	uint32_t lower = 0;
	uint32_t upper = count;
	while (lower < upper) {
		uint32_t middle = lower + (upper - lower) / 2;
		uint32_t probe_y = table[middle].y;
		uint32_t probe_x = table[middle].x;
		if (probe_y == key_y && probe_x == key_x)
			return tmnf_dev_bits_float(table[middle].output);
		if (probe_y < key_y || (probe_y == key_y && probe_x < key_x))
			lower = middle + 1;
		else
			upper = middle;
	}
	return candidate;
}

__device__ static inline float tmnf_dev_sin_r24(float x)
{
	float candidate = (float)sin((double)x);
	return tmnf_dev_correct1(
		TMNF_LIBM_SIN_HARD, TMNF_LIBM_SIN_HARD_COUNT, x, candidate);
}

__device__ static inline float tmnf_dev_cos_r24(float x)
{
	float candidate = (float)cos((double)x);
	return tmnf_dev_correct1(
		TMNF_LIBM_COS_HARD, TMNF_LIBM_COS_HARD_COUNT, x, candidate);
}

__device__ static inline float tmnf_dev_exp_r24(float x)
{
	float candidate = (float)exp((double)x);
	return tmnf_dev_correct1(
		TMNF_LIBM_EXP_HARD, TMNF_LIBM_EXP_HARD_COUNT, x, candidate);
}

__device__ static inline float tmnf_dev_atan2_r24(float y, float x)
{
	float candidate = (float)atan2((double)y, (double)x);
	return tmnf_dev_correct2(
		TMNF_LIBM_ATAN2_HARD, TMNF_LIBM_ATAN2_HARD_COUNT, y, x,
		candidate);
}

#endif /* TMNF_DEV_LIBM_H */
