/* Pointer-free checkpoint/contact storage, shared by CPU, CUDA and snapshots.
 * 2048 bits also fit the collection frame's 64 packed uint32_t words. */
#ifndef TMNF_CHECKPOINT_SET_H
#define TMNF_CHECKPOINT_SET_H

#include "tmnf_hd.h"
#include <stdint.h>
#include <string.h>

enum { TMNF_CHECKPOINT_CAPACITY = 2048,
       TMNF_CHECKPOINT_WORDS = TMNF_CHECKPOINT_CAPACITY / 64 };
typedef struct { uint64_t words[TMNF_CHECKPOINT_WORDS]; } TmnfCheckpointSet;

TMNF_HD static inline void TmnfCheckpointSet_Clear(TmnfCheckpointSet *set)
{
	memset(set, 0, sizeof(*set));
}

TMNF_HD static inline int TmnfCheckpointSet_Test(const TmnfCheckpointSet *set, uint32_t index)
{
	return (set->words[index / 64] >> (index % 64)) & 1;
}

TMNF_HD static inline void TmnfCheckpointSet_Add(TmnfCheckpointSet *set, uint32_t index)
{
	set->words[index / 64] |= UINT64_C(1) << (index % 64);
}

TMNF_HD static inline int TmnfCheckpointSet_All(const TmnfCheckpointSet *set, uint32_t count)
{
	for (uint32_t i = 0; i < count / 64; ++i)
		if (set->words[i] != UINT64_MAX) return 0;
	uint32_t tail = count % 64;
	uint64_t mask = (UINT64_C(1) << tail) - 1;
	return tail == 0 || (set->words[count / 64] & mask) == mask;
}

#endif
