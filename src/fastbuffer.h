#ifndef TMNF_FASTBUFFER_H
#define TMNF_FASTBUFFER_H

#include "tmnf_hd.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SHmsPhysicalCollision;

/* 32-bit game layout. Native storage is kept separately below. */
typedef struct {
	uint32_t count;
	uint32_t data;
	uint32_t capacity;
} CFastBufferLayout32;

_Static_assert(sizeof(CFastBufferLayout32) == 0x0c, "CFastBuffer layout size");
_Static_assert(offsetof(CFastBufferLayout32, count) == 0x00, "CFastBuffer count");
_Static_assert(offsetof(CFastBufferLayout32, data) == 0x04, "CFastBuffer data");
_Static_assert(offsetof(CFastBufferLayout32, capacity) == 0x08, "CFastBuffer capacity");

typedef struct {
	uint32_t count;
	struct SHmsPhysicalCollision *data;
	uint32_t capacity;
} CFastBuffer_SHmsPhysicalCollision;

typedef int (*SHmsPhysicalCollisionCompare)(
	const struct SHmsPhysicalCollision *,
	const struct SHmsPhysicalCollision *);

TMNF_HD void CFastBuffer_SHmsPhysicalCollision_Init(
	CFastBuffer_SHmsPhysicalCollision *self);
TMNF_HD void CFastBuffer_SHmsPhysicalCollision_Destroy(
	CFastBuffer_SHmsPhysicalCollision *self);

/* 0x00536DA0  Returns the indexed 0x4c-byte collision record. */
/* UNVALIDATED */
TMNF_HD struct SHmsPhysicalCollision *CFastBuffer_SHmsPhysicalCollision_At(
	CFastBuffer_SHmsPhysicalCollision *self, uint32_t index);

/* 0x00537370  Grows capacity with the game's 1.5x policy. */
/* UNVALIDATED */
TMNF_HD void CFastBuffer_SHmsPhysicalCollision_SetSizeAtLeast(
	CFastBuffer_SHmsPhysicalCollision *self, uint32_t requested);

/* 0x00537400  Appends one uninitialized collision record. */
/* UNVALIDATED */
TMNF_HD struct SHmsPhysicalCollision *CFastBuffer_SHmsPhysicalCollision_AddNewElem(
	CFastBuffer_SHmsPhysicalCollision *self);

/* 0x00547DE0  Sorts records with the game's MSVC quicksort implementation. */
/* UNVALIDATED */
TMNF_HD void CFastBuffer_SHmsPhysicalCollision_QSort(
	CFastBuffer_SHmsPhysicalCollision *self,
	SHmsPhysicalCollisionCompare compare);

#ifdef __cplusplus
}
#endif

#endif
