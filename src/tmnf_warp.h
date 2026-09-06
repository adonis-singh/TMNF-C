/* Warp cooperation for the device build (docs/CUDA.md, "Phase 2").
 *
 * The physics of one environment runs on one thread. Where the work per
 * environment varies a lot between the lanes of a warp (the collision
 * detection: the number of substeps, of static tree queries and of face
 * tests), the device version pools it across the warp instead of letting
 * one lane keep 31 waiting. Such a section is a warp-collective call: every
 * lane in the warp's mask has to make it, in the same sequence, and a lane
 * with no work of its own passes active = 0 and only accompanies the others.
 *
 * The mask names the lanes that are stepping together. The kernel sets it
 * (tmnf_dev_warp_begin) and shrinks it when a lane leaves the tick loop
 * early (tmnf_dev_warp_set_mask). The workspace is the warp's global scratch
 * for the cooperative sections (src/cuda/dev/collision.cu lays it out).
 *
 * On the host every helper is the identity: the CPU build is unchanged. */
#ifndef TMNF_WARP_H
#define TMNF_WARP_H

#include <stddef.h>
#include <stdint.h>

#include "tmnf_hd.h"

enum { TMNF_WARPS_PER_BLOCK_MAX = 4 };

#if defined(__CUDACC__)
__device__ void tmnf_dev_warp_begin(
	uint32_t mask, void *workspace, uint32_t contact_capacity);
__device__ void tmnf_dev_warp_set_mask(uint32_t mask);
__device__ uint32_t tmnf_dev_warp_mask(void);
__device__ void *tmnf_dev_warp_workspace(void);
__device__ uint32_t tmnf_dev_warp_contact_capacity(void);
__device__ uint32_t tmnf_dev_warp_max_u32(uint32_t value);
/* Bytes of workspace one warp needs (src/cuda/dev/collision.cu). */
size_t tmnf_dev_warp_workspace_bytes(uint32_t contact_capacity);
#endif

/* The largest value across the warp's mask; the value itself on the host. */
TMNF_HD static inline uint32_t tmnf_warp_max_u32(uint32_t value)
{
#if defined(__CUDA_ARCH__)
	return tmnf_dev_warp_max_u32(value);
#else
	return value;
#endif
}

#endif /* TMNF_WARP_H */
