/* Per-warp state for the cooperative physics sections (tmnf_warp.h). */
#include "../../tmnf_warp.h"

__shared__ uint32_t tmnf_warp_masks[TMNF_WARPS_PER_BLOCK_MAX];
__shared__ void *tmnf_warp_workspaces[TMNF_WARPS_PER_BLOCK_MAX];
__shared__ uint32_t tmnf_warp_contact_capacities[TMNF_WARPS_PER_BLOCK_MAX];

__device__ void tmnf_dev_warp_begin(
	uint32_t mask, void *workspace, uint32_t contact_capacity)
{
	uint32_t warp = threadIdx.x / 32;
	if (threadIdx.x % 32 == (uint32_t)(__ffs((int)mask) - 1)) {
		tmnf_warp_masks[warp] = mask;
		tmnf_warp_workspaces[warp] = workspace;
		tmnf_warp_contact_capacities[warp] = contact_capacity;
	}
	__syncwarp(mask);
}

__device__ void tmnf_dev_warp_set_mask(uint32_t mask)
{
	if (mask == 0)
		return;
	uint32_t warp = threadIdx.x / 32;
	if (threadIdx.x % 32 == (uint32_t)(__ffs((int)mask) - 1))
		tmnf_warp_masks[warp] = mask;
	__syncwarp(mask);
}

__device__ uint32_t tmnf_dev_warp_mask(void)
{
	return tmnf_warp_masks[threadIdx.x / 32];
}

__device__ void *tmnf_dev_warp_workspace(void)
{
	return tmnf_warp_workspaces[threadIdx.x / 32];
}

__device__ uint32_t tmnf_dev_warp_contact_capacity(void)
{
	return tmnf_warp_contact_capacities[threadIdx.x / 32];
}

__device__ uint32_t tmnf_dev_warp_max_u32(uint32_t value)
{
	return __reduce_max_sync(tmnf_warp_masks[threadIdx.x / 32], value);
}
