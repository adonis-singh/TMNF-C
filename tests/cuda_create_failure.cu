/* A rejected reservation must return an error and release partial allocations.
 * The CUDA context must still support a subsequent ordinary environment. */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda/tmnf_cuda_env.h"

static void check(cudaError_t error)
{
	if (error != cudaSuccess) {
		fprintf(stderr, "create failure test: %s\n", cudaGetErrorString(error));
		exit(1);
	}
}

int main(int argc, char **argv)
{
	if (argc != 3)
		return 2;
	TmnfTrackHeader header;
	FILE *file = fopen(argv[1], "rb");
	if (!file || fread(&header, sizeof(header), 1, file) != 1)
		return 2;
	fclose(file);
	TmnfTrack *track = TmnfTrack_Load(argv[1], header.track_sha256);
	TmnfWorld *world = World_Create(track, argv[2]);
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	TmnfCudaVecEnvLimits limits = TmnfCudaVecEnv_DefaultLimits();
	TmnfCudaVecEnv *warm = TmnfCudaVecEnv_Create(track, world, NULL, 1, &config, &limits);
	if (!warm)
		return 1;
	TmnfCudaVecEnv_Destroy(warm);
	size_t before, total, after;
	check(cudaMemGetInfo(&before, &total));
	for (int attempt = 0; attempt < 3; ++attempt) {
		/* 4.7 TiB of world state: rejected by cudaMalloc, without a large
		 * host allocation or touching the device allocation's pages. */
		TmnfCudaVecEnv *failed =
		    TmnfCudaVecEnv_Create(track, world, NULL, UINT32_C(1) << 30, &config, &limits);
		if (failed != NULL || strstr(TmnfCudaVecEnv_LastError(), "memory") == NULL)
			return 1;
	}
	check(cudaMemGetInfo(&after, &total));
	/* Free memory is device-wide and other jobs may allocate concurrently.
	 * Leak freedom is checked by compute-sanitizer, not by this noisy delta. */
	printf("create failure: free device bytes before %zu, after %zu\n", before, after);
	TmnfCudaVecEnv *env = TmnfCudaVecEnv_Create(track, world, NULL, 1, &config, &limits);
	if (!env)
		return 1;
	TMNFRaceInputs inputs = {};
	TmnfCudaVecEnv_Step(env, &inputs, 10, NULL);
	TmnfCudaVecEnv_Destroy(env);
	World_Destroy(world);
	TmnfTrack_Unload(track);
	puts("create failure: 3 rejected reservations, subsequent step passed");
}
