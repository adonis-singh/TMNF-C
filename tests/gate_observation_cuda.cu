/* Compare all output bytes through independent host and device compilation. */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "race_observation.h"
struct Scene {
	TmnfRouteMetadata metadata;
	TmnfRouteTrigger checkpoints[63], finish[64];
	TmnfRaceState state;
	GmIso4 car;
};
static void check(cudaError_t e) {
	if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); exit(1); }
}
__host__ __device__ static void observe(const Scene *scene, TmnfGateObservations *out) {
	TmnfRoute route = {};
	route.metadata = &scene->metadata;
	route.checkpoints = scene->checkpoints;
	route.finish = scene->finish;
	TmnfRace_ObserveGates(&route, &scene->state, &scene->car, out);
}
__global__ static void observe_kernel(const Scene *scenes, TmnfGateObservations *out, int count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < count) observe(scenes + i, out + i);
}
static uint32_t random_state = 19;
static uint32_t next(void) { random_state = random_state * 1664525u + 1013904223u; return random_state; }
static float number(void) { return ((int)(next() % 20001) - 10000) / 16.0f; }
static GmIso4 pose(void) {
	GmIso4 out = {};
	float c = .6f, s = (next() & 1) ? .8f : -.8f;
	out.m[0] = c; out.m[2] = -s; out.m[4] = 1; out.m[6] = s; out.m[8] = c;
	for (int i = 0; i < 3; ++i) out.t[i] = number();
	return out;
}
int main(void) {
	const int count = 1024;
	Scene *scenes; TmnfGateObservations *out;
	check(cudaMallocManaged(&scenes, count * sizeof(*scenes)));
	check(cudaMallocManaged(&out, count * sizeof(*out)));
	memset(scenes, 0, count * sizeof(*scenes));
	for (int i = 0; i < count; ++i) {
		Scene &scene = scenes[i];
		scene.metadata.checkpoint_count = i % 64;
		scene.metadata.finish_count = 1 + (i / 64) % (64 - scene.metadata.checkpoint_count); scene.metadata.lap_count = 3;
		scene.state.completed_laps = i % 3; scene.state.finished = (i % 31 == 0);
		scene.state.visited_checkpoints = ((uint64_t)next() << 32) | next();
		scene.car = pose();
		for (int j = 0; j < 64; ++j) {
			TmnfRouteTrigger &gate = j == 63 ? scene.finish[0] : scene.checkpoints[j];
			gate.transform = pose();
			gate.box.center = {number(), number(), number()};
			gate.box.half_extent = {1.25f, 2.5f, 3.75f};
			gate.no_respawn = j % 2;
		}
		for (uint32_t j = 1; j < scene.metadata.finish_count; ++j) {
			scene.finish[j] = scene.finish[0];
			scene.finish[j].transform = pose();
		}
	}
	observe_kernel<<<(count + 31) / 32, 32>>>(scenes, out, count);
	check(cudaGetLastError()); check(cudaDeviceSynchronize());
	for (int i = 0; i < count; ++i) {
		TmnfGateObservations host;
		observe(scenes + i, &host);
		if (memcmp(&host, out + i, sizeof(host))) {
			fprintf(stderr, "gate observation CPU/CUDA mismatch at %d\n", i); return 1;
		}
	}
	check(cudaFree(out)); check(cudaFree(scenes));
	puts("1024 gate scenes match all 656 output bytes on CPU/CUDA");
}
