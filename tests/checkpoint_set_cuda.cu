/* Run explicitly on a CUDA device: large checkpoint and finish bookkeeping. */
#include <cuda_runtime.h>
#include <stdio.h>
#include "race.h"

struct Scene {
	TmnfRouteMetadata metadata;
	TmnfRouteStart start;
	TmnfRouteTrigger checkpoints[88], finishes[2];
	TmnfRouteReferencePoint points[90];
	uint32_t failure;
};

#define CHECK(condition, code) do { if (!(condition)) { scene->failure = code; return; } } while (0)

__global__ static void check_race(Scene *scene)
{
	TmnfRoute route = {};
	route.metadata = &scene->metadata;
	route.start = &scene->start;
	route.checkpoints = scene->checkpoints;
	route.finish = scene->finishes;
	route.centerline = scene->points;
	TmnfRaceState state = {};
	state.lap_limit = 2;
	for (uint32_t lap = 0; lap < 2; ++lap) {
		TmnfRaceStepResult result;
		TmnfRace_BeginStep(&route, &state, &result);
		(void)TmnfRace_ProcessContact(&route, &state, 87, &result);
		(void)TmnfRace_ProcessContact(&route, &state, 87, &result);
		CHECK(state.visited_count == 1 && result.checkpoint_repeated, 1);
		for (uint32_t i = 0; i < 87; ++i)
			(void)TmnfRace_ProcessContact(&route, &state, i, &result);
		CHECK(TmnfCheckpointSet_All(&state.visited, 88), 2);
		(void)TmnfRace_ProcessContact(&route, &state, 88 + lap, &result);
		CHECK(result.lap_completed && state.completed_laps == lap + 1, 3);
		if (lap == 0) CHECK(!TmnfCheckpointSet_Test(&state.visited, 87), 4);
	}
	CHECK(state.finished && state.finish_time_ms == 20 && state.passed_race_checkpoints == 178, 5);
	TmnfCheckpointSet set = {};
	for (uint32_t i = 0; i < TMNF_CHECKPOINT_CAPACITY; ++i) TmnfCheckpointSet_Add(&set, i);
	CHECK(TmnfCheckpointSet_All(&set, TMNF_CHECKPOINT_CAPACITY), 6);
}

int main()
{
	Scene *scene;
	if (cudaMallocManaged(&scene, sizeof(*scene)) != cudaSuccess) return 2;
	*scene = Scene{};
	scene->metadata.lap_count = 2;
	scene->metadata.checkpoint_count = 88;
	scene->metadata.finish_count = 2;
	scene->metadata.reference_count = scene->metadata.centerline_count = 90;
	scene->metadata.total_race_checkpoints = scene->metadata.race_checkpoint_limit = 178;
	scene->metadata.flags = TMNF_ROUTE_MULTILAP;
	check_race<<<1, 1>>>(scene);
	cudaError_t error = cudaDeviceSynchronize();
	uint32_t failure = scene->failure;
	cudaFree(scene);
	if (error != cudaSuccess || failure) {
		fprintf(stderr, "checkpoint CUDA failure %u: %s\n", failure, cudaGetErrorString(error));
		return 1;
	}
	puts("CUDA: 88-checkpoint laps, independent finishes and all 2048 set bits passed");
	return 0;
}
