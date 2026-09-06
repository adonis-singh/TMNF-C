/* Resident-input physics benchmark. No race, reward, observation or learner
 * work is timed. Input uploads and validation are outside the CUDA events.
 * Every tick is executed; input schedules containing respawns are rejected.
 * Build target bench_physics_cuda; arguments match bench_physics. */
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "tmnf_cuda_env.h"

static void fail(const char *s)
{
	fprintf(stderr, "bench_physics_cuda: %s\n", s);
	exit(2);
}
static void check(cudaError_t error)
{
	if (error != cudaSuccess)
		fail(cudaGetErrorString(error));
}
static uint32_t positive(const char *s)
{
	char *end;
	unsigned long n = strtoul(s, &end, 10);
	if (!*s || *end || !n || n > UINT32_MAX)
		fail("invalid positive integer");
	return (uint32_t)n;
}
static uint64_t checksum(uint64_t h, const void *data, size_t n)
{
	const uint8_t *p = (const uint8_t *)data;
	for (size_t i = 0; i < n; ++i)
		h = (h ^ p[i]) * UINT64_C(1099511628211);
	return h;
}
#define LOAD(name) \
	auto p_##name = reinterpret_cast<decltype(&name)>(dlsym(library, #name)); \
	if (!p_##name) \
	fail(dlerror())

int main(int argc, char **argv)
{
	if (argc != 7 && argc != 8)
		fail("LIBRARY TRACK VEHICLE INPUTS WORLDS REPETITIONS [TICKS]");
	uint32_t count = positive(argv[5]), repetitions = positive(argv[6]);
	TmnfTrackHeader header;
	FILE *file = fopen(argv[2], "rb");
	if (!file || fread(&header, sizeof(header), 1, file) != 1)
		fail("track header");
	fclose(file);
	file = fopen(argv[4], "rb");
	if (!file || fseek(file, 0, SEEK_END))
		fail("inputs");
	long length = ftell(file);
	if (length <= 0 || length % sizeof(TMNFRaceInputs) ||
	    (uint64_t)length / sizeof(TMNFRaceInputs) > UINT32_MAX)
		fail("input length");
	uint32_t ticks = (uint32_t)((size_t)length / sizeof(TMNFRaceInputs));
	if (argc == 8) {
		uint32_t requested = positive(argv[7]);
		if (requested > ticks)
			fail("TICKS exceeds schedule length");
		ticks = requested;
	}
	std::vector<TMNFRaceInputs> inputs(ticks);
	rewind(file);
	if (fread(inputs.data(), sizeof(inputs[0]), ticks, file) != ticks)
		fail("read inputs");
	fclose(file);
	for (const auto &input : inputs)
		if (input.respawn)
			fail("respawn schedule needs race layer");
	void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	if (!library)
		fail(dlerror());
	LOAD(TmnfTrack_Load);
	LOAD(TmnfTrack_Unload);
	LOAD(World_Create);
	LOAD(World_Destroy);
	LOAD(World_GetPhysicsWorld);
	LOAD(World_GetPlayerVehicle);
	LOAD(World_GetPlayerState);
	LOAD(World_WritePlayerGameState);
	LOAD(World_AdvanceTimer);
	LOAD(CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs);
	LOAD(CHmsZoneDynamic_PhysicsStep2);
	LOAD(TmnfVecEnv_DefaultConfig);
	LOAD(TmnfCudaVecEnv_DefaultLimits);
	LOAD(TmnfCudaVecEnv_Create);
	LOAD(TmnfCudaVecEnv_Destroy);
	LOAD(TmnfCudaVecEnv_StepDevice);
	LOAD(TmnfCudaVecEnv_CopyWorldToHost);
	TmnfTrack *track = p_TmnfTrack_Load(argv[2], header.track_sha256);
	TmnfWorld *initial = p_World_Create(track, argv[3]);
	TmnfWorld *reference = p_World_Create(track, argv[3]);
	auto *physics = p_World_GetPhysicsWorld(reference);
	auto *vehicle = p_World_GetPlayerVehicle(reference);
	for (const auto &input : inputs) {
		p_CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(&input, vehicle);
		p_World_AdvanceTimer(reference, 10);
		p_CHmsZoneDynamic_PhysicsStep2(physics, 10);
	}
	auto hash_world = [&](TmnfWorld *world) {
		uint8_t car[TMNF_CSCENE_VEHICLE_CAR_GAME_SIZE];
		uint8_t wheels[TMNF_STADIUM_WHEEL_COUNT * TMNF_CSCENE_VEHICLE_CAR_WHEEL_GAME_SIZE];
		p_World_WritePlayerGameState(world, car, wheels);
		uint64_t h = checksum(UINT64_C(14695981039346656037), p_World_GetPlayerState(world),
		                      sizeof(CHmsStateDyna));
		h = checksum(h, car, sizeof(car));
		return checksum(h, wheels, sizeof(wheels));
	};
	uint64_t expected = hash_world(reference);
	/* A bounded upload ring: at 65,536 worlds this is 144 MiB, not a whole
	 * episode's input tensor. The same buffer is reused between event pairs. */
	constexpr uint32_t chunk = 32;
	std::vector<TMNFRaceInputs> staging((size_t)count * chunk);
	TMNFRaceInputs *resident;
	check(cudaMalloc(&resident, staging.size() * sizeof(staging[0])));
	cudaStream_t stream;
	check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	cudaEvent_t start, stop;
	check(cudaEventCreate(&start));
	check(cudaEventCreate(&stop));
	TmnfVecEnvConfig config = p_TmnfVecEnv_DefaultConfig();
	TmnfCudaVecEnvLimits limits = p_TmnfCudaVecEnv_DefaultLimits();
	limits.stream = stream;
	TmnfWorld *download = p_World_Create(track, argv[3]);
	size_t cold_create_bytes = 0;
	for (uint32_t trial = 0; trial <= repetitions; ++trial) {
		size_t free_before, free_after, total;
		check(cudaMemGetInfo(&free_before, &total));
		auto *env = p_TmnfCudaVecEnv_Create(track, initial, NULL, count, &config, &limits);
		if (!env)
			fail("create returned null");
		check(cudaMemGetInfo(&free_after, &total));
		if (!trial)
			cold_create_bytes = free_before > free_after ? free_before - free_after : 0;
		double milliseconds = 0;
		for (uint32_t first = 0; first < ticks; first += chunk) {
			uint32_t n = ticks - first < chunk ? ticks - first : chunk;
			for (uint32_t tick = 0; tick < n; ++tick)
				for (uint32_t i = 0; i < count; ++i)
					staging[(size_t)tick * count + i] = inputs[first + tick];
			check(cudaMemcpyAsync(resident, staging.data(), (size_t)n * count * sizeof(staging[0]),
			                      cudaMemcpyHostToDevice, stream));
			check(cudaEventRecord(start, stream));
			for (uint32_t tick = 0; tick < n; ++tick)
				p_TmnfCudaVecEnv_StepDevice(env, resident + (size_t)tick * count, 10);
			check(cudaEventRecord(stop, stream));
			check(cudaEventSynchronize(stop));
			float elapsed;
			check(cudaEventElapsedTime(&elapsed, start, stop));
			milliseconds += elapsed;
		}
		/* Cover a full warp, a partial warp, and the final environment. */
		uint32_t checked = 0;
		for (uint32_t i = 0; i < count; ++i) {
			if (i >= 33 && i + 1 != count)
				continue;
			p_TmnfCudaVecEnv_CopyWorldToHost(env, i, download);
			if (hash_world(download) != expected)
				fail("CPU/GPU final state differs");
			++checked;
		}
		if (trial) {
			printf("{\"trial\":%u,\"worlds\":%u,\"ticks\":%" PRIu64 ",\"gpu_seconds\":%.9f,"
			       "\"ticks_per_second\":%.3f,\"state_fnv1a64\":\"%016" PRIx64
			       "\",\"validated_worlds\":%u,"
			       "\"create_device_bytes\":%zu,\"cold_create_device_bytes\":%zu,\"input_device_"
			       "bytes\":%zu}\n",
			       trial - 1, count, (uint64_t)count * ticks, milliseconds / 1000,
			       (double)count * ticks * 1000 / milliseconds, expected, checked,
			       free_before > free_after ? free_before - free_after : 0, cold_create_bytes,
			       staging.size() * sizeof(staging[0]));
			fflush(stdout);
		}
		p_TmnfCudaVecEnv_Destroy(env);
	}
	check(cudaFree(resident));
	check(cudaEventDestroy(start));
	check(cudaEventDestroy(stop));
	check(cudaStreamDestroy(stream));
	p_World_Destroy(download);
	p_World_Destroy(reference);
	p_World_Destroy(initial);
	p_TmnfTrack_Unload(track);
	dlclose(library);
}
