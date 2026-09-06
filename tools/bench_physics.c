/* Engine-only benchmark using real per-tick input schedules.
 *
 * cc -O3 -std=gnu11 -Isrc tools/bench_physics.c -ldl -lm -o build/bench_physics
 * bench_physics LIBRARY TRACK VEHICLE INPUTS WORLDS REPETITIONS [TICKS]
 *
 * Only input application, the game timer and PhysicsStep2 are timed. No
 * observation construction, rewards, race rules, Python or learner runs.
 * Fresh worlds are created outside every timed repetition. Input schedules
 * with respawns are rejected because respawns need the race layer.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "world.h"

static void fail(const char *message)
{
	fprintf(stderr, "bench_physics: %s\n", message);
	exit(2);
}

static double now(void)
{
	struct timespec t;
	if (clock_gettime(CLOCK_MONOTONIC_RAW, &t)) fail("clock_gettime");
	return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static uint32_t positive(const char *s)
{
	char *end;
	unsigned long n = strtoul(s, &end, 10);
	if (!*s || *end || !n || n > UINT32_MAX) fail("invalid positive integer");
	return (uint32_t)n;
}

static void *allocate(size_t count, size_t size)
{
	if (size && count > SIZE_MAX / size) fail("allocation overflow");
	void *p = calloc(count, size);
	if (!p) fail("allocation failed");
	return p;
}

static uint64_t checksum(uint64_t h, const void *data, size_t size)
{
	const uint8_t *p = data;
	for (size_t i = 0; i < size; ++i) h = (h ^ p[i]) * UINT64_C(1099511628211);
	return h;
}

#define LOAD(name) \
	__typeof__(&name) p_##name; \
	do { \
		void *symbol = dlsym(library, #name); \
		if (!symbol) fail(dlerror()); \
		_Static_assert(sizeof(symbol) == sizeof(p_##name), "function pointer size"); \
		memcpy(&p_##name, &symbol, sizeof(symbol)); \
	} while (0)

int main(int argc, char **argv)
{
	if (argc != 7 && argc != 8) {
		fprintf(stderr, "usage: %s LIBRARY TRACK VEHICLE INPUTS WORLDS REPETITIONS [TICKS]\n", argv[0]);
		return 2;
	}
	uint32_t count = positive(argv[5]), repetitions = positive(argv[6]);
	FILE *file = fopen(argv[2], "rb");
	TmnfTrackHeader header;
	if (!file || fread(&header, sizeof(header), 1, file) != 1) fail("track header");
	fclose(file);
	file = fopen(argv[4], "rb");
	if (!file || fseek(file, 0, SEEK_END)) fail("input schedule");
	long length = ftell(file);
	if (length <= 0 || length % sizeof(TMNFRaceInputs) ||
		(uint64_t)length / sizeof(TMNFRaceInputs) > UINT32_MAX) fail("input length");
	uint32_t ticks = (uint32_t)((size_t)length / sizeof(TMNFRaceInputs));
	if (argc == 8) {
		uint32_t requested = positive(argv[7]);
		if (requested > ticks) fail("TICKS exceeds schedule length");
		ticks = requested;
	}
	TMNFRaceInputs *inputs = allocate(ticks, sizeof(*inputs));
	rewind(file);
	if (fread(inputs, sizeof(*inputs), ticks, file) != ticks) fail("read inputs");
	fclose(file);
	for (uint32_t i = 0; i < ticks; ++i)
		if (inputs[i].respawn) fail("respawn schedule needs the race layer");

	void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	if (!library) fail(dlerror());
	LOAD(TmnfTrack_Load); LOAD(TmnfTrack_Unload);
	LOAD(World_Create); LOAD(World_Destroy);
	LOAD(World_GetPhysicsWorld); LOAD(World_GetPlayerVehicle);
	LOAD(World_GetPlayerState); LOAD(World_WritePlayerGameState);
	LOAD(World_AdvanceTimer);
	LOAD(CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs);
	LOAD(CHmsZoneDynamic_PhysicsStep2);
	double load_start = now();
	TmnfTrack *track = p_TmnfTrack_Load(argv[2], header.track_sha256);
	double load_seconds = now() - load_start;
	TmnfWorld **worlds = allocate(count, sizeof(*worlds));
	TmnfPhysicsWorld **physics = allocate(count, sizeof(*physics));
	CSceneVehicleCar **vehicles = allocate(count, sizeof(*vehicles));
	/* Diagnose motion in a separate world, outside all timed repetitions.
	 * A mixed-control capture can spend part of its trajectory stationary. */
	TmnfWorld *diagnostic = p_World_Create(track, argv[3]);
	CSceneVehicleCar *diagnostic_vehicle = p_World_GetPlayerVehicle(diagnostic);
	TmnfPhysicsWorld *diagnostic_physics = p_World_GetPhysicsWorld(diagnostic);
	double motion_sum = 0;
	uint32_t moving_ticks = 0;
	for (uint32_t tick = 0; tick < ticks; ++tick) {
		p_CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(&inputs[tick], diagnostic_vehicle);
		p_World_AdvanceTimer(diagnostic, 10);
		p_CHmsZoneDynamic_PhysicsStep2(diagnostic_physics, 10);
		const GmVec3 *v = &p_World_GetPlayerState(diagnostic)->linVel;
		double speed = sqrt((double)v->x * v->x + (double)v->y * v->y + (double)v->z * v->z);
		motion_sum += speed;
		moving_ticks += speed > 1.0;
	}
	p_World_Destroy(diagnostic);
	uint64_t expected_hash = 0;
	for (uint32_t repetition = 0; repetition <= repetitions; ++repetition) {
		for (uint32_t i = 0; i < count; ++i) {
			worlds[i] = p_World_Create(track, argv[3]);
			physics[i] = p_World_GetPhysicsWorld(worlds[i]);
			vehicles[i] = p_World_GetPlayerVehicle(worlds[i]);
		}
		double start = now();
		for (uint32_t tick = 0; tick < ticks; ++tick) {
			for (uint32_t i = 0; i < count; ++i) {
				p_CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(&inputs[tick], vehicles[i]);
				p_World_AdvanceTimer(worlds[i], 10);
				p_CHmsZoneDynamic_PhysicsStep2(physics[i], 10);
			}
		}
		double elapsed = now() - start;
		uint64_t h = UINT64_C(14695981039346656037);
		double speed_sum = 0;
		for (uint32_t i = 0; i < count; ++i) {
			const CHmsStateDyna *state = p_World_GetPlayerState(worlds[i]);
			uint8_t car[TMNF_CSCENE_VEHICLE_CAR_GAME_SIZE];
			uint8_t wheels[TMNF_STADIUM_WHEEL_COUNT * TMNF_CSCENE_VEHICLE_CAR_WHEEL_GAME_SIZE];
			p_World_WritePlayerGameState(worlds[i], car, wheels);
			h = checksum(h, state, sizeof(*state));
			h = checksum(h, car, sizeof(car));
			h = checksum(h, wheels, sizeof(wheels));
			const GmVec3 *v = &state->linVel;
			speed_sum += sqrt((double)v->x * v->x + (double)v->y * v->y + (double)v->z * v->z);
			p_World_Destroy(worlds[i]);
		}
		if (!repetition) expected_hash = h;
		else {
			if (h != expected_hash) fail("repeat changed final state");
			struct rusage usage;
			if (getrusage(RUSAGE_SELF, &usage)) fail("getrusage");
			printf("{\"trial\":%u,\"worlds\":%u,\"ticks\":%" PRIu64
				",\"seconds\":%.9f,\"ticks_per_second\":%.3f,\"final_speed_mps\":%.9f,"
				"\"state_fnv1a64\":\"%016" PRIx64 "\",\"track_load_seconds\":%.9f,"
				"\"peak_rss_kib\":%ld,\"mean_speed_mps\":%.9f,\"moving_tick_fraction\":%.9f}\n",
				repetition - 1, count, (uint64_t)count * ticks, elapsed,
				(double)count * ticks / elapsed, speed_sum / count, h,
				load_seconds, usage.ru_maxrss, motion_sum / ticks, (double)moving_ticks / ticks);
			fflush(stdout);
		}
	}
	free(vehicles); free(physics); free(worlds); free(inputs);
	p_TmnfTrack_Unload(track);
	dlclose(library);
	return 0;
}
