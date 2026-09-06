/*
 * The cold half of a world (World_GetCold) is never written after
 * World_Create: the device shares one copy of it between every environment,
 * so a write there would corrupt every lane. This maps the cold half of 64
 * worlds read-only and drives them through random discrete actions with
 * respawns, resets, snapshot capture and restore. A write is a SIGSEGV and
 * the test fails.
 *
 * usage: world_cold_readonly track vehicle route sha256 ticks
 */
#define _GNU_SOURCE
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "route.h"
#include "track.h"
#include "vec_env.h"
#include "world.h"

enum { COUNT = 64 };

static void fail(const char *message)
{
	fprintf(stderr, "world_cold_readonly: %s\n", message);
	exit(1);
}

static void *allocate(size_t count, size_t size)
{
	void *memory = calloc(count, size);
	if (memory == NULL)
		fail("out of memory");
	return memory;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint32_t rng_next(void)
{
	rng_state ^= rng_state >> 12;
	rng_state ^= rng_state << 25;
	rng_state ^= rng_state >> 27;
	return (uint32_t)((rng_state * 0x2545F4914F6CDD1Dull) >> 32);
}

static uint8_t random_action(void)
{
	uint32_t roll = rng_next() % 100;
	uint32_t longitudinal = roll < 65 ? 1 : roll < 80 ? 0 : roll < 90 ? 2 : 3;
	return (uint8_t)(longitudinal * 3 + rng_next() % 3);
}

static int hex_nibble(char c)
{
	if (c >= '0' && c <= '9')
		return c - '0';
	if (c >= 'a' && c <= 'f')
		return c - 'a' + 10;
	if (c >= 'A' && c <= 'F')
		return c - 'A' + 10;
	fail("bad sha256 digit");
	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 6)
		fail("usage: track vehicle route sha256 ticks");
	uint8_t sha256[32];
	if (strlen(argv[4]) != 64)
		fail("sha256 must be 64 hex digits");
	for (int i = 0; i < 32; ++i) {
		sha256[i] = (uint8_t)(hex_nibble(argv[4][i * 2]) << 4 |
			hex_nibble(argv[4][i * 2 + 1]));
	}
	uint32_t ticks = (uint32_t)strtoul(argv[5], NULL, 10);
	TmnfTrack *track = TmnfTrack_Load(argv[1], sha256);
	TmnfRoute *route = TmnfRoute_Load(argv[3], sha256);

	TmnfWorld **owners = allocate(COUNT, sizeof(*owners));
	TmnfPhysicsWorld **physics = allocate(COUNT, sizeof(*physics));
	uint32_t *indices = allocate(COUNT, sizeof(*indices));
	for (uint32_t i = 0; i < COUNT; ++i) {
		owners[i] = World_Create(track, argv[2]);
		physics[i] = World_GetPhysicsWorld(owners[i]);
	}
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	config.max_race_ticks = 1500;
	config.horizon_ticks = 1200;
	config.off_track_grace_ticks = 60;
	config.stuck_grace_ticks = 120;
	config.respawn_action = 1;
	config.thread_count = 4;
	TmnfVecEnv env;
	TmnfVecEnv_Init(&env, physics, indices, COUNT, route, &config);

	/* Everything above may write the cold half; nothing below may. */
	size_t cold_bytes = (World_ColdSize() + 4095) & ~(size_t)4095;
	for (uint32_t i = 0; i < COUNT; ++i) {
		if (mprotect((void *)World_GetCold(owners[i]), cold_bytes, PROT_READ) != 0)
			fail("mprotect");
	}

	TmnfObservation *observations = allocate(COUNT, sizeof(*observations));
	TmnfStepResult *results = allocate(COUNT, sizeof(*results));
	TmnfEnvSnapshot *snapshots = allocate(COUNT, sizeof(*snapshots));
	uint8_t *actions = allocate(COUNT, 1);
	uint32_t *hold = allocate(COUNT, sizeof(*hold));
	TmnfVecEnv_Reset(&env, NULL, observations);
	for (uint32_t t = 0; t < ticks; ++t) {
		for (uint32_t i = 0; i < COUNT; ++i) {
			actions[i] &= (uint8_t)~TMNF_DISCRETE_RESPAWN_FLAG;
			if (hold[i] == 0) {
				actions[i] = random_action();
				if (rng_next() % 40 == 0)
					actions[i] |= TMNF_DISCRETE_RESPAWN_FLAG;
				hold[i] = 1 + rng_next() % 60;
			}
			hold[i]--;
		}
		TmnfVecEnv_StepDiscrete(&env, actions, 1 + t % 3, results);
		if (t % 100 == 50) {
			TmnfVecEnv_Capture(&env, snapshots);
			/* Rotate the snapshots one environment along and restore. */
			TmnfEnvSnapshot first = snapshots[0];
			memmove(&snapshots[0], &snapshots[1],
				(COUNT - 1) * sizeof(*snapshots));
			snapshots[COUNT - 1] = first;
			TmnfVecEnv_Restore(&env, snapshots);
		}
	}
	printf("world_cold_readonly: %" PRIu32 " ticks x %d envs, cold half of %zu "
		"bytes never written\n", ticks, COUNT, World_ColdSize());
	TmnfVecEnv_Destroy(&env);
	for (uint32_t i = 0; i < COUNT; ++i) {
		if (mprotect((void *)World_GetCold(owners[i]), cold_bytes,
			PROT_READ | PROT_WRITE) != 0)
			fail("mprotect for teardown");
		World_Destroy(owners[i]);
	}
	free(owners);
	free(physics);
	free(indices);
	free(observations);
	free(results);
	free(snapshots);
	free(actions);
	free(hold);
	TmnfRoute_Unload(route);
	TmnfTrack_Unload(track);
	return 0;
}
