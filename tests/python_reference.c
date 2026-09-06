#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "route.h"
#include "vec_env.h"
#include "world.h"

enum {
	ENVIRONMENT_COUNT = 4,
	REFERENCE_STEPS = 64,
	ACTION_REPEAT = 2,
};

static const uint8_t A01_SHA256[32] = {
	0xf0, 0xa8, 0x70, 0x80, 0x9b, 0xe9, 0x9d, 0xa2,
	0xcb, 0x36, 0xad, 0x5d, 0xf4, 0x3a, 0x2c, 0xf6,
	0x3d, 0x8f, 0x74, 0xfe, 0x4a, 0xc3, 0x47, 0x0e,
	0xca, 0xc6, 0x8b, 0x9e, 0x97, 0x62, 0x5d, 0xc3,
};

static void fail(const char *message)
{
	fprintf(stderr, "python reference: %s\n", message);
	exit(1);
}

static void write_exact(
	FILE *file, const void *data, size_t size, size_t count)
{
	if (fwrite(data, size, count, file) != count)
		fail("cannot write output");
}

int main(int argc, char **argv)
{
	if (argc != 5) {
		fprintf(stderr,
			"usage: python_reference TRACK VEHICLE ROUTE OUTPUT\n");
		return 2;
	}
	TmnfTrack *track = TmnfTrack_Load(argv[1], A01_SHA256);
	TmnfWorld *owners[ENVIRONMENT_COUNT];
	TmnfPhysicsWorld *worlds[ENVIRONMENT_COUNT];
	uint32_t player_indices[ENVIRONMENT_COUNT] = {0};
	for (uint32_t i = 0; i < ENVIRONMENT_COUNT; ++i) {
		owners[i] = World_Create(track, argv[2]);
		worlds[i] = World_GetPhysicsWorld(owners[i]);
	}
	TmnfRoute *route = TmnfRoute_Load(argv[3], A01_SHA256);
	/* Default race budget and derived horizon: any bound below the
	 * speed-cap minimum (796 ticks for A01) aborts in TmnfVecEnv_Init. */
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	config.thread_count = 1;
	config.off_track_grace_ticks = 50;
	config.stuck_grace_ticks = 80;
	TmnfVecEnv env;
	TmnfVecEnv_Init(
		&env, worlds, player_indices, ENVIRONMENT_COUNT, route, &config);

	TmnfObservation reset_observations[ENVIRONMENT_COUNT];
	TmnfVecEnv_Reset(&env, NULL, reset_observations);

	FILE *output = fopen(argv[4], "wb");
	if (output == NULL)
		fail("cannot open output");
	const uint32_t header[4] = {
		ENVIRONMENT_COUNT,
		REFERENCE_STEPS,
		sizeof(TmnfObservation),
		sizeof(TmnfStepResult),
	};
	write_exact(output, header, sizeof(header), 1);
	write_exact(
		output, reset_observations,
		sizeof(reset_observations[0]), ENVIRONMENT_COUNT);

	uint8_t actions[ENVIRONMENT_COUNT];
	TmnfStepResult results[ENVIRONMENT_COUNT];
	for (uint32_t step = 0; step < REFERENCE_STEPS; ++step) {
		for (uint32_t i = 0; i < ENVIRONMENT_COUNT; ++i)
			actions[i] = (uint8_t)((step * 5 + i * 7) % 12);
		TmnfVecEnv_StepDiscrete(
			&env, actions, ACTION_REPEAT, results);
		write_exact(output, results, sizeof(results[0]), ENVIRONMENT_COUNT);
	}
	if (fclose(output) != 0)
		fail("cannot close output");

	TmnfVecEnv_Destroy(&env);
	TmnfRoute_Unload(route);
	for (uint32_t i = 0; i < ENVIRONMENT_COUNT; ++i)
		World_Destroy(owners[i]);
	TmnfTrack_Unload(track);
	printf(
		"python reference: %u environments, %u steps written\n",
		ENVIRONMENT_COUNT, REFERENCE_STEPS);
	return 0;
}
