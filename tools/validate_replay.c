/*
 * Replay-validation simulator: steps an exact per-tick input schedule through
 * the physics and the race layer, writes the rigid-body position after every
 * tick to a trace file, and prints every race event. Stops on the tick the
 * race finishes. Driven by tools/validate_replay.py.
 *
 * usage: validate_replay_sim TRACK VEHICLE ROUTE INPUTS TRACK_SHA256 TRACE
 *
 * TRACE: float32 x,y,z per tick; index t is the position entering step
 *        t + 1: after t steps and after the respawn that step applies, if
 *        any; index 0 the spawn, the last index the final state. The game
 *        labels the state after k steps with race time (k + 1) * 10 ms
 *        (tests/replay_tick.c captures) and its ghost sample at race time T
 *        is taken after the respawn press at T has moved the car (the E04
 *        record's sample at 42400 is the checkpoint spawn), so a ghost sample
 *        at race time T lives at index T / 10 - 1.
 * stdout: one line per event
 *   checkpoint tick=N race_time_ms=M index=I
 *   checkpoint_repeated tick=N race_time_ms=M index=I
 *   lap tick=N race_time_ms=M laps=L
 *   respawn tick=N race_time_ms=M
 *   finish tick=N race_time_ms=M
 *   ticks N
 */
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "race.h"
#include "world.h"

enum { TICK_MS = 10 };

static void fail(const char *message)
{
	fprintf(stderr, "validate_replay_sim: %s\n", message);
	exit(2);
}

static uint8_t hex_nibble(char character)
{
	if (character >= '0' && character <= '9')
		return (uint8_t)(character - '0');
	character = (char)tolower((unsigned char)character);
	if (character >= 'a' && character <= 'f')
		return (uint8_t)(character - 'a' + 10);
	fail("track SHA-256 is not hexadecimal");
	return 0;
}

static void parse_sha256(const char *text, uint8_t output[32])
{
	if (strlen(text) != 64)
		fail("track SHA-256 must contain 64 hexadecimal characters");
	for (uint32_t i = 0; i < 32; ++i) {
		output[i] = (uint8_t)(
			hex_nibble(text[i * 2]) << 4 |
			hex_nibble(text[i * 2 + 1]));
	}
}

static TMNFRaceInputs *read_inputs(const char *path, uint32_t *tick_count)
{
	FILE *file = fopen(path, "rb");
	if (file == NULL)
		fail("cannot open input schedule");
	if (fseek(file, 0, SEEK_END) != 0)
		fail("cannot seek input schedule");
	long size = ftell(file);
	if (size <= 0 || size % sizeof(TMNFRaceInputs) != 0)
		fail("input schedule has invalid framing");
	rewind(file);
	*tick_count = (uint32_t)(size / sizeof(TMNFRaceInputs));
	TMNFRaceInputs *inputs = malloc((size_t)*tick_count * sizeof(*inputs));
	if (inputs == NULL)
		fail("cannot allocate input schedule");
	if (fread(inputs, sizeof(*inputs), *tick_count, file) != *tick_count)
		fail("input schedule is truncated");
	fclose(file);
	return inputs;
}

static void write_position(FILE *trace, const CHmsStateDyna *dyna)
{
	float position[3] = { dyna->pos.x, dyna->pos.y, dyna->pos.z };
	if (fwrite(position, sizeof(position), 1, trace) != 1)
		fail("cannot write trace");
}

int main(int argc, char **argv)
{
	if (argc != 7) {
		fprintf(stderr,
			"usage: %s TRACK VEHICLE ROUTE INPUTS TRACK_SHA256 TRACE\n",
			argv[0]);
		return 2;
	}
	uint8_t sha256[32];
	parse_sha256(argv[5], sha256);
	uint32_t tick_count;
	TMNFRaceInputs *inputs = read_inputs(argv[4], &tick_count);
	TmnfTrack *track = TmnfTrack_Load(argv[1], sha256);
	TmnfWorld *owner = World_Create(track, argv[2]);
	TmnfRoute *route = TmnfRoute_Load(argv[3], sha256);
	TmnfPhysicsWorld *world = World_GetPhysicsWorld(owner);
	TmnfPhysicsCorpus *corpus = &world->corpora[0];
	FILE *trace = fopen(argv[6], "wb");
	if (trace == NULL)
		fail("cannot open trace for writing");

	TmnfRaceState race;
	world->route = route;
	TmnfRace_Reset(route, &race, corpus->collision_corpus->live_iso);

	uint32_t ticks = 0;
	for (uint32_t tick = 0; tick < tick_count; ++tick) {
		if (inputs[tick].respawn != 0) {
			/* 0x00472700 SmallRespawn before the tick's control mapping;
			 * without a checkpoint the game restarts the race, which no
			 * finished replay contains. */
			const GmIso4 *spawn = TmnfRace_RespawnLocation(&race);
			if (spawn == NULL)
				fail("respawn before any checkpoint restarts the race");
			World_Respawn(owner, spawn);
			printf("respawn tick=%u race_time_ms=%u\n",
				tick + 1, (tick + 1) * TICK_MS);
		}
		write_position(trace, World_GetPlayerState(owner));
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
			&inputs[tick], World_GetPlayerVehicle(owner));
		World_AdvanceTimer(owner, TICK_MS);
		CHmsZoneDynamic_PhysicsStep2(world, TICK_MS);
		ticks = tick + 1;

		TmnfRaceStepResult result = TmnfRace_Step(
			route, &race, world->trigger_contacts,
			corpus->collision_corpus->live_iso);
		if (result.checkpoint_accepted)
			printf("checkpoint tick=%u race_time_ms=%u index=%u\n",
				ticks, ticks * TICK_MS, result.checkpoint_index);
		if (result.checkpoint_repeated)
			printf("checkpoint_repeated tick=%u race_time_ms=%u index=%u\n",
				ticks, ticks * TICK_MS, result.checkpoint_index);
		if (result.lap_completed)
			printf("lap tick=%u race_time_ms=%u laps=%u\n",
				ticks, ticks * TICK_MS, race.completed_laps);
		if (result.finished) {
			printf("finish tick=%u race_time_ms=%u\n",
				ticks, result.race_time_ms);
			break;
		}
	}
	write_position(trace, World_GetPlayerState(owner));
	printf("ticks %u\n", ticks);

	if (fclose(trace) != 0)
		fail("cannot close trace");
	TmnfRoute_Unload(route);
	World_Destroy(owner);
	TmnfTrack_Unload(track);
	free(inputs);
	return 0;
}
