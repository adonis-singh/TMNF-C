#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "world.h"

static void fail(const char *message)
{
	fprintf(stderr, "smoke_track_env: %s\n", message);
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

static TMNFRaceInputs *read_inputs(
	const char *path, uint32_t *tick_count)
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
	if (fclose(file) != 0)
		fail("cannot close input schedule");
	return inputs;
}

int main(int argc, char **argv)
{
	if (argc != 6) {
		fprintf(
			stderr,
			"usage: %s TRACK VEHICLE ROUTE INPUTS TRACK_SHA256\n",
			argv[0]);
		return 2;
	}
	uint8_t track_sha256[32];
	parse_sha256(argv[5], track_sha256);
	uint32_t tick_count;
	TMNFRaceInputs *inputs = read_inputs(argv[4], &tick_count);

	TmnfTrack *track = TmnfTrack_Load(argv[1], track_sha256);
	TmnfWorld *owner = World_Create(track, argv[2]);
	TmnfRoute *route = TmnfRoute_Load(argv[3], track_sha256);
	TmnfRouteProjection projection = TmnfRoute_Project(
		route, &World_GetPlayerState(owner)->pos);
	float best_progress = projection.arc_length;
	float previous_progress = projection.arc_length;
	float maximum_lateral_offset = projection.lateral_offset;
	uint32_t decreasing_steps = 0;
	for (uint32_t tick = 0; tick < tick_count; ++tick) {
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
			&inputs[tick], World_GetPlayerVehicle(owner));
		World_AdvanceTimer(owner, 10);
		CHmsZoneDynamic_PhysicsStep2(
			World_GetPhysicsWorld(owner), 10);
		projection = TmnfRoute_Project(
			route, &World_GetPlayerState(owner)->pos);
		if (projection.arc_length + 1.0e-5f < previous_progress)
			++decreasing_steps;
		if (projection.arc_length > best_progress)
			best_progress = projection.arc_length;
		if (projection.lateral_offset > maximum_lateral_offset)
			maximum_lateral_offset = projection.lateral_offset;
		previous_progress = projection.arc_length;
	}

	float route_length = TmnfRoute_GetReferenceLength(route);
	printf("ticks: %u\n", tick_count);
	printf("route length: %.3f m\n", route_length);
	printf("best progress: %.3f m\n", best_progress);
	printf("final progress: %.3f m\n", projection.arc_length);
	printf("final remaining: %.3f m\n",
		route_length - projection.arc_length);
	printf("final lateral offset: %.3f m\n", projection.lateral_offset);
	printf("maximum lateral offset: %.3f m\n", maximum_lateral_offset);
	printf("decreasing progress steps: %u\n", decreasing_steps);
	printf("checkpoints: 0/%u\n", route->metadata->checkpoint_count);

	TmnfRoute_Unload(route);
	World_Destroy(owner);
	TmnfTrack_Unload(track);
	free(inputs);
	return 0;
}
