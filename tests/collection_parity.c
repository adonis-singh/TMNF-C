/* Optional local-fixture integration check. No game assets are distributed.
 * Inputs are native TMNFRaceInputs records, one per 10 ms simulation tick. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tmnf_collection_ffi.h"

static void require(int condition, const char *operation)
{
	if (condition) return;
	fprintf(stderr, "%s: %s\n", operation, tmnf_collection_error());
	exit(1);
}

static void compare(const TmnfCollectionFrame *cpu, const TmnfCollectionFrame *gpu)
{
	const unsigned char *a = (const unsigned char *)cpu;
	const unsigned char *b = (const unsigned char *)gpu;
	for (size_t i = 0; i < sizeof(*cpu); ++i) {
		if (a[i] == b[i]) continue;
		fprintf(stderr, "collection differs at %u ms, frame byte %zu: CPU=%02x CUDA=%02x\n",
			cpu->time_ms, i, a[i], b[i]);
		exit(1);
	}
}

int main(int argc, char **argv)
{
	if (argc != 10 && argc != 11) {
		fprintf(stderr, "usage: %s TRACK VEHICLE ROUTE TRIGGERS SHA256 INPUTS LAPS SEED REPLAY_MODE [RESET_POSITION]\n",
			argv[0]);
		return 2;
	}
	uint8_t digest[32];
	if (strlen(argv[5]) != 64) return 2;
	for (size_t i = 0; i < sizeof(digest); ++i) {
		unsigned int value;
		if (sscanf(argv[5] + 2 * i, "%2x", &value) != 1) return 2;
		digest[i] = (uint8_t)value;
	}
	FILE *inputs = fopen(argv[6], "rb");
	if (inputs == NULL) return 2;
	TmnfCollection *envs[2];
	uint32_t laps = (uint32_t)strtoul(argv[7], NULL, 10);
	uint32_t seed = (uint32_t)strtoul(argv[8], NULL, 10);
	int replay_mode = atoi(argv[9]);
	for (int cuda = 0; cuda < 2; ++cuda) {
		envs[cuda] = tmnf_collection_create_geometry(argv[1], argv[2], argv[3],
			strcmp(argv[4], "-") == 0 ? NULL : argv[4], digest, cuda, 1, 1, seed);
		require(envs[cuda] != NULL, "create");
		require(tmnf_collection_set_replay_mode(envs[cuda], replay_mode), "replay mode");
		require(tmnf_collection_set_lap_count(envs[cuda], laps), "recorded laps");
	}
	TmnfCollectionFrame frames[2];
	if (argc == 11 && atoi(argv[10])) {
		require(tmnf_collection_observe(envs[0], &frames[0]), "initial position");
		const float position[] = {
			frames[0].dyna.pos.x, frames[0].dyna.pos.y, frames[0].dyna.pos.z,
		};
		for (int cuda = 0; cuda < 2; ++cuda)
			require(tmnf_collection_reset_position(envs[cuda], seed, position), "position reset");
	}
	TMNFRaceInputs input;
	uint32_t ticks = 0;
	while (fread(&input, sizeof(input), 1, inputs) == 1) {
		for (int cuda = 0; cuda < 2; ++cuda)
			require(tmnf_collection_tick(envs[cuda], &input, &frames[cuda]), "tick");
		compare(&frames[0], &frames[1]);
		++ticks;
		for (int cuda = 0; cuda < 2; ++cuda)
			require(tmnf_collection_observe(envs[cuda], &frames[cuda]), "observe");
		compare(&frames[0], &frames[1]);
		if (frames[0].finished) break;
	}
	if (ferror(inputs) || ticks == 0) return 2;
	const uint32_t *times[2];
	uint32_t count[2];
	for (int cuda = 0; cuda < 2; ++cuda)
		count[cuda] = tmnf_collection_checkpoint_times(envs[cuda], &times[cuda]);
	require(count[0] == count[1], "checkpoint counts");
	require(!count[0] || memcmp(times[0], times[1], count[0] * sizeof(*times[0])) == 0,
		"checkpoint times");
	printf("CPU/CUDA collection exact for %u ticks, %u checkpoints, finish %u ms\n",
		ticks, count[0], frames[0].finish_time_ms);
	for (int cuda = 0; cuda < 2; ++cuda) tmnf_collection_destroy(envs[cuda]);
	fclose(inputs);
	return 0;
}
