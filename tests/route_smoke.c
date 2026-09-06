#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "route.h"

enum {
	RUN_RECORD_SIZE = 1668,
	RUN_POSITION_OFFSET = 56,
};

static const uint8_t A01_SHA256[32] = {
	0xf0, 0xa8, 0x70, 0x80, 0x9b, 0xe9, 0x9d, 0xa2,
	0xcb, 0x36, 0xad, 0x5d, 0xf4, 0x3a, 0x2c, 0xf6,
	0x3d, 0x8f, 0x74, 0xfe, 0x4a, 0xc3, 0x47, 0x0e,
	0xca, 0xc6, 0x8b, 0x9e, 0x97, 0x62, 0x5d, 0xc3,
};

static const uint8_t ROUTE_PAYLOAD_SHA256[32] = {
	0xd2, 0x1f, 0xa7, 0xc0, 0xdc, 0xfe, 0x7f, 0x31,
	0x69, 0xec, 0x42, 0xda, 0xe8, 0x81, 0x67, 0x1b,
	0x3c, 0xaa, 0xe2, 0x5c, 0x05, 0x08, 0xa7, 0x91,
	0xd7, 0x20, 0x65, 0x7a, 0x6c, 0x39, 0x8e, 0xe1,
};

static void require(int condition, const char *message) {
	if (!condition) {
		fprintf(stderr, "route smoke: %s\n", message);
		exit(1);
	}
}

static GmVec3 read_run_position(FILE *run, uint32_t tick_index) {
	GmVec3 position;
	long offset = (long)tick_index * RUN_RECORD_SIZE + RUN_POSITION_OFFSET;
	require(fseek(run, offset, SEEK_SET) == 0, "cannot seek run1 position");
	require(
		fread(&position, sizeof(position), 1, run) == 1,
		"cannot read run1 position");
	return position;
}

int main(int argc, char **argv) {
	const char *route_path = argc > 1
		? argv[1] : "oracle/routes/A01-Race.tmnfroute";
	const char *run_path = argc > 2
		? argv[2] : "oracle/results/run1.bin";
	const uint32_t sample_ticks[] = {0, 50, 100, 150, 200};
	TmnfRoute *route = TmnfRoute_Load(route_path, A01_SHA256);

	require(
		memcmp(
			route->header->payload_sha256,
			ROUTE_PAYLOAD_SHA256,
			sizeof(ROUTE_PAYLOAD_SHA256)) == 0,
		"unexpected canonical payload hash");
	require(TmnfRoute_GetCheckpointCount(route) == 2, "checkpoint count");
	require(TmnfRoute_GetReferencePointCount(route) == 4, "reference count");
	require(route->metadata->lap_count == 1, "lap count");
	require(route->metadata->finish_count == 1, "finish count");
	require(route->metadata->total_race_checkpoints == 3, "total checkpoint count");
	require(route->metadata->race_checkpoint_limit == 3, "checkpoint limit");
	require(route->metadata->flags == 0, "single-lap flags");
	require(TmnfRoute_GetStart(route)->block_index == 43, "start block index");
	require(
		TmnfRoute_GetCheckpoint(route, 0)->block_index == 0 &&
		TmnfRoute_GetCheckpoint(route, 1)->block_index == 134,
		"checkpoint block indices");
	require(TmnfRoute_GetFinish(route)->block_index == 105, "finish block index");
	require(
		fabsf(TmnfRoute_GetReferenceLength(route) - 1871.9808349609375f)
			< 0.001f,
		"reference length");

	FILE *run = fopen(run_path, "rb");
	require(run != NULL, "cannot open run1.bin");
	float previous_arc = -1.0f;
	for (uint32_t i = 0; i < sizeof(sample_ticks) / sizeof(sample_ticks[0]); ++i) {
		GmVec3 position = read_run_position(run, sample_ticks[i]);
		TmnfRouteProjection projection = TmnfRoute_Project(route, &position);
		require(
			projection.arc_length > previous_arc,
			"projection did not advance on run1 prefix");
		require(projection.segment_index == 0, "prefix projected to wrong segment");
		require(isfinite(projection.lateral_offset), "non-finite lateral offset");
		printf(
			"tick=%u arc=%.6f lateral=%.6f segment=%u\n",
			sample_ticks[i] + 1, projection.arc_length,
			projection.lateral_offset, projection.segment_index);
		previous_arc = projection.arc_length;
	}
	require(fclose(run) == 0, "cannot close run1.bin");
	TmnfRoute_Unload(route);
	return 0;
}
