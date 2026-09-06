#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "route.h"

enum {
	RUN_RECORD_SIZE = 1668,
	RUN_POSITION_OFFSET = 56,
	RUN_TICKS = 1000,
	BENCH_REPETITIONS = 1000,
};

static const uint8_t A01_SHA256[32] = {
	0xf0, 0xa8, 0x70, 0x80, 0x9b, 0xe9, 0x9d, 0xa2,
	0xcb, 0x36, 0xad, 0x5d, 0xf4, 0x3a, 0x2c, 0xf6,
	0x3d, 0x8f, 0x74, 0xfe, 0x4a, 0xc3, 0x47, 0x0e,
	0xca, 0xc6, 0x8b, 0x9e, 0x97, 0x62, 0x5d, 0xc3,
};

static void require(int condition, const char *message)
{
	if (!condition) {
		fprintf(stderr, "route centerline: %s\n", message);
		exit(1);
	}
}

static int compare_float(const void *left, const void *right)
{
	float a = *(const float *)left;
	float b = *(const float *)right;
	return (a > b) - (a < b);
}

static GmVec3 *read_positions(const char *path)
{
	FILE *file = fopen(path, "rb");
	require(file != NULL, "cannot open run1.bin");
	GmVec3 *positions = malloc(sizeof(*positions) * RUN_TICKS);
	require(positions != NULL, "cannot allocate run positions");
	for (uint32_t tick = 0; tick < RUN_TICKS; ++tick) {
		require(
			fseek(
				file,
				(long)tick * RUN_RECORD_SIZE + RUN_POSITION_OFFSET,
				SEEK_SET) == 0,
			"cannot seek run position");
		require(
			fread(&positions[tick], sizeof(positions[tick]), 1, file) == 1,
			"cannot read run position");
	}
	require(fseek(file, 0, SEEK_END) == 0, "cannot seek run end");
	require(ftell(file) == (long)RUN_TICKS * RUN_RECORD_SIZE,
		"run1.bin has unexpected size");
	require(fclose(file) == 0, "cannot close run1.bin");
	return positions;
}

static TmnfRouteProjection project_brute(
	const TmnfRoute *route, const GmVec3 *position)
{
	TmnfRouteProjection result = {
		.segment_index = UINT32_MAX,
		.centerline_segment_index = UINT32_MAX,
	};
	float best_distance_sq = INFINITY;
	uint32_t count = TmnfRoute_GetReferencePointCount(route);
	const TmnfRouteReferencePoint *points =
		TmnfRoute_GetReferencePoints(route);
	for (uint32_t index = 0; index + 1 < count; ++index) {
		const TmnfRouteReferencePoint *a = &points[index];
		const TmnfRouteReferencePoint *b = &points[index + 1];
		float dx = b->position.x - a->position.x;
		float dy = b->position.y - a->position.y;
		float dz = b->position.z - a->position.z;
		float px = position->x - a->position.x;
		float py = position->y - a->position.y;
		float pz = position->z - a->position.z;
		float length_sq = dx * dx + dy * dy + dz * dz;
		float t = (px * dx + py * dy + pz * dz) / length_sq;
		if (t < 0.0f)
			t = 0.0f;
		else if (t > 1.0f)
			t = 1.0f;
		float ox = px - t * dx;
		float oy = py - t * dy;
		float oz = pz - t * dz;
		float distance_sq = ox * ox + oy * oy + oz * oz;
		if (distance_sq < best_distance_sq ||
			(distance_sq == best_distance_sq &&
			 index < result.centerline_segment_index)) {
			best_distance_sq = distance_sq;
			result.arc_length = a->arc_length +
				t * (b->arc_length - a->arc_length);
			result.half_width = a->half_width +
				t * (b->half_width - a->half_width);
			result.segment_index = a->leg_index;
			result.centerline_segment_index = index;
		}
	}
	result.lateral_offset = sqrtf(best_distance_sq);
	result.segments_tested = count - 1;
	return result;
}

static GmVec3 trigger_center(const TmnfRouteTrigger *trigger)
{
	const GmVec3 *center = &trigger->box.center;
	const GmIso4 *matrix = &trigger->transform;
	return (GmVec3){
		matrix->m[0] * center->x + matrix->m[1] * center->y +
			matrix->m[2] * center->z + matrix->t[0],
		matrix->m[3] * center->x + matrix->m[4] * center->y +
			matrix->m[5] * center->z + matrix->t[1],
		matrix->m[6] * center->x + matrix->m[7] * center->y +
			matrix->m[8] * center->z + matrix->t[2],
	};
}

static uint64_t nanoseconds(struct timespec start, struct timespec finish)
{
	int64_t seconds = finish.tv_sec - start.tv_sec;
	int64_t nanos = finish.tv_nsec - start.tv_nsec;
	return (uint64_t)(seconds * INT64_C(1000000000) + nanos);
}

int main(int argc, char **argv)
{
	const char *route_path = argc > 1
		? argv[1] : "oracle/routes/A01-Race.tmnfroute";
	const char *run_path = argc > 2
		? argv[2] : "oracle/results/run1.bin";
	TmnfRoute *route = TmnfRoute_Load(route_path, A01_SHA256);
	GmVec3 *positions = read_positions(run_path);
	uint32_t point_count = TmnfRoute_GetReferencePointCount(route);
	require(route->header->version == TMNF_ROUTE_VERSION,
		"route version is not the current one");
	require(point_count == 1109, "unexpected centerline point count");
	require(route->projection_bvh_count == 2 * (point_count - 1) - 1,
		"BVH node count");
	require(route->projection_bvh_height <= 16, "BVH is too deep");

	float offsets[RUN_TICKS];
	float previous_arc = -INFINITY;
	float peak_arc = -INFINITY;
	uint32_t decreasing_steps = 0;
	float maximum_decrease = 0.0f;
	float maximum_drawdown = 0.0f;
	uint64_t tested_total = 0;
	uint32_t tested_maximum = 0;
	for (uint32_t tick = 0; tick < RUN_TICKS; ++tick) {
		TmnfRouteProjection indexed =
			TmnfRoute_Project(route, &positions[tick]);
		TmnfRouteProjection brute =
			project_brute(route, &positions[tick]);
		require(indexed.segment_index == brute.segment_index,
			"BVH and brute-force route leg differ");
		require(
			indexed.centerline_segment_index ==
				brute.centerline_segment_index,
			"BVH and brute-force segment differ");
		require(indexed.arc_length == brute.arc_length,
			"BVH and brute-force arc length differ");
		require(indexed.lateral_offset == brute.lateral_offset,
			"BVH and brute-force offset differ");
		require(indexed.half_width == brute.half_width,
			"BVH and brute-force width differ");
		require(indexed.lateral_offset <= indexed.half_width,
			"run position projects outside local track width");
		offsets[tick] = indexed.lateral_offset;
		if (indexed.arc_length < previous_arc) {
			float decrease = previous_arc - indexed.arc_length;
			decreasing_steps++;
			if (decrease > maximum_decrease)
				maximum_decrease = decrease;
		}
		if (indexed.arc_length > peak_arc)
			peak_arc = indexed.arc_length;
		else if (peak_arc - indexed.arc_length > maximum_drawdown)
			maximum_drawdown = peak_arc - indexed.arc_length;
		previous_arc = indexed.arc_length;
		tested_total += indexed.segments_tested;
		if (indexed.segments_tested > tested_maximum)
			tested_maximum = indexed.segments_tested;
	}
	require(decreasing_steps != 0,
		"run1 no longer demonstrates its recorded reverse motion");
	qsort(offsets, RUN_TICKS, sizeof(offsets[0]), compare_float);
	float median = (offsets[499] + offsets[500]) * 0.5f;
	float p95 = offsets[949] * 0.95f + offsets[950] * 0.05f;

	GmVec3 checkpoint0 = trigger_center(TmnfRoute_GetCheckpoint(route, 0));
	GmVec3 checkpoint1 = trigger_center(TmnfRoute_GetCheckpoint(route, 1));
	GmVec3 finish = trigger_center(TmnfRoute_GetFinish(route));
	float checkpoint0_arc =
		TmnfRoute_Project(route, &checkpoint0).arc_length;
	float checkpoint1_arc =
		TmnfRoute_Project(route, &checkpoint1).arc_length;
	float finish_arc = TmnfRoute_Project(route, &finish).arc_length;
	require(
		checkpoint0_arc < checkpoint1_arc && checkpoint1_arc < finish_arc,
		"trigger projections are out of order");

	volatile float benchmark_sum = 0.0f;
	struct timespec start;
	struct timespec finish_time;
	require(timespec_get(&start, TIME_UTC) == TIME_UTC, "benchmark start time");
	for (uint32_t repetition = 0;
		repetition < BENCH_REPETITIONS; ++repetition) {
		for (uint32_t tick = 0; tick < RUN_TICKS; ++tick) {
			benchmark_sum +=
				TmnfRoute_Project(route, &positions[tick]).arc_length;
		}
	}
	require(
		timespec_get(&finish_time, TIME_UTC) == TIME_UTC,
		"benchmark finish time");
	uint64_t elapsed = nanoseconds(start, finish_time);
	require(isfinite(benchmark_sum), "benchmark result is non-finite");

	printf(
		"centerline: points=%u length=%.3f bvh_nodes=%u bvh_height=%u\n",
		point_count, TmnfRoute_GetReferenceLength(route),
		route->projection_bvh_count, route->projection_bvh_height);
	printf(
		"run1: median=%.3f p95=%.3f max=%.3f "
		"decreasing_steps=%u max_decrease=%.6f max_drawdown=%.6f\n",
		median, p95, offsets[RUN_TICKS - 1],
		decreasing_steps, maximum_decrease, maximum_drawdown);
	printf(
		"projection: average_segments=%.3f max_segments=%u ns_per_call=%.3f\n",
		(double)tested_total / RUN_TICKS, tested_maximum,
		(double)elapsed / (BENCH_REPETITIONS * RUN_TICKS));
	printf(
		"anchors: checkpoint_0=%.3f checkpoint_1=%.3f finish=%.3f\n",
		checkpoint0_arc, checkpoint1_arc, finish_arc);

	free(positions);
	TmnfRoute_Unload(route);
	return 0;
}
