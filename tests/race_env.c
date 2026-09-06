#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "race.h"
#include "vec_env.h"
#include "world.h"

static const uint8_t A01_SHA256[32] = {
	0xf0, 0xa8, 0x70, 0x80, 0x9b, 0xe9, 0x9d, 0xa2,
	0xcb, 0x36, 0xad, 0x5d, 0xf4, 0x3a, 0x2c, 0xf6,
	0x3d, 0x8f, 0x74, 0xfe, 0x4a, 0xc3, 0x47, 0x0e,
	0xca, 0xc6, 0x8b, 0x9e, 0x97, 0x62, 0x5d, 0xc3,
};

typedef struct {
	TmnfRoute route;
	TmnfRouteMetadata metadata;
	TmnfRouteStart start;
	TmnfRouteTrigger checkpoints[2];
	TmnfRouteTrigger finish;
	TmnfRouteReferencePoint centerline[128];
} TestRoute;

static TmnfRaceStepResult race_step_at_position(
	const TmnfRoute *route, TmnfRaceState *state, GmVec3 position);
static GmIso4 identity_at(float x);

/* A one-ellipsoid car collision tree: root AABB plus one child whose
 * ellipsoid radii are 0.5 m. The game's checkpoint contact runs the same
 * ellipsoid-versus-mesh narrowphase as track collisions. */
typedef struct {
	CPlugTree root;
	CPlugTree child;
	CPlugTree *children[1];
	CPlugSurface surface;
	GmSurfEllipsoid ellipsoid;
	uint8_t material_ids[1];
} TestCarTree;

static void make_car_tree(TestCarTree *car)
{
	memset(car, 0, sizeof(*car));
	car->ellipsoid.base.type = GM_SURF_ELLIPSOID;
	car->ellipsoid.radii = (GmVec3){0.5f, 0.5f, 0.5f};
	car->surface.geom = &car->ellipsoid.base;
	car->surface.material_ids = car->material_ids;
	car->surface.material_count = 1;
	car->child.flags = 0x84;
	car->child.box = (GmBoxAligned){
		.center = {0.0f, 0.0f, 0.0f},
		.half_extent = {0.5f, 0.5f, 0.5f},
	};
	car->child.local_iso = identity_at(0.0f);
	car->child.surface = &car->surface;
	car->children[0] = &car->child;
	car->root.flags = 0x84;
	car->root.box = car->child.box;
	car->root.local_iso = identity_at(0.0f);
	car->root.child_count = 1;
	car->root.children = car->children;
}

static void require(int condition, const char *message)
{
	if (!condition) {
		fprintf(stderr, "race env: %s\n", message);
		exit(1);
	}
}

static GmIso4 identity_at(float x)
{
	GmIso4 transform = {0};
	transform.m[0] = 1.0f;
	transform.m[4] = 1.0f;
	transform.m[8] = 1.0f;
	transform.t[0] = x;
	return transform;
}

static GmIso4 identity_at_position(GmVec3 position)
{
	GmIso4 transform = identity_at(position.x);
	transform.t[1] = position.y;
	transform.t[2] = position.z;
	return transform;
}

static TmnfRouteTrigger trigger_at(float x, uint32_t index)
{
	TmnfRouteTrigger trigger = {
		.race_index = index,
		.waypoint_type = TMNF_ROUTE_WAYPOINT_CHECKPOINT,
		.tree_flags = 0x80,
		.box = {
			.center = {0.0f, 0.0f, 0.0f},
			.half_extent = {1.0f, 1.0f, 1.0f},
		},
		.transform = identity_at(x),
	};
	return trigger;
}

static void make_route(
	TestRoute *test, uint32_t checkpoint_count, uint32_t laps)
{
	require(checkpoint_count <= 2,
		"unsupported test checkpoint count");
	memset(test, 0, sizeof(*test));
	test->metadata.lap_count = laps;
	test->metadata.checkpoint_count = checkpoint_count;
	test->metadata.finish_count = 1;
	test->metadata.reference_count = checkpoint_count + 2;
	test->metadata.centerline_count = checkpoint_count + 2;
	test->metadata.total_race_checkpoints =
		(checkpoint_count + 1) * laps;
	test->metadata.race_checkpoint_limit =
		test->metadata.total_race_checkpoints;
	test->metadata.flags = laps > 1 ? TMNF_ROUTE_MULTILAP : 0;
	test->start.transform = identity_at(-10.0f);
	test->start.waypoint_type = TMNF_ROUTE_WAYPOINT_START;
	if (checkpoint_count >= 1)
		test->checkpoints[0] = trigger_at(0.0f, 0);
	if (checkpoint_count == 2)
		test->checkpoints[1] = trigger_at(10.0f, 1);
	test->finish = trigger_at(
		checkpoint_count == 2 ? 20.0f : 10.0f, 0);
	test->finish.waypoint_type = TMNF_ROUTE_WAYPOINT_FINISH;

	test->centerline[0] = (TmnfRouteReferencePoint){
		.position = {-10.0f, 0.0f, 0.0f},
		.arc_length = 0.0f,
		.half_width = 5.0f,
	};
	if (checkpoint_count == 0) {
		test->centerline[1] = (TmnfRouteReferencePoint){
			.position = {10.0f, 0.0f, 0.0f},
			.arc_length = 20.0f,
			.half_width = 5.0f,
		};
		test->route.metadata = &test->metadata;
		test->route.start = &test->start;
		test->route.checkpoints = test->checkpoints;
		test->route.finish = &test->finish;
		test->route.centerline = test->centerline;
		return;
	}
	test->centerline[1] = (TmnfRouteReferencePoint){
		.position = {0.0f, 0.0f, 0.0f},
		.arc_length = 10.0f,
		.half_width = 5.0f,
	};
	if (checkpoint_count == 2) {
		test->centerline[2] = (TmnfRouteReferencePoint){
			.position = {10.0f, 0.0f, 0.0f},
			.arc_length = 20.0f,
			.half_width = 5.0f,
		};
		test->centerline[3] = (TmnfRouteReferencePoint){
			.position = {20.0f, 0.0f, 0.0f},
			.arc_length = 30.0f,
			.half_width = 5.0f,
		};
	} else {
		test->centerline[2] = (TmnfRouteReferencePoint){
			.position = {10.0f, 0.0f, 0.0f},
			.arc_length = 20.0f,
			.half_width = 5.0f,
		};
	}
	test->route.metadata = &test->metadata;
	test->route.start = &test->start;
	test->route.checkpoints = test->checkpoints;
	test->route.finish = &test->finish;
	test->route.centerline = test->centerline;
}

static TmnfCenterlineSample centerline_sample_brute(
	const TmnfRoute *route, float arc_length)
{
	const TmnfRouteReferencePoint *points =
		TmnfRoute_GetReferencePoints(route);
	uint32_t count = TmnfRoute_GetReferencePointCount(route);
	if (arc_length >= points[count - 1].arc_length) {
		return (TmnfCenterlineSample){
			.position = points[count - 1].position,
			.half_width = points[count - 1].half_width,
		};
	}
	uint32_t segment = 0;
	while (points[segment + 1].arc_length <= arc_length)
		segment++;
	const TmnfRouteReferencePoint *a = &points[segment];
	const TmnfRouteReferencePoint *b = &points[segment + 1];
	float t = (arc_length - a->arc_length) /
		(b->arc_length - a->arc_length);
	return (TmnfCenterlineSample){
		.position = {
			a->position.x + t * (b->position.x - a->position.x),
			a->position.y + t * (b->position.y - a->position.y),
			a->position.z + t * (b->position.z - a->position.z),
		},
		.half_width = a->half_width +
			t * (b->half_width - a->half_width),
	};
}

static void test_dense_lookahead(
	const TmnfRoute *route, const TmnfObservation *observation)
{
	static const float expected_distances[
		TMNF_OBSERVATION_LOOKAHEAD_COUNT] = {
		5.0f, 10.0f, 20.0f, 35.0f, 55.0f, 80.0f, 110.0f, 150.0f,
	};
	require(memcmp(
			TMNF_OBSERVATION_LOOKAHEAD_METERS,
			expected_distances, sizeof(expected_distances)) == 0,
		"lookahead distance layout changed");
	TmnfRouteProjection start = TmnfRoute_Project(
		route, &route->start->initial_state.pos);
	require(start.arc_length == 0.0f,
		"known lookahead position is not the route start");
	require(observation->track_half_width == start.half_width,
		"observation omitted the current local half-width");

	GmIso4 inverse_car;
	GmIso4_SetInverse(&inverse_car, &route->start->transform);
	for (uint32_t i = 0;
		i < TMNF_OBSERVATION_LOOKAHEAD_COUNT; ++i) {
		TmnfCenterlineSample expected = centerline_sample_brute(
			route, expected_distances[i]);
		GmVec3 local;
		GmVec3_SetMult_Iso4(
			&local, &expected.position, &inverse_car);
		require(memcmp(
				&observation->centerline_lookahead[i].position,
				&local, sizeof(local)) == 0,
			"dense lookahead local position is wrong");
		require(
			observation->centerline_lookahead[i].half_width ==
				expected.half_width,
			"dense lookahead half-width is wrong");
	}
}

static void test_local_width_and_jump_gaps(const TmnfRoute *route)
{
	const TmnfRouteReferencePoint *points =
		TmnfRoute_GetReferencePoints(route);
	uint32_t count = TmnfRoute_GetReferencePointCount(route);
	require(count == 1109, "unexpected centerline test fixture");
	uint32_t narrowest = 0;
	for (uint32_t i = 1; i < count; ++i) {
		if (points[i].half_width < points[narrowest].half_width)
			narrowest = i;
	}

	/* Off-track is a ground-plane rule: every wheel in contact rests on
	 * grass. Width against the raster is an observation only; every one of
	 * the six record replays leaves the raster width while on asphalt. */
	TmnfRaceState state = {0};
	require(!TmnfRace_UpdateOffTrack(&state, 3, 4, 0) &&
		state.off_track_ticks == 0,
		"four asphalt wheels counted as off-track");
	require(!TmnfRace_UpdateOffTrack(&state, 3, 4, 4) &&
		state.off_track_ticks == 1,
		"four grass wheels did not start the off-track count");
	require(!TmnfRace_UpdateOffTrack(&state, 3, 0, 0) &&
		state.off_track_ticks == 1,
		"airborne tick did not hold the off-track count");
	require(!TmnfRace_UpdateOffTrack(&state, 3, 1, 1) &&
		state.off_track_ticks == 2,
		"one grass wheel alone did not advance the count");
	/* Wall ride or road edge: one wheel on a track surface, three on grass. */
	require(!TmnfRace_UpdateOffTrack(&state, 3, 4, 3) &&
		state.off_track_ticks == 0,
		"one track-surface wheel did not clear the count");
	(void)TmnfRace_UpdateOffTrack(&state, 3, 2, 2);
	(void)TmnfRace_UpdateOffTrack(&state, 3, 4, 4);
	require(TmnfRace_UpdateOffTrack(&state, 3, 4, 4) &&
		state.off_track_ticks == 3,
		"grass contact did not fire at the grace count");
	require(TmnfRace_IsGroundPlaneMaterial(2) &&
		TmnfRace_IsGroundPlaneMaterial(20) &&
		!TmnfRace_IsGroundPlaneMaterial(16) &&
		!TmnfRace_IsGroundPlaneMaterial(0) &&
		!TmnfRace_IsGroundPlaneMaterial(4),
		"ground-plane material classification is wrong");

	static const struct {
		uint32_t first;
		uint32_t last;
		uint32_t before;
		uint32_t after;
	} gaps[2] = {
		{858, 863, 857, 864},
		{869, 875, 868, 876},
	};
	uint32_t airborne_points = 0;
	for (uint32_t gap = 0; gap < 2; ++gap) {
		float borrowed_width = fminf(
			points[gaps[gap].before].half_width,
			points[gaps[gap].after].half_width);
		for (uint32_t i = gaps[gap].first;
			i <= gaps[gap].last; ++i) {
			require(points[i].half_width == borrowed_width,
				"jump gap width is not borrowed from its endpoints");
			airborne_points++;
		}
	}
	require(airborne_points == 13,
		"jump-gap test did not cover all airborne points");
}

static void test_a01_forced_dense_branch(const TmnfRoute *route)
{
	const TmnfRouteReferencePoint *points =
		TmnfRoute_GetReferencePoints(route);
	require(points[592].leg_index == 0 &&
		points[867].leg_index == 2,
		"A01 forced-branch fixture changed");

	TmnfRaceState state = {
		.projection_segment = points[592].leg_index,
		.centerline_segment = 592,
		.arc_length = points[592].arc_length,
		.lateral_offset = 0.0f,
		.half_width = points[592].half_width,
		.unwrapped_progress = points[592].arc_length,
		.best_progress = points[592].arc_length,
		.previous_progress = points[592].arc_length,
		.previous_car_transform =
			identity_at_position(points[592].position),
	};
	TmnfRouteProjection expected =
		TmnfRoute_Project(route, &points[867].position);
	(void)race_step_at_position(
		route, &state, points[867].position);
	require(
		state.centerline_segment ==
			expected.centerline_segment_index &&
		state.projection_segment == expected.segment_index &&
		state.arc_length == expected.arc_length &&
		state.lateral_offset == expected.lateral_offset &&
		state.half_width == expected.half_width,
		"A01 forced branch mixed projection sources");
	require(
		state.centerline_segment == 866 &&
		state.arc_length == 1727.752808f &&
		state.lateral_offset == 0.0f &&
		state.half_width == 8.25f,
		"A01 forced branch no longer matches its reproduction");
}

static void test_trigger_boundary(void)
{
	TmnfRouteTrigger trigger = trigger_at(100.0f, 0);
	TestCarTree car;
	make_car_tree(&car);
	GmIso4 iso = identity_at(101.5f);
	require(TmnfRace_TriggerContact(&trigger, &car.root, &iso),
		"ellipsoid touching the trigger face was rejected");
	iso.t[0] = nextafterf(101.5f, INFINITY);
	require(!TmnfRace_TriggerContact(&trigger, &car.root, &iso),
		"ellipsoid beyond the trigger face was accepted");
	/* Root AABBs overlap after a 45 degree yaw although the ellipsoid does
	 * not reach the face: the game's narrowphase decides, not the AABB. */
	GmIso4 yawed = identity_at(101.6f);
	float c = 0x1.6a09e6p-1f;
	yawed.m[0] = c;
	yawed.m[2] = -c;
	yawed.m[6] = c;
	yawed.m[8] = c;
	GmBoxAligned car_box;
	GmBoxAligned trigger_box;
	GmBoxAligned_SetMult(&car_box, &car.root.box, &yawed);
	GmBoxAligned_SetMult(&trigger_box, &trigger.box, &trigger.transform);
	require(GmBoxAligned_TestInter(&car_box, &trigger_box),
		"yawed root boxes should overlap in this fixture");
	require(!TmnfRace_TriggerContact(&trigger, &car.root, &yawed),
		"root AABB overlap alone registered a trigger contact");
}

/* One tick whose only detection pass sees the car at `position`: the mask
 * the physics step would accumulate, then the race step on it. */
static TmnfRaceStepResult race_step_at_position(
	const TmnfRoute *route, TmnfRaceState *state, GmVec3 position)
{
	TestCarTree car;
	make_car_tree(&car);
	GmIso4 iso = identity_at_position(position);
	uint64_t contacts = TmnfRace_TriggerContactMask(route, &car.root, &iso);
	return TmnfRace_Step(route, state, contacts, &iso);
}

static void race_reset_at(
	const TmnfRoute *route, TmnfRaceState *state, const GmIso4 *iso)
{
	TmnfRace_Reset(route, state, iso);
}

static TmnfRaceStepResult race_step_at(
	TestRoute *route, TmnfRaceState *state, float x)
{
	return race_step_at_position(
		&route->route, state, (GmVec3){x, 0.0f, 0.0f});
}

static void set_centerline_point(
	TestRoute *route, uint32_t index, GmVec3 position,
	float half_width, uint32_t leg_index)
{
	float arc_length = 0.0f;
	if (index != 0) {
		GmVec3 previous = route->centerline[index - 1].position;
		float dx = position.x - previous.x;
		float dy = position.y - previous.y;
		float dz = position.z - previous.z;
		arc_length = route->centerline[index - 1].arc_length +
			sqrtf(dx * dx + dy * dy + dz * dz);
	}
	route->centerline[index] = (TmnfRouteReferencePoint){
		.position = position,
		.arc_length = arc_length,
		.half_width = half_width,
		.leg_index = leg_index,
	};
}

static void test_same_leg_self_intersection(void)
{
	TestRoute route;
	make_route(&route, 1, 1);
	route.checkpoints[0] = trigger_at(1000.0f, 0);
	route.finish = trigger_at(1100.0f, 0);
	route.finish.waypoint_type = TMNF_ROUTE_WAYPOINT_FINISH;
	route.metadata.centerline_count = 73;
	for (uint32_t i = 0; i <= 40; ++i) {
		set_centerline_point(
			&route, i,
			(GmVec3){(float)i - 10.0f, 1.0f, 0.0f},
			5.0f, 0);
	}
	set_centerline_point(
		&route, 41, (GmVec3){30.0f, 30.0f, 0.0f}, 5.0f, 0);
	for (uint32_t i = 42; i <= 71; ++i) {
		set_centerline_point(
			&route, i,
			(GmVec3){(float)(71 - i), 30.0f, 0.0f},
			50.0f, 0);
	}
	set_centerline_point(
		&route, 72, (GmVec3){0.0f, 0.0f, 0.0f}, 50.0f, 1);

	GmVec3 previous_position = {0.0f, 1.0f, 0.0f};
	GmIso4 previous = identity_at_position(previous_position);
	TmnfRaceState state;
	race_reset_at(&route.route, &state, &previous);
	require(state.centerline_segment == 9,
		"self-intersection setup cursor is wrong");

	GmVec3 crossing = {0.0f, 0.0f, 0.0f};
	TmnfRouteProjection global =
		TmnfRoute_Project(&route.route, &crossing);
	require(global.centerline_segment_index == 71,
		"naive projection did not jump at self-intersection");
	(void)race_step_at_position(&route.route, &state, crossing);
	require(
		state.centerline_segment == 9 &&
		state.arc_length == route.centerline[10].arc_length &&
		state.lateral_offset == 1.0f &&
		state.half_width == 5.0f &&
		state.projection_segment == 0,
		"dense window left the current self-intersection branch");
}

static void test_teleport_dense_reacquisition(void)
{
	TestRoute route;
	make_route(&route, 1, 1);
	route.checkpoints[0] = trigger_at(1000.0f, 0);
	route.finish = trigger_at(1100.0f, 0);
	route.finish.waypoint_type = TMNF_ROUTE_WAYPOINT_FINISH;
	route.metadata.centerline_count = 100;
	for (uint32_t i = 0; i < route.metadata.centerline_count; ++i) {
		set_centerline_point(
			&route, i, (GmVec3){2.0f * (float)i, 0.0f, 0.0f},
			5.0f + 0.25f * (float)i,
			i + 1 == route.metadata.centerline_count ? 1 : 0);
	}

	GmIso4 previous =
		identity_at_position(route.centerline[5].position);
	TmnfRaceState state;
	race_reset_at(&route.route, &state, &previous);
	GmVec3 restored_position = {160.0f, 3.0f, 0.0f};
	(void)race_step_at_position(
		&route.route, &state, restored_position);
	require(
		state.centerline_segment == 79 &&
		state.arc_length == 160.0f &&
		state.lateral_offset == 3.0f &&
		state.half_width == route.centerline[80].half_width &&
		state.projection_segment ==
			route.centerline[79].leg_index,
		"teleport did not reacquire one coherent dense segment");
}

/* Progress is credited only inside the route corridor: horizontal offset
 * within max(3 half-widths, 28 m) of the projected centerline point and
 * height within [-16, 44] m of it. Outside, unwrapped_progress freezes (no
 * advance, no reversal) while the projection itself keeps following the car,
 * and it catches up on re-entry. */
static void test_corridor_freezes_progress(void)
{
	TestRoute route;
	make_route(&route, 1, 1);
	route.checkpoints[0] = trigger_at(1000.0f, 0);
	route.finish = trigger_at(1100.0f, 0);
	route.finish.waypoint_type = TMNF_ROUTE_WAYPOINT_FINISH;
	route.metadata.centerline_count = 100;
	for (uint32_t i = 0; i < route.metadata.centerline_count; ++i) {
		set_centerline_point(
			&route, i, (GmVec3){2.0f * (float)i, 0.0f, 0.0f}, 5.0f,
			i + 1 == route.metadata.centerline_count ? 1 : 0);
	}
	float allowed = TMNF_RACE_CORRIDOR_WIDTH_FACTOR * 5.0f;
	if (allowed < TMNF_RACE_CORRIDOR_WIDTH_FLOOR_METERS)
		allowed = TMNF_RACE_CORRIDOR_WIDTH_FLOOR_METERS;
	require(allowed == 28.0f,
		"corridor width fixture: floor must bind on a 5 m half-width");

	GmIso4 start = identity_at_position((GmVec3){0.0f, 0.0f, 0.0f});
	TmnfRaceState state;
	race_reset_at(&route.route, &state, &start);
	require(!state.outside_corridor && state.unwrapped_progress == 0.0f,
		"reset on the centerline is outside the corridor");

	/* Inside, off-centre: progress follows the projection. */
	(void)race_step_at_position(
		&route.route, &state, (GmVec3){10.0f, 0.0f, 27.0f});
	require(!state.outside_corridor &&
		state.unwrapped_progress == 10.0f &&
		state.arc_length == 10.0f &&
		state.corridor_lateral == 27.0f &&
		state.corridor_vertical == 0.0f &&
		state.lateral_offset == 27.0f,
		"car inside the corridor gained no progress");

	/* One metre beyond the lateral bound: frozen, projection still moves. */
	(void)race_step_at_position(
		&route.route, &state, (GmVec3){20.0f, 0.0f, 29.0f});
	require(state.outside_corridor &&
		state.unwrapped_progress == 10.0f &&
		state.previous_progress == 10.0f &&
		state.arc_length == 20.0f &&
		state.corridor_lateral == 29.0f &&
		state.half_width == 5.0f,
		"car beyond the corridor width gained progress");
	/* Frozen also means no reversal. */
	(void)race_step_at_position(
		&route.route, &state, (GmVec3){4.0f, 0.0f, 29.0f});
	require(state.outside_corridor &&
		state.unwrapped_progress == 10.0f &&
		state.arc_length == 4.0f,
		"frozen progress reversed outside the corridor");

	/* Re-entry credits the current projection at once. */
	(void)race_step_at_position(
		&route.route, &state, (GmVec3){30.0f, 0.0f, 28.0f});
	require(!state.outside_corridor &&
		state.unwrapped_progress == 30.0f &&
		state.previous_progress == 10.0f,
		"re-entering the corridor did not credit the projection");

	/* Vertical band: 16 m below is inside, one metre further is not;
	 * 44 m above is inside, 45 m is not. */
	(void)race_step_at_position(
		&route.route, &state, (GmVec3){40.0f, -16.0f, 0.0f});
	require(!state.outside_corridor &&
		state.unwrapped_progress == 40.0f &&
		state.corridor_vertical == -16.0f,
		"car 16 m below the route was frozen");
	(void)race_step_at_position(
		&route.route, &state, (GmVec3){50.0f, -17.0f, 0.0f});
	require(state.outside_corridor &&
		state.unwrapped_progress == 40.0f &&
		state.arc_length == 50.0f &&
		state.corridor_vertical == -17.0f &&
		state.corridor_lateral == 0.0f,
		"car 17 m below the route gained progress");
	(void)race_step_at_position(
		&route.route, &state, (GmVec3){60.0f, 44.0f, 0.0f});
	require(!state.outside_corridor &&
		state.unwrapped_progress == 60.0f,
		"car 44 m above the route was frozen");
	(void)race_step_at_position(
		&route.route, &state, (GmVec3){70.0f, 45.0f, 0.0f});
	require(state.outside_corridor &&
		state.unwrapped_progress == 60.0f &&
		state.best_progress == 60.0f,
		"car 45 m above the route gained progress");

	/* The 3 half-width factor binds once the road is wider than the floor:
	 * a 10 m half-width allows 30 m. */
	for (uint32_t i = 0; i < route.metadata.centerline_count; ++i)
		route.centerline[i].half_width = 10.0f;
	(void)race_step_at_position(
		&route.route, &state, (GmVec3){80.0f, 0.0f, 30.0f});
	require(!state.outside_corridor &&
		state.unwrapped_progress == 80.0f,
		"car at three half-widths was frozen");
	(void)race_step_at_position(
		&route.route, &state, (GmVec3){90.0f, 0.0f, 31.0f});
	require(state.outside_corridor &&
		state.unwrapped_progress == 80.0f,
		"car beyond three half-widths gained progress");

	/* A teleport (full-route reacquisition) evaluates the corridor too. */
	(void)race_step_at_position(
		&route.route, &state, (GmVec3){150.0f, -20.0f, 0.0f});
	require(state.outside_corridor &&
		state.unwrapped_progress == 80.0f &&
		state.arc_length == 150.0f &&
		state.corridor_vertical == -20.0f,
		"teleport outside the corridor gained progress");
	(void)race_step_at_position(
		&route.route, &state, (GmVec3){150.0f, 0.0f, 0.0f});
	require(!state.outside_corridor &&
		state.unwrapped_progress == 150.0f,
		"return after a teleport did not credit progress");
}

/* TMNF counts checkpoints in any order, once per lap; the finish counts
 * only after every checkpoint. The record replays (analysis/game_rules.md)
 * all match the ghost order, so the game's count-based rule was verified on
 * the A08 finish-before-checkpoint case: the start/finish contact at 1.16 s
 * is ignored because the lap's checkpoint has not been taken. */
static void test_alternative_finishes(void)
{
	TestRoute route;
	make_route(&route, 1, 1);
	TmnfRouteTrigger finishes[2] = {route.finish, route.finish};
	finishes[1].transform = identity_at(30.0f);
	finishes[1].race_index = 1;
	route.metadata.finish_count = 2;
	route.route.finish = finishes;
	TmnfRaceState state;
	GmIso4 start = identity_at(-10.0f);
	race_reset_at(&route.route, &state, &start);
	TmnfRaceStepResult result = race_step_at(&route, &state, 28.6f);
	require(!result.finished && state.trigger_contacts == (UINT64_C(1) << 62),
		"alternative finish must have an independent contact bit and reject early entry");
	result = race_step_at(&route, &state, -1.4f);
	require(result.checkpoint_accepted, "alternative-route checkpoint missing");
	result = race_step_at(&route, &state, 28.6f);
	require(result.finished && state.finish_time_ms == 30 && state.passed_race_checkpoints == 2,
		"alternative finish failed to complete the race");
	/* Entering another finish while the early finish stays contacted must
	 * still count. No extra state or union-of-finish contact edge is needed. */
	race_reset_at(&route.route, &state, &start);
	(void)TmnfRace_Step(&route.route, &state, UINT64_C(1) << 63, &start);
	(void)TmnfRace_Step(&route.route, &state, (UINT64_C(1) << 63) | 1, &start);
	result = TmnfRace_Step(&route.route, &state,
		(UINT64_C(1) << 63) | (UINT64_C(1) << 62), &start);
	require(result.finished, "early finish contact masked a different finish entry");
	/* Overlapping alternatives in one tick complete only one lap. */
	race_reset_at(&route.route, &state, &start);
	result = TmnfRace_Step(&route.route, &state,
		(UINT64_C(1) << 63) | (UINT64_C(1) << 62) | 1, &start);
	require(result.finished && state.completed_laps == 1 && state.passed_race_checkpoints == 2,
		"simultaneous alternative finishes counted more than once");
}

static void test_any_order_checkpoints_and_finish_time(void)
{
	/* Triggers are 2 m boxes at x = 0, 10 (checkpoints) and 20 (finish);
	 * the test car is a 0.5 m ellipsoid. A contact needs the car surface
	 * inside the box while its centre stays outside, as a car entering a
	 * trigger always is; x - 1.4 touches the near face from the left. */
	TestRoute route;
	make_route(&route, 2, 1);
	TmnfRaceState state;
	GmIso4 start = identity_at(-10.0f);
	race_reset_at(&route.route, &state, &start);

	TmnfRaceStepResult result = race_step_at(&route, &state, 18.6f);
	require(!result.finished && !result.lap_completed &&
		state.completed_laps == 0,
		"finish counted before any checkpoint");
	(void)race_step_at(&route, &state, 15.0f);
	result = race_step_at(&route, &state, 8.6f);
	require(result.checkpoint_accepted && result.checkpoint_index == 1 &&
		state.visited_count == 1 && state.visited_checkpoints == 2,
		"second checkpoint taken first was not accepted");
	TmnfGateObservations gates;
	TmnfRace_ObserveGates(&route.route, &state, &state.previous_car_transform, &gates);
	require(gates.remaining_checkpoints == 1 &&
		gates.gates[0].checkpoint_index == 0 && !gates.gates[1].valid &&
		!gates.gates[7].ready,
		"gate observations did not remove the out-of-order checkpoint");
	(void)race_step_at(&route, &state, 15.0f);
	result = race_step_at(&route, &state, 18.6f);
	require(!result.finished && state.completed_laps == 0,
		"finish counted with one checkpoint missing");
	(void)race_step_at(&route, &state, 15.0f);
	result = race_step_at(&route, &state, 8.6f);
	require(!result.checkpoint_accepted && result.checkpoint_repeated &&
		state.visited_count == 1 && state.passed_race_checkpoints == 1,
		"re-entering a visited checkpoint counted again");
	(void)race_step_at(&route, &state, 5.0f);
	result = race_step_at(&route, &state, -1.4f);
	require(result.checkpoint_accepted && result.checkpoint_index == 0 &&
		state.visited_count == 2 && state.visited_checkpoints == 3,
		"first checkpoint taken second was not accepted");
	TmnfRace_ObserveGates(&route.route, &state, &state.previous_car_transform, &gates);
	require(gates.remaining_checkpoints == 0 && gates.gates[7].ready,
		"finish observation was not unlocked by the last checkpoint");
	(void)race_step_at(&route, &state, 5.0f);
	(void)race_step_at(&route, &state, 15.0f);
	result = race_step_at(&route, &state, 18.6f);
	require(result.finished && state.finished &&
		result.race_time_ms == 120 &&
		state.passed_race_checkpoints == 3,
		"finish timing or checkpoint total is wrong");
	TmnfRace_ObserveGates(&route.route, &state, &state.previous_car_transform, &gates);
	require(gates.finished && !gates.remaining_laps && !gates.gates[7].valid,
		"finished race still exposed a finish obligation");
}

/* Staying inside a trigger is one contact edge, not one event per tick. */
static void test_continuous_contact_is_one_event(void)
{
	TestRoute route;
	make_route(&route, 1, 1);
	TmnfRaceState state;
	GmIso4 start = identity_at(-10.0f);
	race_reset_at(&route.route, &state, &start);
	TmnfRaceStepResult result = race_step_at(&route, &state, -1.4f);
	require(result.checkpoint_accepted, "checkpoint entry was missed");
	result = race_step_at(&route, &state, -1.2f);
	require(!result.checkpoint_accepted && !result.checkpoint_repeated,
		"continuous trigger contact produced a second event");
	/* Fully inside the box no face is in front of the ellipsoid centre, so
	 * the game's one-sided mesh test reports no contact. Leaving through
	 * the far face touches it from outside again: a second edge, which the
	 * visited set turns into a harmless repeat. */
	result = race_step_at(&route, &state, 0.0f);
	require(!result.checkpoint_accepted && !result.checkpoint_repeated &&
		state.trigger_contacts == 0,
		"car centre inside the trigger reported a contact");
	result = race_step_at(&route, &state, 1.2f);
	require(!result.checkpoint_accepted && result.checkpoint_repeated &&
		state.visited_count == 1,
		"leaving through the far face was not a repeat");
	(void)race_step_at(&route, &state, 5.0f);
	result = race_step_at(&route, &state, -1.4f);
	require(result.checkpoint_repeated,
		"re-entry after leaving was not reported");
}

/* Tracks like A10-Acrobatic have a start and a finish but no intermediate
 * checkpoints. The race must validate, progress, and finish with
 * checkpoint_count == 0. */
static void test_zero_checkpoint_route(void)
{
	TestRoute route;
	make_route(&route, 0, 1);
	TmnfRaceState state;
	GmIso4 start = identity_at(-10.0f);
	race_reset_at(&route.route, &state, &start);
	require(state.visited_count == 0 && state.trigger_contacts == 0,
		"zero-checkpoint reset state is wrong");

	TmnfRaceStepResult result = race_step_at(&route, &state, -5.0f);
	require(!result.finished && !result.checkpoint_accepted,
		"zero-checkpoint route finished before the finish trigger");
	(void)race_step_at(&route, &state, 5.0f);
	result = race_step_at(&route, &state, 8.6f);
	require(result.finished && state.finished &&
		state.passed_race_checkpoints == 1,
		"zero-checkpoint route did not finish at the finish trigger");
}

static void test_lap_counting(void)
{
	TestRoute route;
	make_route(&route, 1, 2);
	TmnfRaceState state;
	GmIso4 start = identity_at(-10.0f);
	race_reset_at(&route.route, &state, &start);

	(void)race_step_at(&route, &state, -1.4f);
	(void)race_step_at(&route, &state, 5.0f);
	float route_length = TmnfRoute_GetReferenceLength(&route.route);
	float total_length = route_length * route.metadata.lap_count;
	float progress_before_finish = state.unwrapped_progress;
	float potential_before_finish =
		-(total_length - progress_before_finish) / 50.0f;
	TmnfRaceStepResult result =
		race_step_at(&route, &state, 8.6f);
	require(result.lap_completed && !result.finished &&
		state.completed_laps == 1 && state.visited_count == 0 &&
		state.visited_checkpoints == 0,
		"first lap did not reset the visited checkpoint set");
	TmnfGateObservations gates;
	TmnfRace_ObserveGates(&route.route, &state, &state.previous_car_transform, &gates);
	require(gates.remaining_laps == 1 && gates.remaining_checkpoints == 1 &&
		gates.gates[0].valid && !gates.gates[7].ready,
		"gate observations failed to restore obligations on the next lap");
	float remaining_distance =
		total_length - state.unwrapped_progress;
	float boundary_potential = -remaining_distance / 50.0f;
	require(
		state.arc_length == 0.0f &&
		state.unwrapped_progress == route_length &&
		state.previous_progress == progress_before_finish &&
		remaining_distance == route_length,
		"non-final finish created phantom lap progress");
	require(
		fabsf(
			(boundary_potential - potential_before_finish) -
			(route_length - progress_before_finish) / 50.0f) <
			1.0e-6f,
		"potential is discontinuous at the lap boundary");
	(void)race_step_at(&route, &state, 5.0f);
	(void)race_step_at(&route, &state, -1.4f);
	(void)race_step_at(&route, &state, 5.0f);
	result = race_step_at(&route, &state, 8.6f);
	require(result.finished && state.completed_laps == 2 &&
		state.passed_race_checkpoints == 4 &&
		result.race_time_ms == 70,
		"multilap finish state is wrong");
}

static void test_env_determinism_and_restore(
	const char *track_path,
	const char *vehicle_path,
	const char *route_path)
{
	TmnfTrack *track = TmnfTrack_Load(track_path, A01_SHA256);
	TmnfWorld *owners[2] = {
		World_Create(track, vehicle_path),
		World_Create(track, vehicle_path),
	};
	TmnfPhysicsWorld *worlds[2] = {
		World_GetPhysicsWorld(owners[0]),
		World_GetPhysicsWorld(owners[1]),
	};
	uint32_t player_indices[2] = {0, 0};
	TmnfRoute *route = TmnfRoute_Load(route_path, A01_SHA256);
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	config.max_race_ticks = 2000;
	config.horizon_ticks = 1000;
	config.off_track_grace_ticks = 100;
	config.stuck_grace_ticks = 100;
	TmnfVecEnv env;
	TmnfVecEnv_Init(
		&env, worlds, player_indices, 2, route, &config);

	/* The reset state is the world's spawn (game state after the spawn
	 * tick), located at the route start; its velocities are not zero. */
	const CHmsStateDyna *spawn = World_GetPlayerState(owners[0]);
	require(memcmp(&spawn->pos, &route->start->initial_state.pos,
			sizeof(spawn->pos)) == 0 &&
		memcmp(&spawn->rot, &route->start->initial_state.rot,
			sizeof(spawn->rot)) == 0,
		"reset did not place the car at the route start");
	require(memcmp(spawn, World_GetPlayerState(owners[1]),
			sizeof(*spawn)) == 0,
		"two worlds reset to different spawn states");
	TmnfObservation reset_observations[2];
	TmnfVecEnv_Reset(&env, NULL, reset_observations);
	require(memcmp(
			&reset_observations[0], &reset_observations[1],
			sizeof(reset_observations[0])) == 0,
		"identical resets produced different observations");
	require(
		reset_observations[0].turbo_active == 0.0f &&
		reset_observations[0].turbo_type == 0.0f &&
		reset_observations[0].turbo_remaining_progress == 0.0f,
		"no-turbo reset observation is not zero");
	TmnfEnvSnapshot initial_snapshots[2];
	TmnfVecEnv_Capture(&env, initial_snapshots);
	CSceneVehicleCarAuxContext *aux =
		worlds[0]->corpora[0].vehicle_compute->aux;
	aux->turbo_type = TMNF_TURBO_NORMAL;
	aux->turbo_progress = 0.25f;
	uint32_t turbo_index = 0;
	TmnfEnvSnapshot turbo_snapshot;
	TmnfVecEnv_CaptureIndices(
		&env, &turbo_index, 1, &turbo_snapshot);
	aux->turbo_type = TMNF_TURBO_NONE;
	aux->turbo_progress = 0.0f;
	TmnfObservation turbo_observation;
	TmnfVecEnv_RestoreIndices(
		&env, &turbo_index, 1, &turbo_snapshot,
		&turbo_observation);
	require(
		turbo_observation.turbo_active == 1.0f &&
		turbo_observation.turbo_type == (float)TMNF_TURBO_NORMAL &&
		turbo_observation.turbo_remaining_progress == 0.75f,
		"turbo observation or snapshot restore is wrong");
	TmnfVecEnv_Restore(&env, initial_snapshots);
	test_dense_lookahead(route, &reset_observations[0]);
	test_local_width_and_jump_gaps(route);
	test_a01_forced_dense_branch(route);

	uint8_t actions[2] = {4, 4};
	TmnfStepResult results[2];
	for (uint32_t tick = 0; tick < 12; ++tick) {
		TmnfVecEnv_StepDiscrete(&env, actions, 1, results);
		require(memcmp(
				&results[0].observation,
				&results[1].observation,
				sizeof(results[0].observation)) == 0,
			"identical actions changed observation bytes");
		require(memcmp(
				&results[0].reward,
				&results[1].reward,
				sizeof(results[0].reward)) == 0,
			"identical actions changed reward bytes");
	}

	env.race_states[0].visited_checkpoints = 1;
	env.race_states[0].visited_count = 1;
	env.race_states[0].passed_race_checkpoints = 1;
	TmnfEnvSnapshot snapshots[2];
	TmnfVecEnv_Capture(&env, snapshots);
	require(
		snapshots[0].version == TMNF_ENV_SNAPSHOT_VERSION,
		"snapshot version is wrong");
	TmnfRaceState saved_race = snapshots[0].race_state;
	require(saved_race.visited_count == 1 &&
		saved_race.visited_checkpoints == 1 &&
		saved_race.passed_race_checkpoints == 1,
		"snapshot did not capture checkpoint progress");
	TmnfStepResult expected[3];
	for (uint32_t tick = 0; tick < 3; ++tick) {
		TmnfVecEnv_StepDiscrete(&env, actions, 1, results);
		expected[tick] = results[0];
	}
	TmnfVecEnv_Restore(&env, snapshots);
	require(memcmp(
			&env.race_states[0], &saved_race,
			sizeof(saved_race)) == 0,
		"snapshot restore lost race state");
	for (uint32_t tick = 0; tick < 3; ++tick) {
		TmnfVecEnv_StepDiscrete(&env, actions, 1, results);
		require(memcmp(
				&results[0].observation,
				&expected[tick].observation,
				sizeof(results[0].observation)) == 0 &&
			memcmp(
				&results[0].reward,
				&expected[tick].reward,
				sizeof(results[0].reward)) == 0,
			"restore/replay changed transition bytes");
	}

	TmnfVecEnv_Capture(&env, snapshots);
	TmnfStepResult repeated;
	TmnfVecEnv_StepDiscrete(&env, actions, 3, results);
	repeated = results[0];
	TmnfVecEnv_Restore(&env, snapshots);
	float combined_reward = 0.0f;
	float combined_discount = 1.0f;
	for (uint32_t tick = 0; tick < 3; ++tick) {
		TmnfVecEnv_StepDiscrete(&env, actions, 1, results);
		combined_reward += combined_discount * results[0].reward;
		combined_discount *= results[0].transition_discount;
	}
	require(memcmp(
			&repeated.observation, &results[0].observation,
			sizeof(repeated.observation)) == 0 &&
		memcmp(
			&repeated.reward, &combined_reward,
			sizeof(repeated.reward)) == 0 &&
		memcmp(
			&repeated.transition_discount, &combined_discount,
			sizeof(repeated.transition_discount)) == 0,
		"action repeat differs from one-tick transitions");

	TmnfVecEnv_Destroy(&env);
	TmnfRoute_Unload(route);
	World_Destroy(owners[1]);
	World_Destroy(owners[0]);
	TmnfTrack_Unload(track);
}

static uint32_t lcg_next(uint32_t *state)
{
	*state = *state * 1664525u + 1013904223u;
	return *state >> 8;
}

/*
 * A snapshot is a state transfer between environments: env 0's snapshot
 * restored into env 1 must continue byte-identically to env 0 under identical
 * actions, through wall contacts and resets, and two environments in the
 * same state must produce byte-identical snapshots (no address of either
 * environment may be inside one).
 */
static void test_cross_env_restore(
	const char *track_path,
	const char *vehicle_path,
	const char *route_path)
{
	TmnfTrack *track = TmnfTrack_Load(track_path, A01_SHA256);
	TmnfWorld *owners[2] = {
		World_Create(track, vehicle_path),
		World_Create(track, vehicle_path),
	};
	TmnfPhysicsWorld *worlds[2] = {
		World_GetPhysicsWorld(owners[0]),
		World_GetPhysicsWorld(owners[1]),
	};
	uint32_t player_indices[2] = {0, 0};
	TmnfRoute *route = TmnfRoute_Load(route_path, A01_SHA256);
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	config.max_race_ticks = 1500;
	config.horizon_ticks = 1500;
	config.off_track_grace_ticks = 100;
	config.stuck_grace_ticks = 100;
	TmnfVecEnv env;
	TmnfVecEnv_Init(&env, worlds, player_indices, 2, route, &config);

	TmnfEnvSnapshot snapshots[2];
	TmnfVecEnv_Capture(&env, snapshots);
	require(memcmp(&snapshots[0], &snapshots[1], sizeof(snapshots[0])) == 0,
		"identical spawn states produced different snapshot bytes");

	uint32_t rng = 7u;
	uint8_t actions[2];
	TmnfStepResult results[2];
	/* Drive both identically for a while (full throttle, wandering steer)
	 * so wheel and body contacts exist, then snapshots must still match. */
	for (uint32_t tick = 0; tick < 300; ++tick) {
		actions[0] = (uint8_t)(3 + lcg_next(&rng) % 3);
		actions[1] = actions[0];
		TmnfVecEnv_StepDiscrete(&env, actions, 1, results);
	}
	TmnfVecEnv_Capture(&env, snapshots);
	require(memcmp(&snapshots[0], &snapshots[1], sizeof(snapshots[0])) == 0,
		"identical trajectories produced different snapshot bytes");
	int any_contact_body = 0;
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		if (snapshots[0].contact_body_refs[i] != TMNF_ENV_NO_CONTACT_BODY)
			any_contact_body = 1;
		require(snapshots[0].contact_wheels[i].contact_body == NULL &&
			snapshots[0].contact_wheels[i].wheel == NULL,
			"snapshot carries a contact wheel pointer");
	}
	require(any_contact_body, "driving left no wheel contact body token");
	require(snapshots[0].contact.air_control_immediate == NULL &&
		snapshots[0].contact.event_metric_c == NULL &&
		snapshots[0].aux.runtime == NULL && snapshots[0].car.wheels == NULL,
		"snapshot carries topology pointers");

	/* Let env 1 diverge, then move env 0's state into it. */
	for (uint32_t tick = 0; tick < 200; ++tick) {
		actions[0] = 4;
		actions[1] = (uint8_t)(lcg_next(&rng) % TMNF_DISCRETE_ACTION_COUNT);
		TmnfVecEnv_StepDiscrete(&env, actions, 1, results);
	}
	TmnfEnvSnapshot source;
	uint32_t source_index = 0;
	uint32_t target_index = 1;
	TmnfVecEnv_CaptureIndices(&env, &source_index, 1, &source);
	TmnfObservation restored;
	TmnfVecEnv_RestoreIndices(&env, &target_index, 1, &source, &restored);
	TmnfVecEnv_Capture(&env, snapshots);
	require(memcmp(&snapshots[0], &snapshots[1], sizeof(snapshots[0])) == 0,
		"cross-env restore did not reproduce the source snapshot");

	/* Lockstep through the rest of the episode and past a reset. */
	uint32_t resets = 0;
	for (uint32_t tick = 0; tick < 2500; ++tick) {
		actions[0] = (uint8_t)(3 + lcg_next(&rng) % 3);
		actions[1] = actions[0];
		TmnfVecEnv_StepDiscrete(&env, actions, 1, results);
		require(memcmp(results, &results[1], sizeof(results[0])) == 0,
			"cross-env restored environment diverged from its source");
		if (results[0].final_observation_valid)
			resets++;
	}
	require(resets > 0, "lockstep never crossed an episode end");
	TmnfVecEnv_Capture(&env, snapshots);
	require(memcmp(&snapshots[0], &snapshots[1], sizeof(snapshots[0])) == 0,
		"lockstep environments ended with different snapshot bytes");

	TmnfVecEnv_Destroy(&env);
	TmnfRoute_Unload(route);
	World_Destroy(owners[1]);
	World_Destroy(owners[0]);
	TmnfTrack_Unload(track);
}

/* Fastest finish the game allows: laps * route length at the 1,000 km/h
 * rigid-body speed cap, in 10 ms ticks. */
static uint32_t minimum_finish_ticks(const TmnfRoute *route)
{
	double total = (double)TmnfRoute_GetReferenceLength(route) *
		route->metadata->lap_count;
	return (uint32_t)ceil(total / TMNF_MAX_LINEAR_SPEED_MPS / 0.01);
}

static void expect_init_abort(
	TmnfPhysicsWorld **world, uint32_t *player_index,
	const TmnfRoute *route, const TmnfVecEnvConfig *config,
	const char *message)
{
	fflush(stderr);
	pid_t child = fork();
	require(child >= 0, "fork failed");
	if (child == 0) {
		int null = open("/dev/null", O_WRONLY);
		if (null >= 0)
			dup2(null, STDERR_FILENO);
		TmnfVecEnv env;
		TmnfVecEnv_Init(&env, world, player_index, 1, route, config);
		_exit(0);
	}
	int status = 0;
	require(waitpid(child, &status, 0) == child, "waitpid failed");
	require(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT, message);
}

static void test_horizon_derivation_and_fail_fast(
	TmnfPhysicsWorld **world, uint32_t *player_index,
	const TmnfRoute *route)
{
	uint32_t minimum = minimum_finish_ticks(route);
	require(minimum == 796, "A01 minimum finish tick fixture changed");
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	require(config.horizon_ticks == 0 && config.max_race_ticks == 0,
		"default configuration must derive both budgets");
	TmnfVecEnv env;
	TmnfVecEnv_Init(&env, world, player_index, 1, route, &config);
	/* 2 * 2209.685 m / 50 m/s = 88.39 s -> 8,839 ticks. */
	require(env.config.horizon_ticks == 8839,
		"derived horizon is not twice the route at reference speed");
	require(env.config.max_race_ticks == 8839,
		"derived race timeout is not twice the route at reference speed");
	TmnfVecEnv_Destroy(&env);

	/* No clamp: a short explicit race timeout leaves the horizon derived. */
	config.max_race_ticks = 1000;
	TmnfVecEnv_Init(&env, world, player_index, 1, route, &config);
	require(env.config.horizon_ticks == 8839,
		"explicit race timeout altered the derived horizon");
	require(env.config.max_race_ticks == 1000,
		"explicit race timeout was altered");
	TmnfVecEnv_Destroy(&env);

	config.max_race_ticks = 12000;
	config.horizon_ticks = minimum;
	TmnfVecEnv_Init(&env, world, player_index, 1, route, &config);
	require(env.config.horizon_ticks == minimum,
		"explicit horizon at the speed-cap bound was altered");
	TmnfVecEnv_Destroy(&env);

	config.horizon_ticks = minimum - 1;
	expect_init_abort(world, player_index, route, &config,
		"horizon below the speed-cap finish bound did not abort");
	config.horizon_ticks = 0;
	config.max_race_ticks = minimum - 1;
	expect_init_abort(world, player_index, route, &config,
		"race timeout below the speed-cap finish bound did not abort");
}

static void place_car(TmnfPhysicsCorpus *corpus, GmVec3 position)
{
	corpus->dyna->liveState->pos = position;
	corpus->dyna->stateB->pos = position;
	corpus->dyna->tempState.pos = position;
	GmVec3 zero = {0.0f, 0.0f, 0.0f};
	corpus->dyna->liveState->linVel = zero;
	corpus->dyna->stateB->linVel = zero;
	corpus->dyna->tempState.linVel = zero;
	corpus->dyna->liveState->angVel = zero;
	corpus->dyna->stateB->angVel = zero;
	corpus->dyna->tempState.angVel = zero;
}

/* Every TmnfVecEnv_Init needs a world at its spawn state: Init captures the
 * world as the reset state and refuses a world that has been stepped. */
static void renew_world(
	TmnfTrack *track, const char *vehicle_path,
	TmnfWorld **owner, TmnfPhysicsWorld **world)
{
	if (*owner != NULL)
		World_Destroy(*owner);
	*owner = World_Create(track, vehicle_path);
	*world = World_GetPhysicsWorld(*owner);
}

static void test_autoreset_and_failures(
	const char *track_path,
	const char *vehicle_path,
	const char *route_path)
{
	TmnfTrack *track = TmnfTrack_Load(track_path, A01_SHA256);
	TmnfWorld *owner = NULL;
	TmnfPhysicsWorld *world = NULL;
	renew_world(track, vehicle_path, &owner, &world);
	uint32_t player_index = 0;
	TmnfRoute *route = TmnfRoute_Load(route_path, A01_SHA256);
	test_horizon_derivation_and_fail_fast(&world, &player_index, route);
	uint32_t minimum = minimum_finish_ticks(route);

	/* Truncation and timeout runs hold full throttle for 8 s from the A01
	 * start; the failure rules must stay out of the way. */
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	config.max_race_ticks = 2000;
	config.horizon_ticks = minimum;
	config.off_track_grace_ticks = 1000;
	config.stuck_grace_ticks = 1000;
	TmnfVecEnv env;
	TmnfVecEnv_Init(
		&env, &world, &player_index, 1, route, &config);
	uint8_t action = 1;
	TmnfStepResult result;
	TmnfVecEnv_StepDiscrete(&env, &action, minimum, &result);
	require(!result.terminated && result.truncated &&
		result.executed_ticks == minimum &&
		result.final_observation_valid && !result.reset_only,
		"SAME_STEP truncation flags are wrong");
	require(memcmp(
			&result.observation.position,
			&route->start->initial_state.pos,
			sizeof(GmVec3)) == 0,
		"SAME_STEP did not return the reset observation");
	TmnfVecEnv_Destroy(&env);

	/* Autoreset leaves the world at its spawn, so a second Init over it is
	 * legal; a world stepped outside the env is not a spawn and is refused. */
	TmnfVecEnv_Init(
		&env, &world, &player_index, 1, route, &config);
	TmnfVecEnv_Destroy(&env);
	TMNFRaceInputs gas = { .accelerate = 1 };
	for (uint32_t tick = 0; tick < 50; ++tick) {
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
			&gas, World_GetPlayerVehicle(owner));
		World_AdvanceTimer(owner, TMNF_RACE_TICK_MS);
		CHmsZoneDynamic_PhysicsStep2(world, TMNF_RACE_TICK_MS);
	}
	expect_init_abort(&world, &player_index, route, &config,
		"init over a stepped world did not abort");

	renew_world(track, vehicle_path, &owner, &world);
	config.max_race_ticks = minimum;
	config.horizon_ticks = minimum + 1;
	TmnfVecEnv_Init(
		&env, &world, &player_index, 1, route, &config);
	TmnfVecEnv_StepDiscrete(&env, &action, minimum, &result);
	require(result.terminated && !result.truncated &&
		result.termination_reason == TMNF_TERMINATION_TIMEOUT &&
		isfinite(result.reward),
		"race timeout did not produce bounded termination");
	TmnfVecEnv_Destroy(&env);

	renew_world(track, vehicle_path, &owner, &world);
	config.max_race_ticks = 2000;
	config.horizon_ticks = minimum;
	config.autoreset_mode = TMNF_AUTORESET_NEXT_STEP;
	TmnfVecEnv_Init(
		&env, &world, &player_index, 1, route, &config);
	TmnfVecEnv_StepDiscrete(&env, &action, minimum, &result);
	require(!result.terminated && result.truncated &&
		result.final_observation_valid &&
		memcmp(
			&result.observation,
			&result.final_observation,
			sizeof(result.observation)) == 0,
		"NEXT_STEP did not return the terminal observation");
	TmnfVecEnv_StepDiscrete(&env, &action, 1, &result);
	require(result.reset_only && result.reward == 0.0f &&
		!result.terminated && !result.truncated,
		"NEXT_STEP did not emit one reset-only slot");
	TmnfVecEnv_Destroy(&env);

	renew_world(track, vehicle_path, &owner, &world);
	config.autoreset_mode = TMNF_AUTORESET_SAME_STEP;
	config.max_race_ticks = 2000;
	config.horizon_ticks = 1000;
	config.off_track_grace_ticks = 10;
	config.stuck_grace_ticks = 1;
	config.stuck_progress_epsilon = 1.0f;
	TmnfVecEnv_Init(
		&env, &world, &player_index, 1, route, &config);
	TmnfVecEnv_StepDiscrete(&env, &action, 1, &result);
	require(result.terminated &&
		result.termination_reason == TMNF_TERMINATION_STUCK,
		"stuck state did not terminate");
	TmnfVecEnv_Destroy(&env);

	renew_world(track, vehicle_path, &owner, &world);
	config.off_track_grace_ticks = 1;
	config.stuck_grace_ticks = 10;
	config.stuck_progress_epsilon = 0.001f;
	TmnfVecEnv_Init(
		&env, &world, &player_index, 1, route, &config);
	TmnfPhysicsCorpus *corpus = &world->corpora[player_index];
	const TmnfRouteReferencePoint *jump =
		&TmnfRoute_GetReferencePoints(route)[860];
	GmVec3 airborne = jump->position;
	airborne.y += 16.0f;
	place_car(corpus, airborne);
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i)
		corpus->vehicle->wheels[i].real_time.has_ground_contact = 0;
	TmnfVecEnv_StepDiscrete(&env, &action, 1, &result);
	require(result.termination_reason != TMNF_TERMINATION_OFF_TRACK &&
		env.race_states[0].off_track_ticks == 0,
		"airborne jump state terminated as off-track");
	TmnfVecEnv_Destroy(&env);

	/* Ground plane: the A01 reverse-recovery capture rests all four wheels
	 * on Grass at (120.7, 9.02, 688.0), 79 m below the road. Dropped there,
	 * the car must terminate as off-track once the grace count is reached,
	 * and never before wheel contact exists. */
	renew_world(track, vehicle_path, &owner, &world);
	corpus = &world->corpora[player_index];
	config.off_track_grace_ticks = 25;
	config.stuck_grace_ticks = 1000;
	config.horizon_ticks = 1000;
	TmnfVecEnv_Init(
		&env, &world, &player_index, 1, route, &config);
	place_car(corpus, (GmVec3){120.7087f, 9.4f, 687.969f});
	uint32_t grass_ticks = 0;
	uint32_t steps = 0;
	uint8_t coast = 0;
	for (; steps < 400; ++steps) {
		uint32_t on_grass = 0;
		uint32_t in_contact = 0;
		TmnfVecEnv_StepDiscrete(&env, &coast, 1, &result);
		if (result.terminated)
			break;
		for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
			if (result.observation.wheel_contact[i] != 0.0f) {
				in_contact++;
				if (TmnfRace_IsGroundPlaneMaterial(
						(int32_t)result.observation.wheel_material[i]))
					on_grass++;
			}
		}
		if (in_contact != 0 && in_contact == on_grass)
			grass_ticks++;
		require(env.race_states[0].off_track_ticks == grass_ticks,
			"off-track count does not track grass-only contact ticks");
	}
	require(result.terminated &&
		result.termination_reason == TMNF_TERMINATION_OFF_TRACK &&
		result.final_observation_valid &&
		grass_ticks + 1 == config.off_track_grace_ticks,
		"car resting on the ground plane was not terminated as off-track");
	TmnfVecEnv_Destroy(&env);

	TmnfRoute_Unload(route);
	World_Destroy(owner);
	TmnfTrack_Unload(track);
}

/* Discrete action of a game input packet, as vec_env_matches_replay maps
 * the committed policy lap. */
static uint8_t discrete_action_of(const TMNFRaceInputs *input)
{
	CSceneVehicleCar scratch;
	memset(&scratch, 0, sizeof(scratch));
	CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
		input, &scratch);
	uint32_t steering = scratch.input_steer < 0.0f ? 0
		: scratch.input_steer > 0.0f ? 2 : 1;
	uint32_t longitudinal = (scratch.input_gas != 0.0f ? 1u : 0u) +
		(scratch.input_brake != 0.0f ? 2u : 0u);
	return (uint8_t)(longitudinal * 3 + steering);
}

static TMNFRaceInputs *read_schedule(const char *path, uint32_t *count)
{
	FILE *file = fopen(path, "rb");
	require(file != NULL, "cannot open the policy lap schedule");
	require(fseek(file, 0, SEEK_END) == 0, "cannot seek the schedule");
	long size = ftell(file);
	require(size > 0 && size % (long)sizeof(TMNFRaceInputs) == 0,
		"schedule is not whole records");
	rewind(file);
	*count = (uint32_t)(size / (long)sizeof(TMNFRaceInputs));
	TMNFRaceInputs *inputs = malloc((size_t)size);
	require(inputs != NULL &&
		fread(inputs, sizeof(*inputs), *count, file) == *count,
		"cannot read the schedule");
	fclose(file);
	return inputs;
}

typedef struct {
	double discounted_return;   /* sum_t gamma^(t-1) r_t at repeat 1 */
	float initial_potential;    /* phi(s_0) */
	float terminal_potential;   /* phi(s_T) from final_observation */
	float potential_before_last;/* phi(s_(T-1)) */
	float last_reward;          /* r_T */
	uint32_t ticks;             /* T */
	TmnfTerminationReason reason;
} Episode;

/* Same expression as race_potential(): the observation's remaining distance
 * is total - unwrapped_progress, so this is bit-identical to the env's phi. */
static float potential_of(const TmnfObservation *observation, float speed)
{
	return -observation->remaining_distance / speed;
}

/* Mirrors failure_base_reward() in src/vec_env.c, operation for operation. */
static float failure_lump(const TmnfVecEnvConfig *config, uint32_t elapsed_before)
{
	uint32_t remaining = config->max_race_ticks - elapsed_before;
	float gamma = config->discount_per_tick;
	return -0.01f * (1.0f - powf(gamma, (float)remaining)) / (1.0f - gamma);
}

/* Steps one environment from a fresh reset until the episode ends, holding
 * one action or following a schedule. */
static Episode run_episode(
	TmnfVecEnv *env, uint8_t action,
	const TMNFRaceInputs *schedule, uint32_t schedule_ticks)
{
	Episode episode = {0};
	float speed = env->config.reference_speed;
	TmnfObservation reset;
	TmnfVecEnv_Reset(env, NULL, &reset);
	episode.initial_potential = potential_of(&reset, speed);
	float previous_potential = episode.initial_potential;
	double weight = 1.0;
	for (;;) {
		uint8_t step_action = action;
		if (schedule != NULL) {
			require(episode.ticks < schedule_ticks,
				"schedule ran out before the episode ended");
			step_action = discrete_action_of(&schedule[episode.ticks]);
		}
		TmnfStepResult result;
		TmnfVecEnv_StepDiscrete(env, &step_action, 1, &result);
		require(result.executed_ticks == 1 && !result.reset_only,
			"episode step did not execute one tick");
		episode.ticks++;
		episode.discounted_return += weight * (double)result.reward;
		weight *= (double)env->config.discount_per_tick;
		if (result.terminated || result.truncated) {
			require(result.terminated && !result.truncated &&
				result.final_observation_valid,
				"identity episode ended by truncation");
			episode.reason = result.termination_reason;
			episode.terminal_potential =
				potential_of(&result.final_observation, speed);
			episode.potential_before_last = previous_potential;
			episode.last_reward = result.reward;
			return episode;
		}
		previous_potential = potential_of(&result.observation, speed);
	}
}

/* G_finish = -phi(s_0) - 0.01 (1 - gamma^T) / (1 - gamma) */
static double finish_identity(const TmnfVecEnvConfig *config, const Episode *e)
{
	double gamma = config->discount_per_tick;
	return -(double)e->initial_potential -
		0.01 * (1.0 - pow(gamma, (double)e->ticks)) / (1.0 - gamma);
}

/* G_fail = gamma^Tmax phi(s_T) - phi(s_0) - 0.01 (1 - gamma^Tmax) / (1 - gamma),
 * whatever the failure tick T. */
static double failure_identity(const TmnfVecEnvConfig *config, const Episode *e)
{
	double gamma = config->discount_per_tick;
	double budget = pow(gamma, (double)config->max_race_ticks);
	return budget * (double)e->terminal_potential -
		(double)e->initial_potential -
		0.01 * (1.0 - budget) / (1.0 - gamma);
}

/* Every tick reward is a float32 sum around a potential of about -44 (ulp
 * 3.8e-6); the double accumulation of T of them is exact to T * 4e-6. */
static double identity_tolerance(uint32_t ticks)
{
	return 4.0e-6 * (double)ticks + 1.0e-4;
}

static void require_failure_episode(
	const TmnfVecEnvConfig *config, const Episode *e,
	TmnfTerminationReason reason, const char *label)
{
	fprintf(stderr, "race env: %s: reason %u, T %u, progress %.1f m, "
		"G %.6f, identity %.6f\n",
		label, e->reason, e->ticks,
		(double)(config->reference_speed *
			(e->terminal_potential - e->initial_potential)),
		e->discounted_return, failure_identity(config, e));
	require(e->reason == reason, "failure episode ended for another reason");
	require(fabs(e->discounted_return - failure_identity(config, e)) <
			identity_tolerance(e->ticks),
		"failure return does not match gamma^Tmax phi(s_T) - phi(s_0) - c");
	/* The terminal tick itself: lump + gamma gamma^(Tmax - T) phi(s_T) -
	 * phi(s_(T-1)), the float32 operations of tick_reward() in order. */
	float deferred = e->terminal_potential * powf(config->discount_per_tick,
		(float)(config->max_race_ticks - e->ticks));
	float expected_last = failure_lump(config, e->ticks - 1) +
		config->discount_per_tick * deferred -
		e->potential_before_last;
	require(memcmp(&expected_last, &e->last_reward, sizeof(float)) == 0,
		"failure tick reward is not lump + gamma^(Tmax - T + 1) phi(s_T) - phi(s_(T-1))");
}

/*
 * Reward identities on A01 (docs: analysis/rl_env.md, src/vec_env.c
 * tick_reward). One race budget for the finish and two failures, so the
 * time constant c = 0.01 (1 - gamma^Tmax) / (1 - gamma) is shared:
 *   finish (committed policy lap, 2,527 ticks) = -phi(s_0) - 0.01 (1 - gamma^T) / (1 - gamma)
 *   failures (coast, full gas)                 = gamma^Tmax phi(s_T) - phi(s_0) - c
 * The finish must beat both failures and the failure with more progress must
 * beat the one with less by gamma^Tmax (phi_gas - phi_coast). A second budget
 * checks a stuck failure (idle car, grace 100, T = 100 of 1,000) against a
 * full-gas failure with the same Tmax: same identity, no term in T. A car
 * dropped below the world floor ends as FELL with the same terminal-tick
 * arithmetic.
 */
static void test_return_identities(
	const char *track_path, const char *vehicle_path,
	const char *route_path, const char *schedule_path)
{
	TmnfTrack *track = TmnfTrack_Load(track_path, A01_SHA256);
	TmnfRoute *route = TmnfRoute_Load(route_path, A01_SHA256);
	uint32_t schedule_ticks = 0;
	TMNFRaceInputs *schedule = read_schedule(schedule_path, &schedule_ticks);
	TmnfWorld *owner = NULL;
	TmnfPhysicsWorld *world = NULL;
	uint32_t player_index = 0;
	TmnfVecEnv env;
	const uint8_t coast = 1;     /* no input */
	const uint8_t full_gas = 4;  /* gas, neutral steering */

	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	config.max_race_ticks = schedule_ticks + 1000;
	config.horizon_ticks = config.max_race_ticks + 1;
	config.off_track_grace_ticks = UINT32_MAX;
	config.stuck_grace_ticks = UINT32_MAX;

	renew_world(track, vehicle_path, &owner, &world);
	TmnfVecEnv_Init(&env, &world, &player_index, 1, route, &config);
	Episode finish = run_episode(&env, 0, schedule, schedule_ticks);
	TmnfVecEnv_Destroy(&env);
	require(finish.reason == TMNF_TERMINATION_FINISH &&
		finish.ticks == 2527 && finish.terminal_potential == 0.0f,
		"policy lap did not finish at tick 2527 with zero remaining");
	fprintf(stderr, "race env: finish: T %u, G %.6f, identity %.6f\n",
		finish.ticks, finish.discounted_return,
		finish_identity(&config, &finish));
	require(fabs(finish.discounted_return - finish_identity(&config, &finish)) <
			identity_tolerance(finish.ticks),
		"finish return does not match -phi(s_0) - 0.01 (1 - gamma^T) / (1 - gamma)");
	float expected_finish_last = -0.01f +
		config.discount_per_tick * 0.0f - finish.potential_before_last;
	require(memcmp(&expected_finish_last, &finish.last_reward,
			sizeof(float)) == 0,
		"finish tick reward is not -0.01 - phi(s_(T-1))");

	renew_world(track, vehicle_path, &owner, &world);
	TmnfVecEnv_Init(&env, &world, &player_index, 1, route, &config);
	Episode coasting = run_episode(&env, coast, NULL, 0);
	TmnfVecEnv_Destroy(&env);
	require_failure_episode(&config, &coasting, TMNF_TERMINATION_TIMEOUT,
		"coast to timeout");
	require(coasting.ticks == config.max_race_ticks,
		"coasting car did not time out at max_race_ticks");

	renew_world(track, vehicle_path, &owner, &world);
	TmnfVecEnv_Init(&env, &world, &player_index, 1, route, &config);
	Episode driving = run_episode(&env, full_gas, NULL, 0);
	TmnfVecEnv_Destroy(&env);
	require_failure_episode(&config, &driving, TMNF_TERMINATION_TIMEOUT,
		"full gas to timeout");
	require(driving.terminal_potential > coasting.terminal_potential + 1.0f,
		"full gas did not progress at least 50 m beyond coasting");

	/* Ordering: finish over both failures, more progress over less by
	 * exactly gamma^Tmax (phi_gas - phi_coast). */
	double gamma = config.discount_per_tick;
	double progress_gap = pow(gamma, (double)config.max_race_ticks) *
		((double)driving.terminal_potential -
		 (double)coasting.terminal_potential);
	require(finish.discounted_return > driving.discounted_return + 1.0 &&
		driving.discounted_return > coasting.discounted_return + 0.5,
		"finish does not beat every failure or progress does not order failures");
	require(fabs((driving.discounted_return - coasting.discounted_return) -
			progress_gap) <
			identity_tolerance(2 * config.max_race_ticks),
		"failure return difference is not gamma^Tmax (phi_a - phi_b)");

	/* Second budget: idle car ends STUCK (progress-only rule), full gas
	 * ends by the same budget's timeout or a stuck against a wall; both
	 * satisfy the failure identity and the one with more progress wins. */
	config.max_race_ticks = 1000;
	config.horizon_ticks = 1001;
	config.stuck_grace_ticks = 100;
	renew_world(track, vehicle_path, &owner, &world);
	TmnfVecEnv_Init(&env, &world, &player_index, 1, route, &config);
	Episode idle = run_episode(&env, coast, NULL, 0);
	TmnfVecEnv_Destroy(&env);
	require_failure_episode(&config, &idle, TMNF_TERMINATION_STUCK,
		"idle to stuck");
	require(idle.ticks < config.max_race_ticks,
		"idle car was not ended by the stuck rule before the timeout");
	renew_world(track, vehicle_path, &owner, &world);
	TmnfVecEnv_Init(&env, &world, &player_index, 1, route, &config);
	Episode driving_short = run_episode(&env, full_gas, NULL, 0);
	TmnfVecEnv_Destroy(&env);
	require_failure_episode(&config, &driving_short, driving_short.reason,
		"full gas under the 1000-tick budget");
	require(driving_short.reason == TMNF_TERMINATION_TIMEOUT ||
		driving_short.reason == TMNF_TERMINATION_STUCK,
		"full gas under the short budget ended for an unexpected reason");
	require(driving_short.discounted_return > idle.discounted_return + 0.5,
		"progress before failing did not raise the return");

	/* FELL: one tick below the route floor minus one block. */
	renew_world(track, vehicle_path, &owner, &world);
	TmnfVecEnv_Init(&env, &world, &player_index, 1, route, &config);
	TmnfPhysicsCorpus *corpus = &world->corpora[player_index];
	GmVec3 below = route->start->initial_state.pos;
	below.y = env.fell_floor_y - 1.0f;
	TmnfObservation reset;
	TmnfVecEnv_Reset(&env, NULL, &reset);
	place_car(corpus, below);
	TmnfStepResult result;
	TmnfVecEnv_StepDiscrete(&env, &coast, 1, &result);
	require(result.terminated &&
		result.termination_reason == TMNF_TERMINATION_FELL &&
		result.final_observation.position.y < env.fell_floor_y,
		"car below the world floor did not end as FELL");
	float fell_deferred =
		potential_of(&result.final_observation, config.reference_speed) *
		powf(config.discount_per_tick, (float)(config.max_race_ticks - 1));
	float fell_expected = failure_lump(&config, 0) +
		config.discount_per_tick * fell_deferred -
		potential_of(&reset, config.reference_speed);
	require(memcmp(&fell_expected, &result.reward, sizeof(float)) == 0,
		"FELL tick reward is not lump + gamma phi(s_T) - phi(s_0)");
	TmnfVecEnv_Destroy(&env);

	free(schedule);
	TmnfRoute_Unload(route);
	World_Destroy(owner);
	TmnfTrack_Unload(track);
}

int main(int argc, char **argv)
{
	if (argc != 5) {
		fprintf(stderr,
			"usage: race_env TRACK VEHICLE ROUTE POLICY_LAP_INPUTS\n");
		return 2;
	}
	test_trigger_boundary();
	test_same_leg_self_intersection();
	test_teleport_dense_reacquisition();
	test_corridor_freezes_progress();
	test_alternative_finishes();
	test_any_order_checkpoints_and_finish_time();
	test_continuous_contact_is_one_event();
	test_zero_checkpoint_route();
	test_lap_counting();
	test_env_determinism_and_restore(
		argv[1], argv[2], argv[3]);
	test_cross_env_restore(argv[1], argv[2], argv[3]);
	test_autoreset_and_failures(
		argv[1], argv[2], argv[3]);
	test_return_identities(argv[1], argv[2], argv[3], argv[4]);
	printf(
		"race_env: trigger/projection/corridor/any-order/laps/timing/"
		"snapshot/cross-env-restore/determinism/autoreset/horizon/"
		"off-track/return-identities passed\n");
	return 0;
}
