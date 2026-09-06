/*
 * Replays an exact per-tick input schedule through the physics and the race
 * layer, and reports every tick where an environment rule would have acted:
 * checkpoint trigger entries (with their route index), width violations while
 * grounded, wheel-contact loss, body collisions, stuck ticks, corridor exits
 * (frozen progress) with the extreme horizontal and vertical offsets reached
 * grounded and airborne, and the finish tick. Rules are evaluated as
 * observers here; nothing terminates the run.
 *
 * usage: wr_passthrough TRACK VEHICLE ROUTE INPUTS TRACK_SHA256
 *        [--csv TICK_CSV] [--expect-finish-ms N]
 */
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "race.h"
#include "world.h"

enum {
	TICK_MS = 10,
	OLD_OFF_TRACK_GRACE = 100,
	STUCK_GRACE = 500,
	OLD_HORIZON = 6000,
	NEW_OFF_TRACK_GRACE = 100,
	MAX_EVENTS = 4096,
};

/* Mirrors src/vec_env.c: |progress delta| <= epsilon, whatever the speed. */
#define STUCK_PROGRESS_EPSILON 0.001f
/* Mirrors TMNF_FELL_MARGIN_METERS in src/vec_env.h. */
#define FELL_MARGIN_METERS 16.0f

static void fail(const char *message)
{
	fprintf(stderr, "wr_passthrough: %s\n", message);
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

typedef struct {
	uint32_t length;
	uint32_t start_tick;
	uint32_t longest;
	uint32_t longest_start;
	uint32_t runs_over[4];
} RunTracker;

static void run_update(RunTracker *tracker, int active, uint32_t tick)
{
	static const uint32_t thresholds[4] = {50, 100, 200, 500};
	if (active) {
		if (tracker->length == 0)
			tracker->start_tick = tick;
		tracker->length++;
		for (uint32_t i = 0; i < 4; ++i) {
			if (tracker->length == thresholds[i])
				tracker->runs_over[i]++;
		}
		if (tracker->length > tracker->longest) {
			tracker->longest = tracker->length;
			tracker->longest_start = tracker->start_tick;
		}
	} else {
		tracker->length = 0;
	}
}

static void print_tree(const CPlugTree *tree, int depth)
{
	printf("%*stree flags=0x%x box center=(%.4f,%.4f,%.4f) half=(%.4f,%.4f,%.4f) "
		"iso_t=(%.4f,%.4f,%.4f) children=%u",
		depth * 2, "", tree->flags,
		tree->box.center.x, tree->box.center.y, tree->box.center.z,
		tree->box.half_extent.x, tree->box.half_extent.y,
		tree->box.half_extent.z,
		tree->local_iso.t[0], tree->local_iso.t[1], tree->local_iso.t[2],
		tree->child_count);
	if (tree->surface != NULL && tree->surface->geom != NULL) {
		const GmSurf *geom = tree->surface->geom;
		printf(" surface type=%u", geom->type);
		if (geom->type == GM_SURF_SPHERE)
			printf(" radius=%.4f", ((const GmSurfSphere *)geom)->radius);
		if (geom->type == GM_SURF_ELLIPSOID) {
			const GmSurfEllipsoid *e = (const GmSurfEllipsoid *)geom;
			printf(" radii=(%.4f,%.4f,%.4f)",
				e->radii.x, e->radii.y, e->radii.z);
		}
		if (geom->type == GM_SURF_BOX) {
			const GmSurfBox *b = (const GmSurfBox *)geom;
			printf(" center=(%.4f,%.4f,%.4f) half=(%.4f,%.4f,%.4f)",
				b->center.x, b->center.y, b->center.z,
				b->half_extent.x, b->half_extent.y, b->half_extent.z);
		}
		if (geom->type == GM_SURF_MESH) {
			const GmSurfMesh *m = (const GmSurfMesh *)geom;
			printf(" vertices=%u faces=%u nodes=%u",
				m->vertex_count, m->face_count, m->node_count);
		}
	}
	printf("\n");
	for (uint32_t i = 0; i < tree->child_count; ++i)
		print_tree(tree->children[i], depth + 1);
}

static void run_print(const char *name, const RunTracker *tracker)
{
	printf("%s: longest_run=%u start_tick=%u "
		"runs>=50:%u runs>=100:%u runs>=200:%u runs>=500:%u\n",
		name, tracker->longest, tracker->longest_start,
		tracker->runs_over[0], tracker->runs_over[1],
		tracker->runs_over[2], tracker->runs_over[3]);
}

/* Extreme corridor offsets: horizontal distance in half-widths, height above
 * and below the projected centerline point, each with its tick. */
typedef struct {
	float ratio;
	uint32_t ratio_tick;
	float ratio_lateral;
	float ratio_half_width;
	float above;
	uint32_t above_tick;
	float below;
	uint32_t below_tick;
	uint32_t ticks;
} OffsetExtremes;

static void extremes_update(
	OffsetExtremes *extremes, const TmnfRaceState *race, uint32_t tick)
{
	extremes->ticks++;
	if (race->half_width > 0.0f &&
		race->corridor_lateral / race->half_width > extremes->ratio) {
		extremes->ratio = race->corridor_lateral / race->half_width;
		extremes->ratio_tick = tick;
		extremes->ratio_lateral = race->corridor_lateral;
		extremes->ratio_half_width = race->half_width;
	}
	if (race->corridor_vertical > extremes->above) {
		extremes->above = race->corridor_vertical;
		extremes->above_tick = tick;
	}
	if (-race->corridor_vertical > extremes->below) {
		extremes->below = -race->corridor_vertical;
		extremes->below_tick = tick;
	}
}

static void extremes_print(const char *name, const OffsetExtremes *extremes)
{
	printf("corridor_%s: ticks=%u max_ratio=%.3f (%.2f m / %.2f m) at tick %u "
		"max_above=%.3f at tick %u max_below=%.3f at tick %u\n",
		name, extremes->ticks, extremes->ratio, extremes->ratio_lateral,
		extremes->ratio_half_width, extremes->ratio_tick,
		extremes->above, extremes->above_tick,
		extremes->below, extremes->below_tick);
}

int main(int argc, char **argv)
{
	if (argc < 6) {
		fprintf(stderr,
			"usage: %s TRACK VEHICLE ROUTE INPUTS TRACK_SHA256 "
			"[--csv TICK_CSV] [--expect-finish-ms N]\n",
			argv[0]);
		return 2;
	}
	const char *csv_path = NULL;
	long expected_finish_ms = -1;
	for (int i = 6; i < argc; ++i) {
		if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
			csv_path = argv[++i];
		} else if (strcmp(argv[i], "--expect-finish-ms") == 0 &&
			i + 1 < argc) {
			expected_finish_ms = strtol(argv[++i], NULL, 10);
		} else {
			fail("unknown option");
		}
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
	FILE *csv = NULL;
	if (csv_path != NULL) {
		csv = fopen(csv_path, "w");
		if (csv == NULL)
			fail("cannot open tick CSV");
		fprintf(csv,
			"tick,arc_length,unwrapped_progress,lateral_offset,"
			"half_width,width_violation,wheels_in_contact,"
			"body_collisions,lateral_contact,speed,"
			"materials,x,y,z,corridor_lateral,corridor_vertical,"
			"outside_corridor\n");
	}

	if (getenv("TMNF_PASSTHROUGH_DUMP") != NULL)
		print_tree(corpus->collision_corpus->tree, 0);

	TmnfRaceState race;
	world->route = route;
	TmnfRace_Reset(route, &race, corpus->collision_corpus->live_iso);
	uint64_t contact_mask = race.trigger_contacts;
	const uint32_t checkpoint_count = route->metadata->checkpoint_count;
	uint32_t *entry_counts = calloc(checkpoint_count + 1, sizeof(uint32_t));
	uint32_t entry_order[MAX_EVENTS];
	uint32_t entry_ticks[MAX_EVENTS];
	uint32_t entry_events = 0;
	uint32_t out_of_order_events = 0;
	uint32_t finish_tick = 0;
	uint32_t finish_time_ms = 0;
	uint32_t finish_entries_ignored = 0;
	RunTracker width_grounded = {0};
	RunTracker width_any = {0};
	RunTracker no_wheel_contact = {0};
	RunTracker no_contact_at_all = {0};
	RunTracker width_and_no_wheel = {0};
	RunTracker stuck = {0};
	RunTracker stuck_in_corridor = {0};
	RunTracker outside_corridor = {0};
	OffsetExtremes grounded_extremes = {0};
	OffsetExtremes airborne_extremes = {0};
	uint32_t outside_corridor_ticks = 0;
	uint32_t old_off_track_kill_tick = 0;
	uint32_t stuck_kill_tick = 0;
	uint32_t stuck_in_corridor_kill_tick = 0;
	uint32_t new_off_track_kill_tick = 0;
	uint32_t fell_tick = 0;
	float min_y = 0.0f;
	uint32_t min_y_tick = 0;
	float fell_floor_y;
	{
		uint32_t point_count = TmnfRoute_GetReferencePointCount(route);
		const TmnfRouteReferencePoint *points =
			TmnfRoute_GetReferencePoints(route);
		fell_floor_y = points[0].position.y;
		for (uint32_t i = 1; i < point_count; ++i) {
			if (points[i].position.y < fell_floor_y)
				fell_floor_y = points[i].position.y;
		}
		fell_floor_y -= FELL_MARGIN_METERS;
	}
	uint32_t ground_plane_ticks = 0;
	uint32_t width_violation_grounded_ticks = 0;
	uint32_t width_violation_ticks = 0;
	uint32_t airborne_ticks = 0;
	uint32_t body_collision_ticks = 0;
	float max_offset_ratio = 0.0f;
	uint32_t max_offset_tick = 0;
	uint32_t teleport_like = 0;
	uint32_t progress_regressions = 0;
	float max_progress_drop = 0.0f;
	uint32_t material_mask = 0;
	uint32_t horizon_truncated_before_finish = 0;
	uint8_t car_block[TMNF_CSCENE_VEHICLE_CAR_GAME_SIZE];
	uint8_t wheel_block[TMNF_STADIUM_WHEEL_COUNT *
		TMNF_CSCENE_VEHICLE_CAR_WHEEL_GAME_SIZE];

	for (uint32_t tick = 0; tick < tick_count; ++tick) {
		if (race.finished)
			break;
		if (inputs[tick].respawn != 0) {
			const GmIso4 *spawn = TmnfRace_RespawnLocation(&race);
			if (spawn == NULL)
				fail("respawn before any checkpoint restarts the race");
			World_Respawn(owner, spawn);
			printf("respawn tick=%u race_time_ms=%u\n",
				tick + 1, (tick + 1) * TICK_MS);
		}
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
			&inputs[tick], World_GetPlayerVehicle(owner));
		World_AdvanceTimer(owner, TICK_MS);
		CHmsZoneDynamic_PhysicsStep2(world, TICK_MS);

		const GmIso4 *iso = corpus->collision_corpus->live_iso;
		float dx = iso->t[0] - race.previous_car_transform.t[0];
		float dy = iso->t[1] - race.previous_car_transform.t[1];
		float dz = iso->t[2] - race.previous_car_transform.t[2];
		if (dx * dx + dy * dy + dz * dz >
			(float)(TMNF_RACE_TELEPORT_DISTANCE_METERS *
				TMNF_RACE_TELEPORT_DISTANCE_METERS)) {
			teleport_like++;
		}

		/* Observe trigger entries directly (the step's detection-pass
		 * mask) so the race state machine cannot hide which checkpoint
		 * was crossed or in which order relative to the ghost (route
		 * index order). */
		uint64_t contacts_now = world->trigger_contacts;
		uint64_t entered = contacts_now & ~contact_mask;
		contact_mask = contacts_now;
		for (uint32_t i = 0; i < checkpoint_count; ++i) {
			if ((entered & ((uint64_t)1 << i)) == 0)
				continue;
			if (entry_events < MAX_EVENTS) {
				entry_order[entry_events] = i;
				entry_ticks[entry_events] = tick + 1;
			}
			entry_events++;
			entry_counts[i]++;
			printf("checkpoint_entry: tick=%u race_time=%u "
				"route_index=%u ghost_order_next=%u %s\n",
				tick + 1, (tick + 1) * TICK_MS, i,
				race.visited_count,
				i == race.visited_count ? "ghost_order"
				: "DIFFERENT_ORDER");
			if (i != race.visited_count)
				out_of_order_events++;
		}
		if (entered & ((uint64_t)1 << TMNF_RACE_FINISH_CONTACT_BIT)) {
			printf("finish_entry: tick=%u race_time=%u "
				"visited=%u/%u completed_laps=%u\n",
				tick + 1, (tick + 1) * TICK_MS,
				race.visited_count, checkpoint_count,
				race.completed_laps);
			if (race.visited_count != checkpoint_count)
				finish_entries_ignored++;
		}

		const char *dump = getenv("TMNF_PASSTHROUGH_DUMP");
		if (dump != NULL) {
			unsigned long lo = 0, hi = 0;
			if (sscanf(dump, "%lu-%lu", &lo, &hi) == 2 &&
				tick + 1 >= lo && tick + 1 <= hi) {
				const GmBoxAligned *car_box =
					&corpus->collision_corpus->tree->box;
				GmIso4 inverse;
				GmIso4 car_in_trigger = *iso;
				GmBoxAligned transformed;
				GmMat3 rotation;
				GmVec3 translation = {
					car_in_trigger.t[0], car_in_trigger.t[1],
					car_in_trigger.t[2]};
				GmIso4_SetInverse(&inverse, &route->finish->transform);
				memcpy(rotation.m, car_in_trigger.m, sizeof(rotation.m));
				GmMat3_Mult(&rotation, (const GmMat3 *)&inverse);
				memcpy(car_in_trigger.m, rotation.m, sizeof(rotation.m));
				GmVec3_Mult_Iso4(&translation, &inverse);
				car_in_trigger.t[0] = translation.x;
				car_in_trigger.t[1] = translation.y;
				car_in_trigger.t[2] = translation.z;
				GmBoxAligned_SetMult(&transformed, car_box, &car_in_trigger);
				const CHmsStateDyna *d = World_GetPlayerState(owner);
				printf("dump tick=%u pos=(%.4f,%.4f,%.4f) vel=(%.3f,%.3f,%.3f) "
					"car_box_local center=(%.4f,%.4f,%.4f) half=(%.4f,%.4f,%.4f) "
					"trigger center=(%.4f,%.4f,%.4f) half=(%.4f,%.4f,%.4f) "
					"root_box center=(%.4f,%.4f,%.4f) half=(%.4f,%.4f,%.4f)\n",
					tick + 1, d->pos.x, d->pos.y, d->pos.z,
					d->linVel.x, d->linVel.y, d->linVel.z,
					transformed.center.x, transformed.center.y,
					transformed.center.z, transformed.half_extent.x,
					transformed.half_extent.y, transformed.half_extent.z,
					route->finish->box.center.x, route->finish->box.center.y,
					route->finish->box.center.z,
					route->finish->box.half_extent.x,
					route->finish->box.half_extent.y,
					route->finish->box.half_extent.z,
					car_box->center.x, car_box->center.y, car_box->center.z,
					car_box->half_extent.x, car_box->half_extent.y,
					car_box->half_extent.z);
			}
		}

		float progress_before = race.unwrapped_progress;
		TmnfRaceStepResult result = TmnfRace_Step(
			route, &race, contacts_now, iso);
		if (result.lap_completed)
			printf("lap_completed: tick=%u laps=%u\n",
				tick + 1, race.completed_laps);
		if (result.finished) {
			finish_tick = tick + 1;
			finish_time_ms = result.race_time_ms;
		}
		if (!result.lap_completed &&
			race.unwrapped_progress + 1.0e-3f < progress_before) {
			progress_regressions++;
			float drop = progress_before - race.unwrapped_progress;
			if (drop > max_progress_drop)
				max_progress_drop = drop;
		}

		uint32_t wheels_in_contact = 0;
		uint32_t wheels_on_ground_plane = 0;
		uint32_t materials = 0;
		for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
			const CSceneVehicleCarWheelRealTimeState *state =
				&corpus->vehicle->wheels[i].real_time;
			if (state->has_ground_contact) {
				uint16_t material = (uint16_t)state->contact_material_id;
				wheels_in_contact++;
				if (material < 32) {
					materials |= 1u << material;
					material_mask |= 1u << material;
				}
				if (TmnfRace_IsGroundPlaneMaterial(material))
					wheels_on_ground_plane++;
			}
		}
		const CFastBuffer_SHmsPhysicalCollision *collisions =
			World_GetLastCollisions(owner);
		World_WritePlayerGameState(owner, car_block, wheel_block);
		int32_t lateral_contact;
		memcpy(&lateral_contact, car_block + 0x5dc, 4);
		int grounded = wheels_in_contact > 0;
		int width_violation = !race.finished &&
			race.lateral_offset > race.half_width;
		const CHmsStateDyna *dyna = World_GetPlayerState(owner);
		float speed = sqrtf(
			dyna->linVel.x * dyna->linVel.x +
			dyna->linVel.y * dyna->linVel.y +
			dyna->linVel.z * dyna->linVel.z);

		if (width_violation) {
			width_violation_ticks++;
			if (grounded)
				width_violation_grounded_ticks++;
		}
		if (!grounded)
			airborne_ticks++;
		if (collisions->count != 0)
			body_collision_ticks++;
		if (race.half_width > 0.0f &&
			race.lateral_offset / race.half_width > max_offset_ratio) {
			max_offset_ratio = race.lateral_offset / race.half_width;
			max_offset_tick = tick + 1;
		}

		run_update(&width_grounded, width_violation && grounded, tick + 1);
		run_update(&width_any, width_violation, tick + 1);
		run_update(&no_wheel_contact, !grounded, tick + 1);
		run_update(&no_contact_at_all,
			!grounded && collisions->count == 0, tick + 1);
		run_update(&width_and_no_wheel,
			width_violation && !grounded, tick + 1);
		if (old_off_track_kill_tick == 0 &&
			width_grounded.length >= OLD_OFF_TRACK_GRACE)
			old_off_track_kill_tick = tick + 1;

		float delta = fabsf(race.unwrapped_progress - race.previous_progress);
		run_update(&stuck, delta <= STUCK_PROGRESS_EPSILON, tick + 1);
		if (stuck_kill_tick == 0 && stuck.length >= STUCK_GRACE)
			stuck_kill_tick = tick + 1;
		/* Frozen progress inside the corridor is the rule acting on the
		 * line itself. Outside the corridor progress is frozen by
		 * TmnfRace_Step; a long run there on a record line means the
		 * route reference does not cover the line (A08 before its
		 * ghost-first regeneration: north road against a south
		 * out-and-back reference). */
		run_update(&stuck_in_corridor,
			delta <= STUCK_PROGRESS_EPSILON && !race.outside_corridor,
			tick + 1);
		if (stuck_in_corridor_kill_tick == 0 &&
			stuck_in_corridor.length >= STUCK_GRACE)
			stuck_in_corridor_kill_tick = tick + 1;
		run_update(&outside_corridor, race.outside_corridor != 0, tick + 1);
		if (race.outside_corridor)
			outside_corridor_ticks++;
		if (!race.finished)
			extremes_update(
				grounded ? &grounded_extremes : &airborne_extremes,
				&race, tick + 1);
		if (tick == 0 || dyna->pos.y < min_y) {
			min_y = dyna->pos.y;
			min_y_tick = tick + 1;
		}
		if (fell_tick == 0 && dyna->pos.y < fell_floor_y)
			fell_tick = tick + 1;
		if (wheels_in_contact != 0 && wheels_on_ground_plane == wheels_in_contact)
			ground_plane_ticks++;
		if (TmnfRace_UpdateOffTrack(&race, NEW_OFF_TRACK_GRACE,
				wheels_in_contact, wheels_on_ground_plane) &&
			new_off_track_kill_tick == 0)
			new_off_track_kill_tick = tick + 1;

		if (csv != NULL) {
			fprintf(csv,
				"%u,%.3f,%.3f,%.3f,%.3f,%d,%d,%u,%d,%.3f,0x%x,"
				"%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
				tick + 1, race.arc_length, race.unwrapped_progress,
				race.lateral_offset, race.half_width,
				width_violation, wheels_in_contact,
				collisions->count, lateral_contact, speed,
				materials, dyna->pos.x, dyna->pos.y, dyna->pos.z,
				race.corridor_lateral, race.corridor_vertical,
				race.outside_corridor);
		}
	}
	if (finish_tick == 0 || finish_tick > OLD_HORIZON)
		horizon_truncated_before_finish = 1;

	printf("ticks_simulated: %u\n",
		finish_tick != 0 ? finish_tick : tick_count);
	printf("route_length: %.3f laps=%u checkpoints=%u\n",
		TmnfRoute_GetReferenceLength(route),
		route->metadata->lap_count, checkpoint_count);
	printf("finish: %s tick=%u race_time_ms=%u\n",
		finish_tick != 0 ? "yes" : "no", finish_tick, finish_time_ms);
	printf("checkpoint_entry_events: %u different_from_ghost_order=%u "
		"finish_entries_before_all_checkpoints=%u\n",
		entry_events, out_of_order_events, finish_entries_ignored);
	printf("checkpoint_entry_sequence:");
	for (uint32_t i = 0; i < entry_events && i < MAX_EVENTS; ++i)
		printf(" %u@%u", entry_order[i], entry_ticks[i]);
	printf("\n");
	printf("checkpoint_entry_counts:");
	for (uint32_t i = 0; i < checkpoint_count; ++i)
		printf(" %u", entry_counts[i]);
	printf("\n");
	printf("old_horizon_6000_truncates_before_finish: %u\n",
		horizon_truncated_before_finish);
	printf("old_off_track_rule_kill_tick: %u\n", old_off_track_kill_tick);
	printf("stuck_rule_kill_tick: %u in_corridor_kill_tick: %u "
		"(progress-only, %u ticks)\n",
		stuck_kill_tick, stuck_in_corridor_kill_tick, STUCK_GRACE);
	printf("corridor_rule: width_factor=%.2f width_floor=%.2f above=%.2f "
		"below=%.2f frozen_progress_ticks=%u\n",
		(double)TMNF_RACE_CORRIDOR_WIDTH_FACTOR,
		(double)TMNF_RACE_CORRIDOR_WIDTH_FLOOR_METERS,
		(double)TMNF_RACE_CORRIDOR_ABOVE_METERS,
		(double)TMNF_RACE_CORRIDOR_BELOW_METERS,
		outside_corridor_ticks);
	extremes_print("grounded", &grounded_extremes);
	extremes_print("airborne", &airborne_extremes);
	printf("fell_rule: floor_y=%.3f min_y=%.3f at tick %u kill_tick=%u\n",
		fell_floor_y, min_y, min_y_tick, fell_tick);
	printf("ground_plane_contact_ticks: %u new_off_track_rule_kill_tick: %u\n",
		ground_plane_ticks, new_off_track_kill_tick);
	printf("width_violation_ticks: total=%u grounded=%u\n",
		width_violation_ticks, width_violation_grounded_ticks);
	printf("airborne_ticks: %u body_collision_ticks: %u\n",
		airborne_ticks, body_collision_ticks);
	printf("max_offset_ratio: %.3f at tick %u\n",
		max_offset_ratio, max_offset_tick);
	printf("teleport_like_displacements: %u\n", teleport_like);
	printf("progress_regressions: %u max_drop=%.3f\n",
		progress_regressions, max_progress_drop);
	printf("wheel_materials_mask: 0x%x\n", material_mask);
	run_print("width_violation_grounded", &width_grounded);
	run_print("width_violation_any", &width_any);
	run_print("no_wheel_contact", &no_wheel_contact);
	run_print("no_wheel_contact_and_no_body_collision", &no_contact_at_all);
	run_print("width_violation_airborne", &width_and_no_wheel);
	run_print("stuck", &stuck);
	run_print("stuck_in_corridor", &stuck_in_corridor);
	run_print("outside_corridor", &outside_corridor);

	if (csv != NULL)
		fclose(csv);
	free(entry_counts);
	TmnfRoute_Unload(route);
	World_Destroy(owner);
	TmnfTrack_Unload(track);
	free(inputs);
	if (expected_finish_ms >= 0 &&
		(finish_tick == 0 || (long)finish_time_ms != expected_finish_ms)) {
		fprintf(stderr,
			"wr_passthrough: env finish %u ms differs from game %ld ms\n",
			finish_time_ms, expected_finish_ms);
		return 1;
	}
	if (expected_finish_ms >= 0 && (new_off_track_kill_tick != 0 ||
		stuck_kill_tick != 0 || fell_tick != 0)) {
		fprintf(stderr,
			"wr_passthrough: a rule would have ended the record line "
			"(off-track tick %u, stuck tick %u [in corridor %u, "
			"frozen outside the corridor %u ticks, longest run %u from "
			"tick %u], fell tick %u)\n",
			new_off_track_kill_tick, stuck_kill_tick,
			stuck_in_corridor_kill_tick, outside_corridor_ticks,
			outside_corridor.longest, outside_corridor.longest_start,
			fell_tick);
		return 1;
	}
	return 0;
}
