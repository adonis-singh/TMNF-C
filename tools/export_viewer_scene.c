#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "route.h"
#include "world.h"

enum {
	TICK_MS = 10,
	MATERIAL_COUNT = TMNF_TRACK_MATERIAL_COUNT,
};

static const GmBoxAligned CAR_LOCAL_BOX = {
	.center = { 1.01327896e-06f, 0.502316654f, 0.0937180519f },
	.half_extent = { 1.0670011f, 0.713816702f, 2.05237103f },
};

static const GmVec3 WHEEL_OFFSETS[4] = {
	{ 0.863012016f, 0.152499989f, 1.78208899f },
	{ -0.862990022f, 0.152499989f, 1.78208899f },
	{ -0.88499999f, 0.152503982f, -1.20550203f },
	{ 0.885002017f, 0.152503982f, -1.20550203f },
};

typedef struct {
	uint32_t *data;
	uint32_t count;
} IndexBuffer;

typedef struct {
	float *positions;
	uint32_t vertex_count;
	uint32_t triangle_count;
	IndexBuffer indices[MATERIAL_COUNT];
	GmVec3 minimum;
	GmVec3 maximum;
} TrackScene;

static void fail(const char *format, ...)
{
	va_list arguments;
	fprintf(stderr, "export_viewer_scene: ");
	va_start(arguments, format);
	vfprintf(stderr, format, arguments);
	va_end(arguments);
	fputc('\n', stderr);
	exit(2);
}

static uint8_t hex_nibble(char value)
{
	if ('0' <= value && value <= '9')
		return (uint8_t)(value - '0');
	if ('a' <= value && value <= 'f')
		return (uint8_t)(value - 'a' + 10);
	if ('A' <= value && value <= 'F')
		return (uint8_t)(value - 'A' + 10);
	fail("track SHA-256 must contain exactly 64 hexadecimal characters");
	return 0;
}

static void parse_sha256(const char *text, uint8_t output[32])
{
	if (strlen(text) != 64)
		fail("track SHA-256 must contain exactly 64 hexadecimal characters");
	for (uint32_t i = 0; i < 32; ++i) {
		output[i] = (uint8_t)(
			(hex_nibble(text[i * 2]) << 4) |
			hex_nibble(text[i * 2 + 1]));
	}
}

static void validate_game_visual_scene(const char *scene)
{
	if (*scene == '\0')
		fail("game visual scene must not be empty");
	for (const char *cursor = scene; *cursor != '\0'; ++cursor) {
		if ((*cursor < 'a' || *cursor > 'z') &&
			(*cursor < '0' || *cursor > '9') &&
			*cursor != '_' && *cursor != '-')
			fail("game visual scene must use lowercase ASCII identifiers");
	}
}

static void *allocate_array(size_t count, size_t size)
{
	if (size != 0 && count > SIZE_MAX / size)
		fail("allocation size overflow");
	if (count == 0)
		return NULL;
	void *memory = malloc(count * size);
	if (memory == NULL)
		fail("out of memory");
	return memory;
}

static TMNFRaceInputs *read_inputs(const char *path, uint32_t *tick_count)
{
	FILE *file = fopen(path, "rb");
	if (file == NULL)
		fail("cannot open input schedule %s", path);
	if (fseek(file, 0, SEEK_END) != 0)
		fail("cannot seek input schedule");
	long size = ftell(file);
	if (size <= 0 || size % (long)sizeof(TMNFRaceInputs) != 0)
		fail("input schedule is not a whole number of records");
	uint64_t count = (uint64_t)size / sizeof(TMNFRaceInputs);
	if (count > UINT32_MAX)
		fail("input schedule has too many records");
	rewind(file);
	TMNFRaceInputs *inputs = allocate_array((size_t)count, sizeof(*inputs));
	if (fread(inputs, sizeof(*inputs), (size_t)count, file) != count)
		fail("input schedule is truncated");
	if (fclose(file) != 0)
		fail("cannot close input schedule");
	*tick_count = (uint32_t)count;
	return inputs;
}

static uint32_t face_material(
	const CPlugSurface *surface, const GmSurfMeshFace *face)
{
	if (face->material_index >= surface->material_count)
		fail("track face material is outside its surface remap");
	uint32_t material = surface->material_ids[face->material_index];
	if (material >= MATERIAL_COUNT)
		fail("track face uses material %u, maximum is %u",
			material, MATERIAL_COUNT - 1);
	return material;
}

static int active_entry(const HmsStaticCollisionEntry *entry)
{
	return (entry->tree_flags & 0x80u) != 0;
}

static const GmSurfMesh *entry_mesh(
	const HmsStaticCollisionEntry *entry)
{
	if (entry->surface == NULL || entry->surface->geom == NULL ||
		entry->surface->geom->type != GM_SURF_MESH)
		fail("active track entry is not a triangle mesh");
	return (const GmSurfMesh *)entry->surface->geom;
}

static TrackScene gather_track(const TmnfTrack *track)
{
	TrackScene scene = {
		.minimum = { INFINITY, INFINITY, INFINITY },
		.maximum = { -INFINITY, -INFINITY, -INFINITY },
	};
	uint64_t vertex_count = 0;
	uint64_t triangle_count = 0;
	uint64_t material_index_counts[MATERIAL_COUNT] = { 0 };

	for (uint32_t i = 0; i < track->entry_count; ++i) {
		const HmsStaticCollisionEntry *entry = &track->entries[i];
		if (!active_entry(entry))
			continue;
		const GmSurfMesh *mesh = entry_mesh(entry);
		vertex_count += mesh->vertex_count;
		triangle_count += mesh->face_count;
		if (vertex_count > UINT32_MAX || triangle_count > UINT32_MAX)
			fail("track geometry exceeds 32-bit viewer limits");
		for (uint32_t face = 0; face < mesh->face_count; ++face) {
			uint32_t material =
				face_material(entry->surface, &mesh->faces[face]);
			material_index_counts[material] += 3;
			if (material_index_counts[material] > UINT32_MAX)
				fail("track material index buffer is too large");
		}
	}
	if (vertex_count == 0 || triangle_count == 0)
		fail("track has no active triangle geometry");

	scene.vertex_count = (uint32_t)vertex_count;
	scene.triangle_count = (uint32_t)triangle_count;
	scene.positions =
		allocate_array((size_t)scene.vertex_count * 3, sizeof(float));
	for (uint32_t material = 0; material < MATERIAL_COUNT; ++material) {
		scene.indices[material].data = allocate_array(
			(size_t)material_index_counts[material], sizeof(uint32_t));
	}

	uint32_t vertex_base = 0;
	uint32_t material_offsets[MATERIAL_COUNT] = { 0 };
	for (uint32_t i = 0; i < track->entry_count; ++i) {
		const HmsStaticCollisionEntry *entry = &track->entries[i];
		if (!active_entry(entry))
			continue;
		const GmSurfMesh *mesh = entry_mesh(entry);
		for (uint32_t vertex = 0; vertex < mesh->vertex_count; ++vertex) {
			GmVec3 world;
			GmVec3_SetMult_Iso4(
				&world, &mesh->vertices[vertex], &entry->iso);
			if (!isfinite(world.x) || !isfinite(world.y) ||
				!isfinite(world.z))
				fail("track transform produced a non-finite vertex");
			float *output =
				scene.positions + (size_t)(vertex_base + vertex) * 3;
			output[0] = world.x;
			output[1] = world.y;
			output[2] = world.z;
			if (world.x < scene.minimum.x) scene.minimum.x = world.x;
			if (world.y < scene.minimum.y) scene.minimum.y = world.y;
			if (world.z < scene.minimum.z) scene.minimum.z = world.z;
			if (world.x > scene.maximum.x) scene.maximum.x = world.x;
			if (world.y > scene.maximum.y) scene.maximum.y = world.y;
			if (world.z > scene.maximum.z) scene.maximum.z = world.z;
		}
		for (uint32_t face = 0; face < mesh->face_count; ++face) {
			const GmSurfMeshFace *mesh_face = &mesh->faces[face];
			uint32_t material =
				face_material(entry->surface, mesh_face);
			uint32_t *offset = &material_offsets[material];
			for (uint32_t vertex = 0; vertex < 3; ++vertex) {
				scene.indices[material].data[(*offset)++] =
					vertex_base + mesh_face->vertex[vertex];
			}
		}
		vertex_base += mesh->vertex_count;
	}
	if (vertex_base != scene.vertex_count)
		fail("track vertex accounting mismatch");
	for (uint32_t material = 0; material < MATERIAL_COUNT; ++material) {
		scene.indices[material].count = material_offsets[material];
		if (material_offsets[material] != material_index_counts[material])
			fail("track material index accounting mismatch");
	}
	return scene;
}

static void free_track_scene(TrackScene *scene)
{
	free(scene->positions);
	for (uint32_t material = 0; material < MATERIAL_COUNT; ++material)
		free(scene->indices[material].data);
}

static void write_float(FILE *output, float value)
{
	if (!isfinite(value))
		fail("refusing to write a non-finite number");
	fprintf(output, "%.9g", value);
}

static void write_vec3(FILE *output, const GmVec3 *vector)
{
	fputc('[', output);
	write_float(output, vector->x);
	fputc(',', output);
	write_float(output, vector->y);
	fputc(',', output);
	write_float(output, vector->z);
	fputc(']', output);
}

static void write_iso(FILE *output, const GmIso4 *transform)
{
	fputs("{\"rotation\":[", output);
	for (uint32_t i = 0; i < 9; ++i) {
		if (i != 0) fputc(',', output);
		write_float(output, transform->m[i]);
	}
	fputs("],\"translation\":[", output);
	for (uint32_t i = 0; i < 3; ++i) {
		if (i != 0) fputc(',', output);
		write_float(output, transform->t[i]);
	}
	fputs("]}", output);
}

static void write_track(
	FILE *output, const TmnfTrack *track, const TrackScene *scene)
{
	fprintf(output,
		"\"track\":{\"vertexCount\":%u,\"triangleCount\":%u,\"bounds\":[",
		scene->vertex_count, scene->triangle_count);
	write_vec3(output, &scene->minimum);
	fputc(',', output);
	write_vec3(output, &scene->maximum);
	fputs("],\"materials\":[", output);
	for (uint32_t material = 0; material < MATERIAL_COUNT; ++material) {
		if (material != 0) fputc(',', output);
		fprintf(output, "{\"id\":%u,\"friction\":", material);
		write_float(output, track->materials[material].friction);
		fputs(",\"restitution\":", output);
		write_float(output, track->materials[material].restitution);
		fputc('}', output);
	}
	fputs("],\"positions\":[", output);
	for (uint32_t i = 0; i < scene->vertex_count * 3; ++i) {
		if (i != 0) fputc(',', output);
		write_float(output, scene->positions[i]);
	}
	fputs("],\"indices\":[", output);
	uint32_t written = 0;
	for (uint32_t material = 0; material < MATERIAL_COUNT; ++material) {
		for (uint32_t i = 0; i < scene->indices[material].count; ++i) {
			if (written++ != 0) fputc(',', output);
			fprintf(output, "%u", scene->indices[material].data[i]);
		}
	}
	fputs("],\"groups\":[", output);
	uint32_t group_start = 0;
	int first = 1;
	for (uint32_t material = 0; material < MATERIAL_COUNT; ++material) {
		uint32_t count = scene->indices[material].count;
		if (count == 0)
			continue;
		if (!first) fputc(',', output);
		first = 0;
		fprintf(output, "[%u,%u,%u]", group_start, count, material);
		group_start += count;
	}
	if (group_start != scene->triangle_count * 3)
		fail("track group accounting mismatch");
	fputs("]}", output);
}

static void write_trigger(
	FILE *output, const TmnfRouteTrigger *trigger)
{
	fprintf(output,
		"{\"raceIndex\":%u,\"blockIndex\":%u,\"box\":{\"center\":",
		trigger->race_index, trigger->block_index);
	write_vec3(output, &trigger->box.center);
	fputs(",\"halfExtent\":", output);
	write_vec3(output, &trigger->box.half_extent);
	fputs("},\"transform\":", output);
	write_iso(output, &trigger->transform);
	fputc('}', output);
}

static void write_route(FILE *output, const TmnfRoute *route)
{
	uint32_t point_count = TmnfRoute_GetReferencePointCount(route);
	const TmnfRouteReferencePoint *points =
		TmnfRoute_GetReferencePoints(route);
	fprintf(output, ",\"route\":{\"length\":");
	write_float(output, TmnfRoute_GetReferenceLength(route));
	fprintf(output, ",\"centerlineCount\":%u,\"centerline\":[", point_count);
	for (uint32_t i = 0; i < point_count; ++i) {
		if (i != 0) fputc(',', output);
		fputc('[', output);
		write_float(output, points[i].position.x);
		fputc(',', output);
		write_float(output, points[i].position.y);
		fputc(',', output);
		write_float(output, points[i].position.z);
		fputc(',', output);
		write_float(output, points[i].half_width);
		fputc(']', output);
	}
	uint32_t checkpoint_count = TmnfRoute_GetCheckpointCount(route);
	fprintf(output, "],\"checkpoints\":[");
	for (uint32_t i = 0; i < checkpoint_count; ++i) {
		if (i != 0) fputc(',', output);
		write_trigger(output, TmnfRoute_GetCheckpoint(route, i));
	}
	fputs("],\"finish\":", output);
	write_trigger(output, TmnfRoute_GetFinish(route));
	fputc('}', output);
}

static void write_car(FILE *output)
{
	fputs(",\"car\":{\"bodyCenter\":", output);
	write_vec3(output, &CAR_LOCAL_BOX.center);
	fputs(",\"bodyHalfExtent\":", output);
	write_vec3(output, &CAR_LOCAL_BOX.half_extent);
	fputs(",\"wheelOffsets\":[", output);
	for (uint32_t i = 0; i < 4; ++i) {
		if (i != 0) fputc(',', output);
		write_vec3(output, &WHEEL_OFFSETS[i]);
	}
	fputs("],\"wheelRadii\":[0.181999996,0.363999993,0.363999993]}", output);
}

static GmIso4 state_transform(const CHmsStateDyna *state)
{
	GmIso4 transform;
	memcpy(transform.m, state->rot.m, sizeof(transform.m));
	transform.t[0] = state->pos.x;
	transform.t[1] = state->pos.y;
	transform.t[2] = state->pos.z;
	return transform;
}

static void write_lap_summary(
	FILE *output, uint32_t tick_count, uint32_t finish_time,
	const uint32_t *checkpoint_ticks, uint32_t checkpoint_count,
	uint32_t finish_tick)
{
	fprintf(output,
		"\"tickCount\":%u,\"finishTimeMs\":%u,\"checkpointTicks\":[",
		tick_count, finish_time);
	for (uint32_t i = 0; i < checkpoint_count; ++i) {
		if (i != 0) fputc(',', output);
		if (checkpoint_ticks[i] == UINT32_MAX) fputs("null", output);
		else fprintf(output, "%u", checkpoint_ticks[i]);
	}
	fputs("],\"finishTick\":", output);
	if (finish_tick == UINT32_MAX) fputs("null", output);
	else fprintf(output, "%u", finish_tick);
	fputc('}', output);
}

static uint32_t write_lap(
	FILE *output, TmnfWorld *world, const TmnfRoute *route,
	const TMNFRaceInputs *inputs, uint32_t tick_count)
{
	const CHmsStateDyna *initial = World_GetPlayerState(world);
	GmIso4 initial_transform = state_transform(initial);
	TmnfRaceState race;
	TmnfPhysicsWorld *physics = World_GetPhysicsWorld(world);
	physics->route = route;
	TmnfRace_Reset(route, &race, &initial_transform);
	const uint32_t checkpoint_count = TmnfRoute_GetCheckpointCount(route);
	uint32_t *checkpoint_ticks =
		allocate_array(checkpoint_count, sizeof(*checkpoint_ticks));
	for (uint32_t i = 0; i < checkpoint_count; ++i)
		checkpoint_ticks[i] = UINT32_MAX;
	uint32_t finish_tick = UINT32_MAX;

	fputs(",\"lap\":{\"tickMs\":10,\"fields\":["
		"\"raceTimeMs\",\"x\",\"y\",\"z\",\"qx\",\"qy\",\"qz\",\"qw\","
		"\"speedMps\",\"rpm\",\"gear\",\"contactMask\",\"slidingMask\","
		"\"inputSteer\","
		"\"wheelSteerFL\",\"wheelSteerFR\",\"wheelSteerRL\",\"wheelSteerRR\","
		"\"wheelDamperFL\",\"wheelDamperFR\",\"wheelDamperRL\",\"wheelDamperRR\","
		"\"wheelSpeedFL\",\"wheelSpeedFR\",\"wheelSpeedRL\",\"wheelSpeedRR\"],"
		"\"ticks\":[", output);
	for (uint32_t tick = 0; tick < tick_count; ++tick) {
		CSceneVehicleCar *vehicle = World_GetPlayerVehicle(world);
		if (vehicle->wheel_count != 4)
			fail("viewer scenes require exactly four wheels");
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
			&inputs[tick], vehicle);
		const CHmsStateDyna *state = World_GetPlayerState(world);
		GmIso4 transform = state_transform(state);
		/* Replay tick 1 is the spawn state before the first integration;
		 * the mask is the previous step's detection passes. */
		if (tick != 0 && !race.finished) {
			TmnfRaceStepResult step = TmnfRace_Step(
				route, &race, physics->trigger_contacts, &transform);
			if (step.checkpoint_accepted &&
				checkpoint_ticks[step.checkpoint_index] == UINT32_MAX)
				checkpoint_ticks[step.checkpoint_index] = tick;
			if (step.finished)
				finish_tick = tick;
		}
		uint32_t race_time = race.finished
			? race.finish_time_ms : (tick + 1) * TICK_MS;
		uint32_t contact_mask = 0;
		uint32_t sliding_mask = 0;
		for (uint32_t wheel = 0; wheel < 4; ++wheel) {
			if (vehicle->wheels[wheel].real_time.has_ground_contact != 0)
				contact_mask |= 1u << wheel;
			if (vehicle->wheels[wheel].real_time.is_sliding != 0)
				sliding_mask |= 1u << wheel;
		}
		float speed = sqrtf(
			state->linVel.x * state->linVel.x +
			state->linVel.y * state->linVel.y +
			state->linVel.z * state->linVel.z);
		if (tick != 0) fputc(',', output);
		fprintf(output, "[%u,", race_time);
		write_float(output, state->pos.x);
		fputc(',', output);
		write_float(output, state->pos.y);
		fputc(',', output);
		write_float(output, state->pos.z);
		fputc(',', output);
		write_float(output, state->quat.x);
		fputc(',', output);
		write_float(output, state->quat.y);
		fputc(',', output);
		write_float(output, state->quat.z);
		fputc(',', output);
		write_float(output, state->quat.w);
		fputc(',', output);
		write_float(output, speed);
		fputc(',', output);
		write_float(output, vehicle->engine.rpm);
		fprintf(output, ",%d,%u,%u,", vehicle->engine.gear,
			contact_mask, sliding_mask);
		write_float(output, vehicle->input_steer);
		for (uint32_t wheel = 0; wheel < 4; ++wheel) {
			fputc(',', output);
			write_float(
				output, vehicle->wheels[wheel].real_time.blend_value);
		}
		for (uint32_t wheel = 0; wheel < 4; ++wheel) {
			fputc(',', output);
			write_float(
				output, vehicle->wheels[wheel].real_time.damper_absorb);
		}
		for (uint32_t wheel = 0; wheel < 4; ++wheel) {
			fputc(',', output);
			write_float(output, vehicle->wheels[wheel].real_time.field6c);
		}
		fputc(']', output);

		World_AdvanceTimer(world, TICK_MS);
		CHmsZoneDynamic_PhysicsStep2(physics, TICK_MS);
	}
	fputs("],", output);
	write_lap_summary(output, tick_count, race.finish_time_ms,
		checkpoint_ticks, checkpoint_count, finish_tick);
	/* Callers need four lap fields, not another parse of a potentially
	 * hundred-megabyte track mesh. The scene bytes remain unchanged. */
	fputs("lap_summary {", stdout);
	write_lap_summary(stdout, tick_count, race.finish_time_ms,
		checkpoint_ticks, checkpoint_count, finish_tick);
	fputc('\n', stdout);
	free(checkpoint_ticks);
	return race.finish_time_ms;
}

int main(int argc, char **argv)
{
	if (argc != 8) {
		fprintf(stderr,
			"usage: %s TRACK ROUTE VEHICLE INPUTS TRACK_SHA256 "
			"GAME_VISUAL_SCENE out.json\n",
			argv[0]);
		return 2;
	}

	uint8_t track_sha256[32];
	parse_sha256(argv[5], track_sha256);
	validate_game_visual_scene(argv[6]);
	uint32_t tick_count;
	TMNFRaceInputs *inputs = read_inputs(argv[4], &tick_count);
	TmnfTrack *track = TmnfTrack_Load(argv[1], track_sha256);
	TmnfRoute *route = TmnfRoute_Load(argv[2], track_sha256);
	TmnfWorld *world = World_Create(track, argv[3]);
	TrackScene track_scene = gather_track(track);

	size_t temporary_size = strlen(argv[7]) + sizeof(".tmp");
	char *temporary = allocate_array(temporary_size, 1);
	snprintf(temporary, temporary_size, "%s.tmp", argv[7]);
	FILE *output = fopen(temporary, "wb");
	if (output == NULL)
		fail("cannot open output %s: %s", temporary, strerror(errno));

	fprintf(output,
		"{\"format\":\"tmnf-c-viewer-scene\",\"version\":3,"
		"\"gameVisuals\":{\"manifest\":\"assets/game/manifest.json\","
		"\"scene\":\"%s\"},",
		argv[6]);
	write_track(output, track, &track_scene);
	write_route(output, route);
	write_car(output);
	uint32_t finish_time =
		write_lap(output, world, route, inputs, tick_count);
	fputs("}\n", output);
	if (ferror(output) || fclose(output) != 0)
		fail("cannot write output %s", temporary);
	if (rename(temporary, argv[7]) != 0)
		fail("cannot replace output %s: %s", argv[7], strerror(errno));

	printf(
		"exported %u triangles, %u route points, %u ticks, "
		"finish %u ms to %s\n",
		track_scene.triangle_count,
		TmnfRoute_GetReferencePointCount(route),
		tick_count, finish_time, argv[7]);

	free(temporary);
	free_track_scene(&track_scene);
	World_Destroy(world);
	TmnfRoute_Unload(route);
	TmnfTrack_Unload(track);
	free(inputs);
	return 0;
}
