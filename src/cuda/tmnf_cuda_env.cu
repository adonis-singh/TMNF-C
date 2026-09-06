/* CUDA vectorised environment. See tmnf_cuda_env.h and docs/CUDA.md.
 *
 * The per-environment RL logic below mirrors src/vec_env.c function for
 * function (capture_physics, restore_physics, write_rl_observation,
 * failure_reason, tick_reward, step_*_range). tests/cuda_lockstep.c holds the
 * two implementations to byte equality every tick; keep them in step.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <gnu/libc-version.h>
#include <setjmp.h>

#include "tmnf_cuda_env.h"
#include "gm.h"
#include "vehicle_fake_contact_mask.h"
#include "vehicle_water_tuning.h"
#include "vehicle_respawn.h"
#include "tmnf_warp.h"

/* Per-thread stack. After device LTO every kernel is one function (the
 * world graph's function pointers resolve to the only targets), so
 * `cuobjdump -res-usage` reports the whole frame: 7,624 bytes for
 * reset_kernel, 7,536 for the step kernels. The driver reserves this much
 * local memory for every resident thread of the GPU (2048 x 170 SMs): 16 KB
 * cost 4.4 GB per process and let six concurrent ctest processes exhaust
 * the card. A frame above the limit fails the launch, it cannot silently
 * corrupt. */
/* Default cudaLimitStackSize. tmnf_kernel's frame is 14,160 B (cuobjdump
 * -res-usage, TMNF_WORLD_MAX_BYTES of it the local world); with the frames
 * of its indirect callees the lockstep runs at 14,592 B and faults at
 * 14,336 B. 16 KB leaves 1.8 KB of margin. */
enum { TMNF_CUDA_STACK_BYTES = 16384 };

/* ------------------------------------------------------------------------ */
/* Host plumbing                                                             */
/* ------------------------------------------------------------------------ */

/* Inside TmnfCudaVecEnv_Create a device error unwinds to the create call,
 * which frees what it built and returns NULL with the message in
 * last_error; everywhere else it aborts. */
static jmp_buf *create_unwind;
static char last_error[256];

const char *TmnfCudaVecEnv_LastError(void)
{
	return last_error;
}

static void cuda_fail(const char *message)
{
	fprintf(stderr, "tmnf cuda env: %s\n", message);
	abort();
}

static void check(cudaError_t error, const char *what)
{
	if (error == cudaSuccess)
		return;
	snprintf(last_error, sizeof(last_error), "%s: %s", what,
		cudaGetErrorString(error));
	if (create_unwind != NULL) {
		(void)cudaGetLastError();
		longjmp(*create_unwind, 1);
	}
	fprintf(stderr, "tmnf cuda env: %s\n", last_error);
	abort();
}

static void *device_alloc(size_t bytes)
{
	void *memory = NULL;
	check(cudaMalloc(&memory, bytes == 0 ? 1 : bytes), "cudaMalloc");
	check(cudaMemset(memory, 0, bytes == 0 ? 1 : bytes), "cudaMemset");
	return memory;
}

static void *device_upload(const void *source, size_t bytes)
{
	void *memory = device_alloc(bytes);
	check(cudaMemcpy(memory, source, bytes, cudaMemcpyHostToDevice),
		"cudaMemcpy upload");
	return memory;
}

/* ------------------------------------------------------------------------ */
/* Immutable device data                                                      */
/* ------------------------------------------------------------------------ */

/* Rebases a host pointer into the track mapping onto the device copy. */
static uint64_t rebase(
	uint64_t pointer, const void *host_base, size_t size,
	const void *device_base)
{
	uint64_t offset = pointer - (uint64_t)(uintptr_t)host_base;
	if (pointer == 0)
		cuda_fail("null pointer inside an immutable mapping");
	if (offset >= size)
		cuda_fail("pointer leaves its immutable mapping");
	return (uint64_t)(uintptr_t)device_base + offset;
}

typedef struct {
	uint8_t *mapping;
	TmnfTrack *track;
	/* Heap-built parts of the host track (track.c): the static grid, the
	 * per-mesh grids and the static response bodies. */
	void **blocks;
	uint32_t block_count;
} DeviceTrack;

static void track_block(DeviceTrack *result, void *block)
{
	result->blocks = (void **)realloc(
		result->blocks, (result->block_count + 1) * sizeof(void *));
	if (result->blocks == NULL)
		cuda_fail("out of memory");
	result->blocks[result->block_count++] = block;
}

static void *track_upload(DeviceTrack *result, const void *source, size_t bytes)
{
	void *block = device_upload(source, bytes);
	track_block(result, block);
	return block;
}

/* The static grid is a tree of host arrays; every array gets a device copy
 * and the structs that point at them are rewritten with device addresses
 * before they are uploaded themselves. Empty grids (cell_count == 0) stay
 * empty, which makes CHmsCollisionManager scan the entries as the host does. */
static void upload_static_grid(DeviceTrack *result, TmnfTrack *shadow,
	const TmnfTrack *track)
{
	const TmnfStaticGrid *grid = &track->grid;
	shadow->grid = *grid;
	shadow->grid_node_count = track->grid_node_count;
	shadow->grid_mesh_slot_count = track->grid_mesh_slot_count;
	if (grid->cell_count == 0)
		return;
	shadow->grid.cell_offsets = (const uint32_t *)track_upload(result,
		grid->cell_offsets, (size_t)grid->cell_count * sizeof(uint32_t));
	shadow->grid.cell_counts = (const uint32_t *)track_upload(result,
		grid->cell_counts, (size_t)grid->cell_count * sizeof(uint32_t));
	shadow->grid.nodes = (const TmnfStaticCellNode *)track_upload(result,
		grid->nodes, track->grid_node_count * sizeof(TmnfStaticCellNode));
	shadow->grid.mesh_pool = (const TmnfMeshPoolSlot *)track_upload(result,
		grid->mesh_pool,
		track->grid_mesh_slot_count * sizeof(TmnfMeshPoolSlot));
	shadow->grid.entry_inverse_isos = (const GmIso4 *)track_upload(result,
		grid->entry_inverse_isos,
		(size_t)track->entry_count * sizeof(GmIso4));

	TmnfMeshGrid *mesh_grids = (TmnfMeshGrid *)malloc(
		(size_t)track->mesh_count * sizeof(TmnfMeshGrid));
	if (mesh_grids == NULL)
		cuda_fail("out of memory");
	for (uint32_t m = 0; m < track->mesh_count; ++m) {
		mesh_grids[m] = grid->mesh_grids[m];
		mesh_grids[m].pool = shadow->grid.mesh_pool;
		if (mesh_grids[m].sphere_edges != NULL) {
			mesh_grids[m].sphere_edges = (const TmnfSphereFaceEdges *)track_upload(
				result, mesh_grids[m].sphere_edges,
				(size_t)track->meshes[m].face_count * sizeof(TmnfSphereFaceEdges));
		}
		for (int l = 0; l < TMNF_MESH_GRID_LEVELS; ++l) {
			TmnfMeshGridLevel *level = &mesh_grids[m].levels[l];
			if (level->cell_count == 0) {
				level->cells = NULL;
				continue;
			}
			level->cells = (const uint32_t *)track_upload(result,
				level->cells,
				(size_t)level->cell_count * sizeof(uint32_t));
		}
	}
	shadow->grid.mesh_grids = (const TmnfMeshGrid *)track_upload(result,
		mesh_grids, (size_t)track->mesh_count * sizeof(TmnfMeshGrid));
	free(mesh_grids);

	const TmnfMeshGrid **entry_grids = (const TmnfMeshGrid **)malloc(
		(size_t)track->entry_count * sizeof(*entry_grids));
	if (entry_grids == NULL)
		cuda_fail("out of memory");
	for (uint32_t i = 0; i < track->entry_count; ++i) {
		const TmnfMeshGrid *host = grid->entry_mesh_grids[i];
		if (host == NULL) {
			entry_grids[i] = NULL;
			continue;
		}
		size_t index = (size_t)(host - grid->mesh_grids);
		if (index >= track->mesh_count)
			cuda_fail("entry mesh grid outside the mesh grid table");
		entry_grids[i] = shadow->grid.mesh_grids + index;
	}
	shadow->grid.entry_mesh_grids = (const TmnfMeshGrid *const *)track_upload(
		result, entry_grids,
		(size_t)track->entry_count * sizeof(*entry_grids));
	free(entry_grids);
}

static DeviceTrack upload_track(const TmnfTrack *track)
{
	DeviceTrack result;
	memset(&result, 0, sizeof(result));
	size_t size = track->mapping_size;
	uint8_t *staging = (uint8_t *)malloc(size);
	if (staging == NULL)
		cuda_fail("out of memory staging the track");
	memcpy(staging, track->mapping, size);
	result.mapping = (uint8_t *)device_alloc(size);

	const TmnfTrackHeader *header = track->header;
	const uint8_t *host_base = (const uint8_t *)track->mapping;
#define REBASE(field) \
	(field) = rebase((field), host_base, size, result.mapping)
	{
		const TmnfTrackSection *section =
			&header->sections[TMNF_TRACK_ENTRIES];
		for (uint32_t i = 0; i < section->count; ++i) {
			TmnfTrackStaticEntry *entry = (TmnfTrackStaticEntry *)(
				staging + section->offset + (size_t)i * section->stride);
			REBASE(entry->surface_rel);
		}
	}
	{
		const TmnfTrackSection *section =
			&header->sections[TMNF_TRACK_SURFACES];
		for (uint32_t i = 0; i < section->count; ++i) {
			TmnfTrackSurface *surface = (TmnfTrackSurface *)(
				staging + section->offset + (size_t)i * section->stride);
			REBASE(surface->mesh_rel);
			REBASE(surface->material_ids_rel);
		}
	}
	{
		const TmnfTrackSection *section =
			&header->sections[TMNF_TRACK_MESHES];
		for (uint32_t i = 0; i < section->count; ++i) {
			TmnfTrackMesh *mesh = (TmnfTrackMesh *)(
				staging + section->offset + (size_t)i * section->stride);
			REBASE(mesh->vertices_rel);
			REBASE(mesh->faces_rel);
			REBASE(mesh->nodes_rel);
		}
	}
#undef REBASE
	check(cudaMemcpy(result.mapping, staging, size, cudaMemcpyHostToDevice),
		"track upload");
	free(staging);

	TmnfTrack shadow = *track;
	shadow.mapping = result.mapping;
#define REBASE_PTR(field, type) \
	shadow.field = (type)(uintptr_t)rebase( \
		(uint64_t)(uintptr_t)track->field, host_base, size, result.mapping)
	REBASE_PTR(header, const TmnfTrackHeader *);
	REBASE_PTR(entries, const HmsStaticCollisionEntry *);
	REBASE_PTR(surfaces, const CPlugSurface *);
	REBASE_PTR(meshes, const GmSurfMesh *);
	REBASE_PTR(materials, const TmnfTrackMaterialData *);
	REBASE_PTR(collision_pairs, const TmnfTrackCollisionPair *);
	REBASE_PTR(corpus_isos, const GmIso4 *);
	REBASE_PTR(water.cells, const uint8_t *);
#undef REBASE_PTR
	upload_static_grid(&result, &shadow, track);
	shadow.static_response_bodies = (CHmsResponseBody *)track_upload(
		&result, track->static_response_bodies,
		(size_t)track->static_response_count *
			sizeof(*track->static_response_bodies));
	shadow.static_response_present = (uint8_t *)track_upload(
		&result, track->static_response_present,
		track->static_response_count);
	result.track = (TmnfTrack *)device_upload(&shadow, sizeof(shadow));
	return result;
}

typedef struct {
	uint8_t *mapping;
	TmnfRouteBvhNode *bvh;
	TmnfRoute *route;
} DeviceRoute;

static DeviceRoute upload_route(const TmnfRoute *route)
{
	DeviceRoute result;
	if (route->mapping == NULL || route->projection_bvh == NULL)
		cuda_fail("route is not a loaded snapshot with a projection BVH");
	size_t size = route->mapping_size;
	const uint8_t *host_base = (const uint8_t *)route->mapping;
	uint8_t *staging = (uint8_t *)malloc(size);
	if (staging == NULL)
		cuda_fail("out of memory staging the route");
	memcpy(staging, route->mapping, size);
	result.mapping = (uint8_t *)device_alloc(size);
	TmnfRouteMetadata *metadata = (TmnfRouteMetadata *)(
		staging + route->header->sections[TMNF_ROUTE_METADATA].offset);
	metadata->start_rel =
		rebase(metadata->start_rel, host_base, size, result.mapping);
	metadata->checkpoints_rel =
		rebase(metadata->checkpoints_rel, host_base, size, result.mapping);
	metadata->finish_rel =
		rebase(metadata->finish_rel, host_base, size, result.mapping);
	metadata->reference_rel =
		rebase(metadata->reference_rel, host_base, size, result.mapping);
	check(cudaMemcpy(result.mapping, staging, size, cudaMemcpyHostToDevice),
		"route upload");
	free(staging);
	result.bvh = (TmnfRouteBvhNode *)device_upload(
		route->projection_bvh,
		(size_t)route->projection_bvh_count * sizeof(TmnfRouteBvhNode));

	TmnfRoute shadow = *route;
	shadow.mapping = result.mapping;
#define REBASE_PTR(field, type) \
	shadow.field = (type)(uintptr_t)rebase( \
		(uint64_t)(uintptr_t)route->field, host_base, size, result.mapping)
	REBASE_PTR(header, const TmnfRouteHeader *);
	REBASE_PTR(metadata, const TmnfRouteMetadata *);
	REBASE_PTR(start, const TmnfRouteStart *);
	REBASE_PTR(checkpoints, const TmnfRouteTrigger *);
	REBASE_PTR(finish, const TmnfRouteTrigger *);
	REBASE_PTR(centerline, const TmnfRouteReferencePoint *);
#undef REBASE_PTR
	shadow.projection_bvh = result.bvh;
	result.route = (TmnfRoute *)device_upload(&shadow, sizeof(shadow));
	return result;
}

/* ------------------------------------------------------------------------ */
/* Device environment                                                         */
/* ------------------------------------------------------------------------ */

/* Everything a kernel needs, passed by value. Per-environment arrays are
 * indexed by the thread's environment. */
typedef struct {
	uint32_t count;
	size_t world_size;              /* World_Size(): TmnfWorld is opaque */
	const TmnfWorld *template_world; /* the host world as uploaded */
	/* The immutable half of the world (World_ColdSize bytes), one copy at
	 * one address for every environment, linked by init_template. */
	void *cold;
	/* A linked template initializes each environment. Persistent state is
	 * tiled by 16-byte word group across environments: every warp reads and
	 * writes adjacent records without template comparisons or indexed diffs. */
	TmnfWorld *linked_template;
	uintptr_t *template_base;
	uint4 *world_state;
	CSceneVehicleCarWheelHistory *wheel_history;
	const CSceneVehicleCarWheelHistory *history_template;
	/* The template's wheel contact bodies as corpus_ref tokens: the only
	 * state pointers in the vehicle graph, re-resolved on the device. */
	uint32_t template_contact_refs[TMNF_STADIUM_WHEEL_COUNT];
	TmnfWorldLinkSources link;      /* shared sources; scratch bases below */
	SHmsPhysicalCollision *collision_records;
	SHmsPhysicalCollision *contact_records;
	GmVec3 *replacements;
	uint32_t collision_capacity;
	uint32_t contact_capacity;
	uint32_t replacement_capacity;
	/* Workspace of the warp-cooperative sections (tmnf_warp.h), one
	 * warp_workspace_stride-byte block per warp of the largest launch. */
	uint8_t *warp_workspace;
	size_t warp_workspace_stride;

	const TmnfRoute *route;
	TmnfVecEnvConfig config;
	float fell_floor_y;
	const float *lookahead_meters;
	const float *gamma_powers;      /* glibc powf(gamma, k), k <= max ticks */

	TmnfEnvSnapshot *reset_snapshot;
	TmnfRaceState *race_states;
	uint64_t *episode_ids;
	float *episode_returns;
	uint8_t *reset_pending;

	TmnfStepResult *results;
	float *flat_observations;
	float *flat_final_observations;
	float *flat_transitions;
} DeviceEnv;

struct TmnfCudaVecEnv {
	/* Every launch and every host <-> device copy after Create is ordered on
	 * this stream and waited for on it (TmnfCudaVecEnvLimits.stream; NULL is
	 * the legacy default stream). */
	cudaStream_t stream;
	DeviceEnv device;
	DeviceTrack track;
	DeviceRoute route;
	uint8_t *vehicle_blob;
	float *curve_bounds;
	uint8_t *fake_contact_mask;
	float *water_tables;
	float *lookahead_meters;
	float *gamma_powers;
	uint32_t threads_per_block;
	int has_route;

	/* Host-side staging for arguments and results. */
	void *device_actions;           /* count * max(sizeof inputs) */
	TmnfEnvSnapshot *device_snapshots;
	uint32_t snapshot_capacity;
	TmnfObservation *device_observations;
	uint32_t *device_indices;
	uint8_t *device_mask;
	TmnfWorld *device_world;        /* one exported world */
};

__device__ static void env_fail(const char *message)
{
	tmnf_fail(message);
}

/* Every kernel works on a copy of its environment's world in the thread's
 * local memory (world_enter). The generic address of a local buffer is the
 * same in every thread, so one shared word per block names the world of
 * whichever thread asks. */
__shared__ TmnfWorld *block_world;

__device__ static TmnfWorld *env_world(const DeviceEnv *env, uint32_t index)
{
	(void)env;
	(void)index;
	return block_world;
}

__device__ static TmnfPhysicsCorpus *player_corpus(
	const DeviceEnv *env, uint32_t index)
{
	return &World_GetPhysicsWorld(env_world(env, index))->corpora[0];
}

__device__ static CHmsResponseBody *resolve_contact_body(
	const DeviceEnv *env, uint32_t index, uint32_t body_ref);

/* ---- vec_env.c mirrors ------------------------------------------------- */

__device__ static void write_physics_observation(
	const TmnfPhysicsCorpus *corpus, TmnfObservation *out)
{
	const CHmsStateDyna *state = corpus->dyna->liveState;
	const CSceneVehicleCar *car = corpus->vehicle;

	memset(out, 0, sizeof(*out));
	out->position = state->pos;
	out->rotation = state->quat;
	out->linear_speed = state->linVel;
	out->angular_speed = state->angVel;
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		out->wheel_speed[i] = car->wheels[i].real_time.field6c;
		out->wheel_damper[i] =
			car->wheels[i].real_time.damper_absorb;
		out->wheel_contact[i] = (float)
			car->wheels[i].real_time.has_ground_contact;
		out->wheel_sliding[i] = (float)
			car->wheels[i].real_time.is_sliding;
		out->wheel_material[i] = (float)(uint16_t)
			car->wheels[i].real_time.contact_material_id;
	}
	out->engine_rpm = car->engine.rpm;
	out->gear = car->engine.gear;
	out->input_steer = car->input_steer;
	out->input_gas = car->input_gas;
	out->input_brake = car->input_brake;
}

__device__ static TmnfCenterlineSample centerline_sample_at(
	const TmnfRoute *route, float arc_length, uint32_t minimum_segment)
{
	const TmnfRouteReferencePoint *points =
		TmnfRoute_GetReferencePoints(route);
	uint32_t count = TmnfRoute_GetReferencePointCount(route);
	if (count < 2 || minimum_segment + 1 >= count)
		env_fail("centerline sample cursor is invalid");
	if (arc_length < points[minimum_segment].arc_length)
		env_fail("centerline sample precedes projection cursor");
	TmnfCenterlineSample sample;
	if (arc_length >= points[count - 1].arc_length) {
		sample.position = points[count - 1].position;
		sample.half_width = points[count - 1].half_width;
		return sample;
	}

	uint32_t lower = minimum_segment;
	uint32_t upper = count - 1;
	while (lower + 1 < upper) {
		uint32_t middle = lower + (upper - lower) / 2;
		if (points[middle].arc_length <= arc_length)
			lower = middle;
		else
			upper = middle;
	}
	const TmnfRouteReferencePoint *a = &points[lower];
	const TmnfRouteReferencePoint *b = &points[lower + 1];
	float t = (arc_length - a->arc_length) /
		(b->arc_length - a->arc_length);
	sample.position.x = a->position.x + t * (b->position.x - a->position.x);
	sample.position.y = a->position.y + t * (b->position.y - a->position.y);
	sample.position.z = a->position.z + t * (b->position.z - a->position.z);
	sample.half_width = a->half_width +
		t * (b->half_width - a->half_width);
	return sample;
}

__device__ static void write_rl_observation(
	const DeviceEnv *env, uint32_t index, TmnfObservation *out)
{
	const TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	const TmnfRaceState *race = &env->race_states[index];
	const CHmsStateDyna *state = corpus->dyna->liveState;
	const CSceneVehicleCarAuxContext *aux =
		corpus->vehicle_compute->aux;
	write_physics_observation(corpus, out);
	if (aux->turbo_type != TMNF_TURBO_NONE) {
		out->turbo_active = 1.0f;
		out->turbo_type = (float)aux->turbo_type;
		out->turbo_remaining_progress =
			1.0f - aux->turbo_progress;
	}

	float route_length = TmnfRoute_GetReferenceLength(env->route);
	float total_length = route_length * env->route->metadata->lap_count;
	out->arc_length = race->arc_length;
	out->unwrapped_progress = race->unwrapped_progress;
	out->lateral_offset = race->lateral_offset;
	out->track_half_width = race->half_width;
	out->remaining_distance = total_length - race->unwrapped_progress;
	out->elapsed_fraction = (float)race->elapsed_ticks /
		(float)env->config.max_race_ticks;
	out->next_checkpoint_fraction =
		env->route->metadata->checkpoint_count == 0
			? 1.0f
			: (float)race->visited_count /
				(float)env->route->metadata->checkpoint_count;
	out->completed_lap_fraction = (float)race->completed_laps /
		(float)env->route->metadata->lap_count;

	GmIso4 inverse_car;
	GmIso4_SetInverse(
		&inverse_car, (const GmIso4 *)&state->rot);
	for (uint32_t i = 0;
		i < TMNF_OBSERVATION_LOOKAHEAD_COUNT; ++i) {
		float sample_arc = race->arc_length +
			env->lookahead_meters[i];
		TmnfCenterlineSample sample = centerline_sample_at(
			env->route, sample_arc, race->centerline_segment);
		GmVec3_SetMult_Iso4(
			&out->centerline_lookahead[i].position,
			&sample.position, &inverse_car);
		out->centerline_lookahead[i].half_width =
			sample.half_width;
	}
}

__device__ static TMNFVehicleComputeContext *vehicle_context(
	const TmnfPhysicsCorpus *corpus)
{
	TMNFVehicleComputeContext *compute = corpus->vehicle_compute;
	if (compute == NULL || compute->aux == NULL ||
		compute->contact == NULL || compute->model6 == NULL ||
		compute->state == NULL || compute->aux->wheels == NULL ||
		compute->contact->wheels == NULL ||
		compute->model6->state == NULL ||
		compute->contact->timer == NULL ||
		compute->contact->wheel_count != TMNF_STADIUM_WHEEL_COUNT) {
		env_fail("vehicle context graph is incomplete");
	}
	return compute;
}

__device__ static CHmsResponseBody *resolve_contact_body(
	const DeviceEnv *env, uint32_t index, uint32_t body_ref)
{
	const CHmsResponseZone *zone =
		&World_GetPhysicsWorld(env_world(env, index))->response_zone;
	if (zone->resolve_body == NULL)
		env_fail("world has no response body resolver");
	return zone->resolve_body(zone->resolver_user, body_ref);
}

__device__ static uint32_t contact_body_ref(
	const DeviceEnv *env, uint32_t index, const CHmsResponseBody *body)
{
	if (body == NULL)
		return TMNF_ENV_NO_CONTACT_BODY;
	if (resolve_contact_body(env, index, body->corpus_ref) != body)
		env_fail("wheel contact body is not addressable by its token");
	return body->corpus_ref;
}

__device__ static void scrub_car_pointers(CSceneVehicleCar *car)
{
	car->hms_item = NULL;
	car->dyna_state = NULL;
	car->dyna_params = NULL;
	car->tuning = NULL;
	car->wheels = NULL;
}

__device__ static void scrub_aux_pointers(CSceneVehicleCarAuxContext *aux)
{
	aux->vehicle = NULL;
	aux->tuning = NULL;
	aux->wheels = NULL;
	aux->runtime = NULL;
	aux->play_turbo_sound = NULL;
	aux->set_surface_location = NULL;
	aux->finish_integration = NULL;
}

__device__ static void scrub_contact_pointers(
	TMNFVehicleContactContext *contact)
{
	contact->vehicle = NULL;
	contact->tuning = NULL;
	contact->wheels = NULL;
	contact->ground_material_indices = NULL;
	contact->ground_materials = NULL;
	contact->timer = NULL;
	contact->vehicle_contact_rotation = NULL;
	contact->resolve_body_iso = NULL;
	contact->resolve_body_iso_user = NULL;
	contact->wheel_tree_refs = NULL;
	contact->body_tree_refs = NULL;
	contact->air_control_immediate = NULL;
	contact->contact_block_count = NULL;
	contact->event_source_c = NULL;
	contact->event_source_ab = NULL;
	contact->event_metric_a = NULL;
	contact->event_metric_b = NULL;
	contact->event_metric_c = NULL;
}

__device__ static void capture_physics(
	const DeviceEnv *env, uint32_t index, TmnfEnvSnapshot *snapshot)
{
	const TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	const CHmsDyna *dyna = corpus->dyna;
	const CSceneVehicleCar *car = corpus->vehicle;
	const TMNFVehicleComputeContext *compute = vehicle_context(corpus);

	memset(snapshot, 0, sizeof(*snapshot));
	snapshot->version = TMNF_ENV_SNAPSHOT_VERSION;
	snapshot->live_state = *dyna->liveState;
	snapshot->committed_state = *dyna->stateB;
	snapshot->temp_state = dyna->tempState;
	snapshot->dyna_params = *dyna->params;
	snapshot->dyna_dirty = dyna->dirtyFlag;

	snapshot->car = *car;
	scrub_car_pointers(&snapshot->car);
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i)
		CSceneVehicleCarWheel_Capture(&car->wheels[i], &snapshot->wheels[i]);

	snapshot->aux = *compute->aux;
	scrub_aux_pointers(&snapshot->aux);
	memcpy(snapshot->aux_wheels, compute->aux->wheels,
		sizeof(snapshot->aux_wheels));

	snapshot->contact = *compute->contact;
	scrub_contact_pointers(&snapshot->contact);
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		const TMNFVehicleContactWheelState *wheel =
			&compute->contact->wheels[i];
		snapshot->contact_wheels[i] = *wheel;
		snapshot->contact_wheels[i].wheel = NULL;
		snapshot->contact_wheels[i].contact_body = NULL;
		snapshot->contact_body_refs[i] =
			contact_body_ref(env, index, wheel->contact_body);
	}

	snapshot->model6_state = *compute->model6->state;
	snapshot->compute_state = *compute->state;
	snapshot->contact_timer_tick = compute->contact->timer->tick_time;
	snapshot->physics_tick_time = corpus->tick_time;
	snapshot->vehicle_context_valid = 1;
}

__device__ static void restore_physics(
	const DeviceEnv *env, uint32_t index, const TmnfEnvSnapshot *snapshot)
{
	if (snapshot->version != TMNF_ENV_SNAPSHOT_VERSION)
		env_fail("snapshot version is unsupported");
	if (snapshot->vehicle_context_valid == 0)
		env_fail("snapshot has no vehicle context state");
	TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	CHmsDyna *dyna = corpus->dyna;
	CSceneVehicleCar *car = corpus->vehicle;
	TMNFVehicleComputeContext *compute = vehicle_context(corpus);
	CSceneVehicleCarAuxContext *aux = compute->aux;
	TMNFVehicleContactContext *contact = compute->contact;

	CSceneVehicleCar car_topology = *car;
	CSceneVehicleCarAuxContext aux_topology = *aux;
	TMNFVehicleContactContext contact_topology = *contact;
	void *surface_handlers[TMNF_STADIUM_WHEEL_COUNT];
	CSceneVehicleCarWheel *contact_wheel_refs[TMNF_STADIUM_WHEEL_COUNT];
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		surface_handlers[i] = car->wheels[i].surface_handler;
		contact_wheel_refs[i] = contact->wheels[i].wheel;
	}

	*dyna->liveState = snapshot->live_state;
	*dyna->stateB = snapshot->committed_state;
	dyna->tempState = snapshot->temp_state;
	*dyna->params = snapshot->dyna_params;
	dyna->replacementBuf.count = 0;
	dyna->dirtyFlag = snapshot->dyna_dirty;

	*car = snapshot->car;
	car->hms_item = car_topology.hms_item;
	car->dyna_state = car_topology.dyna_state;
	car->dyna_params = car_topology.dyna_params;
	car->tuning = car_topology.tuning;
	car->wheels = car_topology.wheels;
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		CSceneVehicleCarWheel_Restore(&car->wheels[i], &snapshot->wheels[i]);
		car->wheels[i].surface_handler = surface_handlers[i];
	}

	*aux = snapshot->aux;
	aux->vehicle = aux_topology.vehicle;
	aux->tuning = aux_topology.tuning;
	aux->wheels = aux_topology.wheels;
	aux->runtime = aux_topology.runtime;
	aux->play_turbo_sound = aux_topology.play_turbo_sound;
	aux->set_surface_location = aux_topology.set_surface_location;
	aux->finish_integration = aux_topology.finish_integration;
	memcpy(aux->wheels, snapshot->aux_wheels, sizeof(snapshot->aux_wheels));

	*contact = snapshot->contact;
	contact->vehicle = contact_topology.vehicle;
	contact->tuning = contact_topology.tuning;
	contact->wheels = contact_topology.wheels;
	contact->ground_material_indices =
		contact_topology.ground_material_indices;
	contact->ground_materials = contact_topology.ground_materials;
	contact->timer = contact_topology.timer;
	contact->vehicle_contact_rotation =
		contact_topology.vehicle_contact_rotation;
	contact->resolve_body_iso = contact_topology.resolve_body_iso;
	contact->resolve_body_iso_user = contact_topology.resolve_body_iso_user;
	contact->wheel_tree_refs = contact_topology.wheel_tree_refs;
	contact->body_tree_refs = contact_topology.body_tree_refs;
	contact->air_control_immediate = contact_topology.air_control_immediate;
	contact->contact_block_count = contact_topology.contact_block_count;
	contact->event_source_c = contact_topology.event_source_c;
	contact->event_source_ab = contact_topology.event_source_ab;
	contact->event_metric_a = contact_topology.event_metric_a;
	contact->event_metric_b = contact_topology.event_metric_b;
	contact->event_metric_c = contact_topology.event_metric_c;
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		TMNFVehicleContactWheelState *wheel = &contact->wheels[i];
		*wheel = snapshot->contact_wheels[i];
		wheel->wheel = contact_wheel_refs[i];
		uint32_t body_ref = snapshot->contact_body_refs[i];
		if (body_ref == TMNF_ENV_NO_CONTACT_BODY) {
			wheel->contact_body = NULL;
			continue;
		}
		wheel->contact_body = resolve_contact_body(env, index, body_ref);
		if (wheel->contact_body == NULL)
			env_fail("snapshot contact body has no counterpart here");
	}

	*compute->model6->state = snapshot->model6_state;
	*compute->state = snapshot->compute_state;
	contact->timer->tick_time = snapshot->contact_timer_tick;
	corpus->tick_time = snapshot->physics_tick_time;

	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		CPlugTree *tree = (CPlugTree *)car->wheels[i].surface_handler;
		if (tree == NULL)
			env_fail("wheel surface tree is null");
		tree->local_iso = aux->wheels[i].surface_location;
		tree->box.center.x = aux->wheels[i].surface_location.t[0];
		tree->box.center.y = aux->wheels[i].surface_location.t[1];
		tree->box.center.z = aux->wheels[i].surface_location.t[2];
	}
}

__device__ static void reset_one(const DeviceEnv *env, uint32_t index)
{
	TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	restore_physics(env, index, env->reset_snapshot);

	env->episode_ids[index]++;
	env->episode_returns[index] = 0.0f;
	env->reset_pending[index] = 0;
	TmnfRace_Reset(
		env->route, &env->race_states[index],
		corpus->collision_corpus->live_iso);
}

__device__ static void capture_env_state(
	const DeviceEnv *env, uint32_t index, TmnfEnvSnapshot *snapshot)
{
	capture_physics(env, index, snapshot);
	snapshot->race_state = env->race_states[index];
	snapshot->episode_id = env->episode_ids[index];
	snapshot->episode_return = env->episode_returns[index];
	snapshot->reset_pending = env->reset_pending[index];
	snapshot->race_state_valid = 1;
}

__device__ static void restore_env_state(
	const DeviceEnv *env, uint32_t index, const TmnfEnvSnapshot *snapshot)
{
	restore_physics(env, index, snapshot);
	if (snapshot->race_state_valid == 0)
		env_fail("snapshot has no race state");
	env->race_states[index] = snapshot->race_state;
	env->episode_ids[index] = snapshot->episode_id;
	env->episode_returns[index] = snapshot->episode_return;
	env->reset_pending[index] = snapshot->reset_pending;
}

/* TmnfPhysicsCorpus_AdvanceTimer lives in vec_env.c, which is host-only. */
__device__ static void advance_timer(TmnfPhysicsCorpus *corpus, uint32_t tick_ms)
{
	if (corpus == NULL || corpus->vehicle_compute == NULL ||
		corpus->vehicle_compute->contact == NULL ||
		corpus->vehicle_compute->contact->timer == NULL) {
		env_fail("corpus has no game timer");
	}
	TMNFVehicleContactTimer *timer =
		corpus->vehicle_compute->contact->timer;
	timer->tick_time += tick_ms;
	corpus->tick_time = timer->tick_time;
}

/* The respawn press is handled before the tick's control mapping
 * (0x0047DCD0 OnInputEvent); vec_env.c physics_tick. */
__device__ static void apply_respawn(
	const DeviceEnv *env, uint32_t index, const TMNFRaceInputs *input)
{
	if (input->respawn == 0)
		return;
	if (env->race_states == NULL)
		env_fail("respawn needs the race layer");
	const GmIso4 *spawn =
		TmnfRace_RespawnLocation(&env->race_states[index]);
	if (spawn == NULL)
		env_fail("respawn before any checkpoint restarts the race");
	TmnfVehicle_Respawn(player_corpus(env, index), spawn);
}

__device__ static void physics_tick(
	const DeviceEnv *env, uint32_t index, const TMNFRaceInputs *input,
	uint32_t tick_ms)
{
	TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	apply_respawn(env, index, input);
	CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
		input, corpus->vehicle);
	advance_timer(corpus, tick_ms);
	CHmsZoneDynamic_PhysicsStep2(
		World_GetPhysicsWorld(env_world(env, index)), tick_ms);
}

__device__ static TMNFRaceInputs discrete_input(
	const DeviceEnv *env, uint8_t action, uint32_t timestamp)
{
	uint32_t respawn = (action & TMNF_DISCRETE_RESPAWN_FLAG) != 0;
	if (respawn && env->config.respawn_action == 0)
		env_fail("discrete respawn used without respawn_action");
	action &= (uint8_t)~TMNF_DISCRETE_RESPAWN_FLAG;
	if (action >= TMNF_DISCRETE_ACTION_COUNT)
		env_fail("discrete action is out of range");
	uint32_t longitudinal = action / 3;
	uint32_t steering = action % 3;
	TMNFRaceInputs input;
	memset(&input, 0, sizeof(input));
	input.steer_left_time = timestamp;
	input.steer_right_time = timestamp;
	input.accelerate_time = timestamp;
	input.brake_time = timestamp;
	input.steer_left = steering == 0;
	input.steer_right = steering == 2;
	input.accelerate = longitudinal == 1 || longitudinal == 3;
	input.brake = longitudinal == 2 || longitudinal == 3;
	input.respawn = respawn;
	return input;
}

__device__ static int32_t quantize_analog_steer(float steer)
{
	if (!isfinite(steer) || steer < -1.0f || steer > 1.0f)
		env_fail("analog steer is outside [-1, 1]");
	return (int32_t)roundf(steer * 65536.0f);
}

__device__ static TMNFRaceInputs analog_input(
	const DeviceEnv *env, const TmnfAnalogAction *action,
	uint32_t timestamp)
{
	if (action->respawn != 0 && env->config.respawn_action == 0)
		env_fail("analog respawn used without respawn_action");
	int32_t steer = quantize_analog_steer(action->steer);
	TMNFRaceInputs input;
	memset(&input, 0, sizeof(input));
	input.steer_analog_time = timestamp;
	input.steer_analog = (float)(-steer) / 65536.0f;
	input.accelerate_time = timestamp;
	input.accelerate = action->gas;
	input.brake_time = timestamp;
	input.brake = action->brake;
	input.respawn = action->respawn;
	return input;
}

__device__ static float race_potential(const DeviceEnv *env, uint32_t index)
{
	float total_distance = TmnfRoute_GetReferenceLength(env->route) *
		env->route->metadata->lap_count;
	return -(total_distance -
		env->race_states[index].unwrapped_progress) /
		env->config.reference_speed;
}

/* powf(gamma, k) as glibc computes it, tabulated at Create for every
 * k <= max_race_ticks; both call sites pass an exact integer exponent. */
__device__ static float gamma_power(const DeviceEnv *env, uint32_t k)
{
	if (k > env->config.max_race_ticks)
		env_fail("discount exponent exceeds the race budget");
	return env->gamma_powers[k];
}

__device__ static float failure_base_reward(
	const DeviceEnv *env, uint32_t elapsed_before_failure)
{
	uint32_t remaining = env->config.max_race_ticks -
		elapsed_before_failure;
	float gamma = env->config.discount_per_tick;
	if (gamma == 1.0f)
		return -0.01f * (float)remaining;
	return -0.01f *
		(1.0f - gamma_power(env, remaining)) /
		(1.0f - gamma);
}

__device__ static TmnfTerminationReason failure_reason(
	const DeviceEnv *env, uint32_t index)
{
	TmnfRaceState *race = &env->race_states[index];
	const TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	const CHmsStateDyna *state = corpus->dyna->liveState;
	uint32_t wheels_in_contact = 0;
	uint32_t wheels_on_ground_plane = 0;
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		const CSceneVehicleCarWheelRealTimeState *wheel =
			&corpus->vehicle->wheels[i].real_time;
		if (wheel->has_ground_contact == 0)
			continue;
		wheels_in_contact++;
		if (TmnfRace_IsGroundPlaneMaterial(
				(uint16_t)wheel->contact_material_id))
			wheels_on_ground_plane++;
	}
	int off_track = TmnfRace_UpdateOffTrack(
		race, env->config.off_track_grace_ticks,
		wheels_in_contact, wheels_on_ground_plane);

	float delta = fabsf(
		race->unwrapped_progress - race->previous_progress);
	if (delta <= env->config.stuck_progress_epsilon)
		race->stuck_ticks++;
	else
		race->stuck_ticks = 0;

	if (race->elapsed_ticks >= env->config.max_race_ticks)
		return TMNF_TERMINATION_TIMEOUT;
	if (state->pos.y < env->fell_floor_y)
		return TMNF_TERMINATION_FELL;
	if (off_track)
		return TMNF_TERMINATION_OFF_TRACK;
	if (race->stuck_ticks >= env->config.stuck_grace_ticks)
		return TMNF_TERMINATION_STUCK;
	return TMNF_TERMINATION_NONE;
}

__device__ static float tick_reward(
	const DeviceEnv *env, uint32_t index, TmnfTerminationReason reason,
	uint32_t elapsed_before, float previous_potential)
{
	float base = -0.01f;
	float next_potential = race_potential(env, index);
	if (reason == TMNF_TERMINATION_FINISH) {
		float sample_spacing =
			TmnfRoute_GetReferenceLength(env->route) /
			(float)(TmnfRoute_GetReferencePointCount(env->route) - 1);
		if (-next_potential * env->config.reference_speed >
			sample_spacing) {
			env_fail("finish left remaining distance");
		}
		next_potential = 0.0f;
	} else if (reason != TMNF_TERMINATION_NONE) {
		base = failure_base_reward(env, elapsed_before);
		next_potential *= gamma_power(env,
			env->config.max_race_ticks - (elapsed_before + 1));
	}
	return base + env->config.discount_per_tick * next_potential -
		previous_potential;
}

/* vec_env.c step_one_tick_input: `first` marks the first tick of an action
 * repeat; a respawn is a press edge, so only that tick respawns. A press
 * before any respawnable checkpoint is the game's race restart: the tick
 * runs (the car coasts), then the episode ends with
 * TMNF_TERMINATION_RESTART and the failure reward. */
__device__ static int step_one_tick(
	const DeviceEnv *env, uint32_t index, TMNFRaceInputs input, int first,
	float reward_weight, TmnfStepResult *result)
{
	TmnfPhysicsCorpus *corpus = player_corpus(env, index);
	TmnfRaceState *race = &env->race_states[index];
	float previous_potential = race_potential(env, index);
	uint32_t elapsed_before = race->elapsed_ticks;
	int restart = 0;
	if (input.respawn != 0) {
		if (!first || TmnfRace_RespawnLocation(race) == NULL) {
			restart = first;
			input.respawn = 0;
		}
	}
	physics_tick(env, index, &input, TMNF_RACE_TICK_MS);

	TmnfRaceStepResult race_result = TmnfRace_Step(
		env->route, race,
		World_GetPhysicsWorld(env_world(env, index))->trigger_contacts,
		corpus->collision_corpus->live_iso);
	TmnfTerminationReason reason = race_result.finished
		? TMNF_TERMINATION_FINISH
		: failure_reason(env, index);
	if (reason == TMNF_TERMINATION_NONE && restart)
		reason = TMNF_TERMINATION_RESTART;
	result->terminated = reason != TMNF_TERMINATION_NONE;
	result->truncated =
		race->elapsed_ticks >= env->config.horizon_ticks;
	result->termination_reason = reason;
	result->race_time_ms = race_result.race_time_ms;
	result->reward += reward_weight * tick_reward(
		env, index, reason, elapsed_before, previous_potential);
	return result->terminated || result->truncated;
}

__device__ static void flatten_observation(
	const TmnfObservation *observation, float *flat)
{
	memcpy(flat, observation, 34 * sizeof(*flat));
	flat[34] = (float)observation->gear;
	memcpy(flat + 35, &observation->input_steer, 11 * sizeof(*flat));
	memcpy(flat + 46, observation->centerline_lookahead,
		TMNF_OBSERVATION_LOOKAHEAD_COUNT * sizeof(TmnfCenterlineSample));
	memcpy(flat + 78, &observation->turbo_active, 3 * sizeof(*flat));
}

/* The flat rows of one step (TmnfVecEnv_FlattenStepResults), staged in
 * local memory next to the TmnfStepResult they derive from. */
typedef struct {
	float observation[TMNF_POLICY_OBSERVATION_WIDTH];
	float final_observation[TMNF_POLICY_OBSERVATION_WIDTH];
	float transition[TMNF_POLICY_TRANSITION_WIDTH];
} StepOutputs;

__device__ static void publish_result(
	const TmnfStepResult *result, StepOutputs *outputs)
{
	flatten_observation(&result->observation, outputs->observation);
	flatten_observation(&result->final_observation,
		outputs->final_observation);
	float *transition = outputs->transition;
	transition[0] = result->reward;
	transition[1] = result->transition_discount;
	transition[2] = (float)result->terminated;
	transition[3] = (float)result->truncated;
	transition[4] = (float)result->executed_ticks;
}

__device__ static TMNFRaceInputs make_input(
	const DeviceEnv *env, const uint8_t *action, uint32_t timestamp)
{
	return discrete_input(env, *action, timestamp);
}

__device__ static TMNFRaceInputs make_input(
	const DeviceEnv *env, const TmnfAnalogAction *action,
	uint32_t timestamp)
{
	return analog_input(env, action, timestamp);
}

/* step_discrete_range / step_analog_range for one environment. The action
 * is resolved to an input packet per tick through the timestamp. */
__device__ static void write_gate_observation(const DeviceEnv *env,
 uint32_t index, TmnfGateObservations *out)
{
 TmnfRace_ObserveGates(env->route, &env->race_states[index],
  player_corpus(env, index)->collision_corpus->live_iso, out);
}

template <typename Action>
__device__ static void step_rl_one(
	const DeviceEnv *env, uint32_t index, const Action *action,
	uint32_t action_repeat, TmnfStepResult *result, StepOutputs *outputs,
 TmnfGateObservations *gates, TmnfGateObservations *final_gates)
{
	memset(result, 0, sizeof(*result));
 if (final_gates) memset(final_gates, 0, sizeof(*final_gates));
	result->episode_id = env->episode_ids[index];
	/* The physics is warp-collective (tmnf_warp.h): the lanes still
	 * stepping agree on their mask before every tick, and a lane leaving
	 * the loop drops out of it. */
	uint32_t mask = tmnf_dev_warp_mask();
	int pending = env->reset_pending[index] != 0;
	mask = __ballot_sync(mask, !pending);
	if (pending) {
		env->reset_pending[index] = 0;
		result->reset_only = 1;
		result->transition_discount = 1.0f;
		write_rl_observation(env, index, &result->observation);
  if (gates) write_gate_observation(env, index, gates);
		publish_result(result, outputs);
		return;
	}
	tmnf_dev_warp_set_mask(mask);

	int ended = 0;
	float reward_weight = 1.0f;
	for (uint32_t repeat = 0; repeat < action_repeat; ++repeat) {
		result->executed_ticks++;
		TmnfRaceState *race = &env->race_states[index];
		TMNFRaceInputs input = make_input(env,
			action, (race->elapsed_ticks + 1) * TMNF_RACE_TICK_MS);
		ended = step_one_tick(env, index, input, repeat == 0, reward_weight,
			result);
		reward_weight *= env->config.discount_per_tick;
		int more = !ended && repeat + 1 < action_repeat;
		mask = __ballot_sync(mask, more);
		if (!more)
			break;
		tmnf_dev_warp_set_mask(mask);
	}
	env->episode_returns[index] += result->reward;
	result->transition_discount = reward_weight;
	if (!ended) {
		write_rl_observation(env, index, &result->observation);
  if (gates) write_gate_observation(env, index, gates);
		publish_result(result, outputs);
		return;
	}

	write_rl_observation(env, index, &result->final_observation);
	if (final_gates) write_gate_observation(env, index, final_gates);
 result->final_observation_valid = 1;
	result->completed_episode_return = env->episode_returns[index];
	result->completed_episode_ticks = env->race_states[index].elapsed_ticks;
	TmnfObservation terminal = result->final_observation;
	reset_one(env, index);
	if (env->config.autoreset_mode == TMNF_AUTORESET_SAME_STEP) {
		write_rl_observation(env, index, &result->observation);
  if (gates) write_gate_observation(env, index, gates);
	} else {
		result->observation = terminal;
  if (gates) *gates = *final_gates;
		env->reset_pending[index] = 1;
	}
	publish_result(result, outputs);
}

/* ---- worlds in local memory ---------------------------------------------- */

/* The kernel's native world is transient. Persistent state is a dense,
 * coalesced array of 16-byte groups; the separately allocated wheel history
 * is touched only by snapshot/restore, respawn, initialization and export.
 * Local world pointers have one address in this kernel; scratch/history
 * pointers are linked once for each environment at initialization. */
typedef struct {
	uint4 rows[32][9];              /* 128-byte tile per lane, padded */
} WarpTile;

__device__ static WarpTile *warp_tile(void)
{
	extern __shared__ WarpTile tiles[];
	return &tiles[threadIdx.x / 32];
}

__host__ __device__ static size_t warp_tile_bytes(uint32_t threads)
{
	return (size_t)(threads / 32) * sizeof(WarpTile);
}

/* Writes the lane's local record to record `record` of the lane-strided
 * array at base (stride bytes apart) through the warp tile so that global
 * memory sees whole lines. All 32 lanes call this together; a lane without
 * a record passes valid = 0. Element type T sets the granularity: uint4 for
 * 16-byte aligned records, uint32_t otherwise. The lanes exchange 32-bit
 * record indices, not pointers: shuffling 64-bit pointers through
 * __shfl_sync produced wrong addresses inside these LTO-linked kernels. */
template <typename T>
__device__ static void warp_scatter(
	void *base, size_t stride, uint32_t record, const T *local,
	size_t bytes, int valid, WarpTile *tile)
{
	enum { E = 128 / sizeof(T), R = 32 / E, PAD = E + 1 };
	T *rows = (T *)tile->rows;
	uint32_t lane = threadIdx.x % 32;
	uint32_t elements = (uint32_t)(bytes / sizeof(T));
	uint32_t r = lane / E;
	uint32_t e = lane % E;
	for (uint32_t c = 0; c < elements; c += E) {
		if (valid) {
			for (uint32_t k = 0; k < E && c + k < elements; ++k)
				rows[lane * PAD + k] = local[c + k];
		}
		__syncwarp();
		for (uint32_t g = 0; g < 32; g += R) {
			uint32_t dst_record =
				__shfl_sync(0xffffffffu, record, (int)(g + r));
			int dst_valid = __shfl_sync(0xffffffffu, valid, (int)(g + r));
			if (dst_valid && c + e < elements) {
				T *dst = (T *)((uint8_t *)base +
					(size_t)dst_record * stride);
				dst[c + e] = rows[(g + r) * PAD + e];
			}
		}
		__syncwarp();
	}
}

typedef struct {
	__align__(16) uint8_t world[TMNF_WORLD_MAX_BYTES];
} ThreadWorld;

__device__ static void link_scratch(const DeviceEnv *env, uint32_t index)
{
	TmnfWorldLinkSources sources;
	sources.wheel_history = env->wheel_history + (size_t)index * 4;
	sources.collision_records =
		env->collision_records + (size_t)index * env->collision_capacity;
	for (uint32_t i = 0; i < TMNF_WORLD_CONTACT_BUFFER_COUNT; ++i) {
		sources.contact_records[i] = env->contact_records +
			((size_t)index * TMNF_WORLD_CONTACT_BUFFER_COUNT + i) *
				env->contact_capacity;
	}
	sources.replacements =
		env->replacements + (size_t)index * env->replacement_capacity;
	World_LinkScratch(env_world(env, index), &sources);
}

/* Assembles environment index's world in buffer. Called by every thread of
 * the block; valid = 0 for threads without an environment. */
__device__ static void world_enter(
	const DeviceEnv *env, uint32_t index, int valid, int initial, ThreadWorld *buffer)
{
	if (threadIdx.x == 0)
		block_world = (TmnfWorld *)buffer->world;
	__syncthreads();
	if ((TmnfWorld *)buffer->world != block_world)
		env_fail("local world address differs between threads");
	if ((uintptr_t)buffer->world != *env->template_base)
		env_fail("local world address differs from the template's");
	if (!valid)
		return;
	uint4 *target = (uint4 *)buffer->world;
	uint32_t vectors = (uint32_t)(env->world_size / sizeof(uint4));
	if (initial) {
		const uint4 *source = (const uint4 *)env->linked_template;
		for (uint32_t i = 0; i < vectors; ++i)
			target[i] = source[i];
		link_scratch(env, index);
	} else {
		for (uint32_t i = 0; i < vectors; ++i)
			target[i] = env->world_state[(size_t)i * env->count + index];
	}
}

__device__ static void world_leave(
	const DeviceEnv *env, uint32_t index, int valid, ThreadWorld *buffer)
{
	if (!valid)
		return;
	const uint4 *current = (const uint4 *)buffer->world;
	uint32_t vectors = (uint32_t)(env->world_size / sizeof(uint4));
	for (uint32_t i = 0; i < vectors; ++i)
		env->world_state[(size_t)i * env->count + index] = current[i];
}

/* Builds the linked template from the host world: one thread, once. The
 * result is env->linked_template, with pointers for this kernel's local
 * buffer address (recorded in env->template_base) and the scratch of
 * environment 0. */
__device__ static void init_template(const DeviceEnv *env, ThreadWorld *buffer)
{
	if (threadIdx.x == 0)
		block_world = (TmnfWorld *)buffer->world;
	__syncthreads();
	if (threadIdx.x != 0)
		return;
	const uint4 *source = (const uint4 *)env->template_world;
	uint4 *target = (uint4 *)buffer->world;
	uint32_t vectors = (uint32_t)(env->world_size / sizeof(uint4));
	for (uint32_t i = 0; i < vectors; ++i)
		target[i] = source[i];
	TmnfWorldLinkSources sources = env->link;
	sources.collision_records = env->collision_records;
	sources.collision_capacity = env->collision_capacity;
	for (uint32_t i = 0; i < TMNF_WORLD_CONTACT_BUFFER_COUNT; ++i) {
		sources.contact_records[i] =
			env->contact_records + (size_t)i * env->contact_capacity;
		sources.contact_capacities[i] = env->contact_capacity;
	}
	sources.replacements = env->replacements;
	sources.replacement_capacity = env->replacement_capacity;
	World_LinkPointers(env_world(env, 0), &sources);
	TMNFVehicleContactContext *contact =
		World_GetPhysicsWorld(env_world(env, 0))
			->corpora->vehicle_compute->contact;
	for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
		uint32_t body_ref = env->template_contact_refs[i];
		if (body_ref == TMNF_ENV_NO_CONTACT_BODY) {
			contact->wheels[i].contact_body = NULL;
			continue;
		}
		contact->wheels[i].contact_body =
			resolve_contact_body(env, 0, body_ref);
		if (contact->wheels[i].contact_body == NULL)
			env_fail("template contact body has no counterpart here");
	}
	uint4 *linked = (uint4 *)env->linked_template;
	for (uint32_t i = 0; i < vectors; ++i)
		linked[i] = target[i];
	*env->template_base = (uintptr_t)buffer->world;
}

/* ---- the kernel ----------------------------------------------------------- */

/* One kernel does everything so that the local world buffer has one
 * address (see world_enter). The host picks the job with KernelArgs. */
enum KernelKind {
	KERNEL_INIT_TEMPLATE,           /* 1 thread: init_template */
	KERNEL_INIT_HISTORY,
	KERNEL_CAPTURE_RESET,           /* 1 thread: reset snapshot of env 0 */
	KERNEL_RESET,                   /* in: mask or NULL; out: observations or NULL */
	KERNEL_STEP_RAW,                /* in: TMNFRaceInputs; out: observations or NULL */
	KERNEL_APPLY_INPUTS,            /* in: TMNFRaceInputs */
	KERNEL_ADVANCE,
	KERNEL_STEP_DISCRETE,           /* in: uint8_t actions; ticks = action_repeat */
	KERNEL_STEP_ANALOG,             /* in: TmnfAnalogAction; ticks = action_repeat */
	KERNEL_OBSERVE_GATES,
	KERNEL_CAPTURE,                 /* out: TmnfEnvSnapshot per slot */
	KERNEL_RESTORE,                 /* in: TmnfEnvSnapshot per slot; out: observations or NULL */
	KERNEL_EXPORT_WORLD,            /* 1 slot: out = the linked world bytes */
};

typedef struct {
	uint32_t kind;
	uint32_t count;                 /* slots (threads with work) */
	uint32_t ticks;                 /* tick_ms or action_repeat */
	const uint32_t *indices;        /* environment per slot, NULL: slot */
	const void *in;
	void *out;
 TmnfGateObservations *gates, *final_gates;
} KernelArgs;

template <typename Action>
__device__ static void step_rl_slot(
	const DeviceEnv *env, uint32_t index, int valid, const Action *actions,
	uint32_t action_repeat, TmnfGateObservations *gates,
 TmnfGateObservations *final_gates)
{
	__align__(16) TmnfStepResult result;
	StepOutputs outputs;
	if (valid)
		step_rl_one(env, index, &actions[index], action_repeat, &result,
			&outputs, gates ? &gates[index] : NULL,
   final_gates ? &final_gates[index] : NULL);
	WarpTile *tile = warp_tile();
	warp_scatter(env->results, sizeof(result), index, (const uint4 *)&result,
		sizeof(result), valid, tile);
	warp_scatter(env->flat_observations, sizeof(outputs.observation), index,
		(const uint32_t *)outputs.observation, sizeof(outputs.observation),
		valid, tile);
	warp_scatter(env->flat_final_observations, sizeof(outputs.final_observation),
		index, (const uint32_t *)outputs.final_observation,
		sizeof(outputs.final_observation), valid, tile);
	warp_scatter(env->flat_transitions, sizeof(outputs.transition), index,
		(const uint32_t *)outputs.transition, sizeof(outputs.transition),
		valid, tile);
}

__global__ static void tmnf_kernel(DeviceEnv env, KernelArgs args)
{
	ThreadWorld buffer;
	if (args.kind == KERNEL_INIT_TEMPLATE) {
		init_template(&env, &buffer);
		return;
	}
	uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
	int valid = slot < args.count;
	uint32_t index = !valid ? 0 : args.indices == NULL ? slot : args.indices[slot];
	world_enter(&env, index, valid, args.kind == KERNEL_INIT_HISTORY, &buffer);
	uint32_t mask = __ballot_sync(0xffffffffu, valid);
	if (valid)
		tmnf_dev_warp_begin(mask, env.warp_workspace +
			(size_t)(slot / 32) * env.warp_workspace_stride,
			env.contact_capacity);
	switch (args.kind) {
	case KERNEL_INIT_HISTORY:
		if (valid) {
			const uint4 *source = (const uint4 *)env.history_template;
			uint4 *target = (uint4 *)(env.wheel_history + (size_t)index * 4);
			for (uint32_t i = 0; i < 4 * sizeof(CSceneVehicleCarWheelHistory) / sizeof(uint4); ++i)
				target[i] = source[i];
		}
		break;
	case KERNEL_CAPTURE_RESET:
		if (valid)
			capture_physics(&env, index, env.reset_snapshot);
		break;
	case KERNEL_RESET:
		if (valid) {
			const uint8_t *mask = (const uint8_t *)args.in;
			if (mask == NULL || mask[index] != 0)
				reset_one(&env, index);
			if (args.out != NULL)
				write_rl_observation(&env, index,
					&((TmnfObservation *)args.out)[index]);
		}
		break;
	case KERNEL_STEP_RAW:
		if (valid) {
			const TMNFRaceInputs *inputs = (const TMNFRaceInputs *)args.in;
			physics_tick(&env, index, &inputs[index], args.ticks);
			if (args.out != NULL)
				write_physics_observation(player_corpus(&env, index),
					&((TmnfObservation *)args.out)[index]);
		}
		break;
	case KERNEL_APPLY_INPUTS:
		/* First half of physics_tick for tests/replay_tick.c, which
		 * encodes the world between the halves. */
		if (valid) {
			const TMNFRaceInputs *inputs = (const TMNFRaceInputs *)args.in;
			apply_respawn(&env, index, &inputs[index]);
			CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
				&inputs[index], player_corpus(&env, index)->vehicle);
		}
		break;
	case KERNEL_ADVANCE:
		/* Second half. With a route the pair also runs the harness's
		 * race tracker (TmnfRace_Step after the step until the finish),
		 * which is what arms the respawn location a later input may
		 * use. */
		if (valid) {
			TmnfPhysicsCorpus *corpus = player_corpus(&env, index);
			TmnfPhysicsWorld *physics =
				World_GetPhysicsWorld(env_world(&env, index));
			advance_timer(corpus, args.ticks);
			CHmsZoneDynamic_PhysicsStep2(physics, args.ticks);
			if (env.route != NULL && !env.race_states[index].finished) {
				(void)TmnfRace_Step(env.route, &env.race_states[index],
					physics->trigger_contacts,
					corpus->collision_corpus->live_iso);
			}
		}
		break;
	case KERNEL_STEP_DISCRETE:
		step_rl_slot(&env, index, valid, (const uint8_t *)args.in,
			args.ticks, args.gates, args.final_gates);
		break;
	case KERNEL_STEP_ANALOG:
		step_rl_slot(&env, index, valid, (const TmnfAnalogAction *)args.in,
			args.ticks, args.gates, args.final_gates);
		break;
	case KERNEL_OBSERVE_GATES:
  if (valid) write_gate_observation(&env, index, &args.gates[index]);
  break;
	case KERNEL_CAPTURE:
		if (valid)
			capture_env_state(&env, index, &((TmnfEnvSnapshot *)args.out)[slot]);
		break;
	case KERNEL_RESTORE:
		if (valid) {
			restore_env_state(&env, index,
				&((const TmnfEnvSnapshot *)args.in)[slot]);
			if (env.route != NULL) {
				/* Restoring starts a new transition boundary. Match the
				 * CPU binding: no reward or terminal data survives it. */
				TmnfStepResult *result = &env.results[index];
				memset(result, 0, sizeof(*result));
				write_rl_observation(&env, index, &result->observation);
				result->transition_discount = 1.0f;
				result->episode_id = env.episode_ids[index];
				flatten_observation(&result->observation,
					env.flat_observations + (size_t)index * TMNF_POLICY_OBSERVATION_WIDTH);
				memset(env.flat_final_observations + (size_t)index * TMNF_POLICY_OBSERVATION_WIDTH,
					0, TMNF_POLICY_OBSERVATION_WIDTH * sizeof(float));
				float *transition = env.flat_transitions + (size_t)index * TMNF_POLICY_TRANSITION_WIDTH;
				memset(transition, 0, TMNF_POLICY_TRANSITION_WIDTH * sizeof(float));
				transition[1] = 1.0f;
			}
			if (args.out != NULL)
				write_rl_observation(&env, index,
					&((TmnfObservation *)args.out)[slot]);
		}
		break;
	case KERNEL_EXPORT_WORLD:
		if (valid) {
			const uint4 *source = (const uint4 *)buffer.world;
			uint4 *target = (uint4 *)args.out;
			uint32_t vectors = (uint32_t)(env.world_size / sizeof(uint4));
			for (uint32_t i = 0; i < vectors; ++i)
				target[i] = source[i];
		}
		break;
	default:
		env_fail("unknown kernel kind");
	}
	world_leave(&env, index, valid, &buffer);
}

/* FMA canary for tests/cuda_no_fma.sh: plain a * b + c in float and double,
 * compiled and device-linked with the same flags as the physics. With
 * -fmad=false reaching the LTO backend it disassembles to FMUL + FADD and
 * DMUL + DADD; if the nvlink flags are ever lost it contracts to FFMA/DFMA
 * and the test fails before a replay has to. Never launched. */
__global__ void tmnf_fma_canary_kernel(
	const float *a, const double *b, float *out, double *dout)
{
	uint32_t i = threadIdx.x;
	out[i] = a[i] * a[i + 1] + a[i + 2];
	dout[i] = b[i] * b[i + 1] + b[i + 2];
}

/* Host and device compilers must agree on every shared layout. */
__global__ static void layout_kernel(uint64_t *sizes)
{
	sizes[0] = World_Size();
	sizes[1] = sizeof(TmnfEnvSnapshot);
	sizes[2] = sizeof(TmnfStepResult);
	sizes[3] = sizeof(TmnfObservation);
	sizes[4] = sizeof(TmnfTrack);
	sizes[5] = sizeof(TmnfRoute);
	sizes[6] = sizeof(TMNFRaceInputs);
	sizes[7] = sizeof(TmnfWorldLinkSources);
	sizes[8] = World_ColdSize();
}

/* ------------------------------------------------------------------------ */
/* Host API                                                                   */
/* ------------------------------------------------------------------------ */

/* The libm hard-case table (src/cuda/tmnf_libm_hardcases.h) corrects the
 * device's sin/cos/exp/atan2 to glibc's roundings for the glibc and CUDA
 * toolkit it was generated with. A different glibc changes the host side of
 * every replay and lockstep; a different toolkit changes libdevice. Both
 * must match the table, or the table must be regenerated
 * (tools/cuda_libm_hardcases.cu). */
static_assert(CUDART_VERSION == TMNF_LIBM_CUDART_VERSION,
	"CUDA toolkit differs from the libm hard-case table's: regenerate "
	"src/cuda/tmnf_libm_hardcases.h with cuda_libm_hardcases generate");

static void check_libm_pin(void)
{
	const char *glibc = gnu_get_libc_version();
	if (strcmp(glibc, TMNF_LIBM_GLIBC_VERSION) != 0) {
		fprintf(stderr, "tmnf cuda env: glibc %s, libm hard-case table is for "
			"glibc %s: regenerate src/cuda/tmnf_libm_hardcases.h\n",
			glibc, TMNF_LIBM_GLIBC_VERSION);
		abort();
	}
	int runtime = 0;
	check(cudaRuntimeGetVersion(&runtime), "runtime version");
	if (runtime != TMNF_LIBM_CUDART_VERSION) {
		fprintf(stderr, "tmnf cuda env: CUDA runtime %d, libm hard-case table "
			"is for %d: regenerate src/cuda/tmnf_libm_hardcases.h\n",
			runtime, TMNF_LIBM_CUDART_VERSION);
		abort();
	}
}

TmnfCudaVecEnvLimits TmnfCudaVecEnv_DefaultLimits(void)
{
	TmnfCudaVecEnvLimits limits;
	limits.collision_capacity = 256;
	limits.contact_capacity = 64;
	limits.replacement_capacity = 256;
	limits.threads_per_block = 64;
	limits.stack_bytes = TMNF_CUDA_STACK_BYTES;
	/* The sm_86 compiler needs a larger frame than sm_120. Native users
	 * must get the same usable default as the Python binding. */
	int device;
	cudaDeviceProp properties;
	if (cudaGetDevice(&device) == cudaSuccess &&
		cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
		properties.major == 8 && properties.minor == 6)
		limits.stack_bytes = 40960;
	limits.stream = NULL;
	return limits;
}

static uint32_t blocks_for(uint32_t count, uint32_t threads)
{
	return (count + threads - 1) / threads;
}

static void sync(TmnfCudaVecEnv *env, const char *what)
{
	check(cudaGetLastError(), what);
	check(cudaStreamSynchronize(env->stream), what);
}

/* Host <-> device copies ordered on the env's stream. A download returns
 * with the bytes in place; an upload's source may be reused on return (the
 * copy is enqueued before the launch that consumes it, and every launch is
 * followed by a stream sync). */
static void upload(TmnfCudaVecEnv *env, void *destination,
	const void *source, size_t bytes, const char *what)
{
	check(cudaMemcpyAsync(destination, source, bytes,
		cudaMemcpyHostToDevice, env->stream), what);
}

static void download(TmnfCudaVecEnv *env, void *destination,
	const void *source, size_t bytes, const char *what)
{
	check(cudaMemcpyAsync(destination, source, bytes,
		cudaMemcpyDeviceToHost, env->stream), what);
	sync(env, what);
}

/* Runs tmnf_kernel over `count` slots and waits for it. */
static void launch(
	TmnfCudaVecEnv *env, enum KernelKind kind, uint32_t count,
	uint32_t ticks, const uint32_t *indices, const void *in, void *out,
	const char *what, TmnfGateObservations *gates = NULL,
 TmnfGateObservations *final_gates = NULL)
{
	KernelArgs args;
 args.gates = gates; args.final_gates = final_gates;
	args.kind = (uint32_t)kind;
	args.count = count;
	args.ticks = ticks;
	args.indices = indices;
	args.in = in;
	args.out = out;
	uint32_t threads = env->threads_per_block;
	tmnf_kernel<<<blocks_for(count, threads), threads, warp_tile_bytes(threads),
		env->stream>>>(env->device, args);
	sync(env, what);
}

/* vec_env.c: validate_config, minimum_finish_ticks, derive_pace_ticks,
 * route_floor_y, resolve_race_budget. */
static void validate_config(const TmnfVecEnvConfig *config)
{
	if (config == NULL ||
		config->off_track_grace_ticks == 0 ||
		config->stuck_grace_ticks == 0 ||
		!(config->discount_per_tick > 0.0f) ||
		!(config->discount_per_tick <= 1.0f) ||
		!(config->reference_speed > 0.0f) ||
		!(config->stuck_progress_epsilon > 0.0f) ||
		!isfinite(config->discount_per_tick) ||
		!isfinite(config->reference_speed) ||
		!isfinite(config->stuck_progress_epsilon) ||
		(config->autoreset_mode != TMNF_AUTORESET_SAME_STEP &&
		 config->autoreset_mode != TMNF_AUTORESET_NEXT_STEP) ||
		(config->action_space != TMNF_ACTION_SPACE_DISCRETE &&
		 config->action_space != TMNF_ACTION_SPACE_ANALOG)) {
		cuda_fail("invalid configuration");
	}
}

static uint32_t minimum_finish_ticks(const TmnfRoute *route)
{
	double total_length = (double)TmnfRoute_GetReferenceLength(route) *
		route->metadata->lap_count;
	double ticks = ceil(total_length / TMNF_MAX_LINEAR_SPEED_MPS /
		(TMNF_RACE_TICK_MS / 1000.0));
	if (ticks < 1.0)
		ticks = 1.0;
	return (uint32_t)ticks;
}

static uint32_t derive_pace_ticks(
	const TmnfRoute *route, const TmnfVecEnvConfig *config)
{
	double total_length = (double)TmnfRoute_GetReferenceLength(route) *
		route->metadata->lap_count;
	double ticks = ceil(
		TMNF_HORIZON_PACE_FACTOR * total_length /
		(double)config->reference_speed /
		(TMNF_RACE_TICK_MS / 1000.0));
	if (ticks > (double)UINT32_MAX)
		cuda_fail("derived race budget overflows");
	return (uint32_t)ticks;
}

static float route_floor_y(const TmnfRoute *route)
{
	uint32_t count = TmnfRoute_GetReferencePointCount(route);
	const TmnfRouteReferencePoint *points =
		TmnfRoute_GetReferencePoints(route);
	if (count == 0)
		cuda_fail("route has no centerline samples");
	float floor_y = points[0].position.y;
	for (uint32_t i = 1; i < count; ++i) {
		if (points[i].position.y < floor_y)
			floor_y = points[i].position.y;
	}
	if (!isfinite(floor_y))
		cuda_fail("route centerline height is not finite");
	return floor_y;
}

static void resolve_race_budget(TmnfVecEnvConfig *config, const TmnfRoute *route)
{
	uint32_t minimum = minimum_finish_ticks(route);
	uint32_t pace = derive_pace_ticks(route, config);
	if (config->max_race_ticks == 0)
		config->max_race_ticks = pace;
	if (config->horizon_ticks == 0)
		config->horizon_ticks = pace;
	if (config->max_race_ticks < minimum)
		cuda_fail("race timeout makes a finish impossible");
	if (config->horizon_ticks < minimum)
		cuda_fail("collection horizon makes a finish impossible");
}

static void validate_spawn(const TmnfWorld *world, const TmnfRoute *route)
{
	const CHmsStateDyna *spawn = World_GetPlayerState(world);
	const TmnfRouteInitialState *initial = &route->start->initial_state;
	if (memcmp(&spawn->pos, &initial->pos, sizeof(initial->pos)) != 0 ||
		memcmp(&spawn->rot, &initial->rot, sizeof(initial->rot)) != 0) {
		cuda_fail("world spawn does not match the route start");
	}
}

static void check_layouts(void)
{
	uint64_t *device_sizes = (uint64_t *)device_alloc(9 * sizeof(uint64_t));
	layout_kernel<<<1, 1>>>(device_sizes);
	check(cudaGetLastError(), "layout kernel");
	check(cudaDeviceSynchronize(), "layout kernel");
	uint64_t sizes[9];
	check(cudaMemcpy(sizes, device_sizes, sizeof(sizes),
		cudaMemcpyDeviceToHost), "layout copy");
	check(cudaFree(device_sizes), "layout free");
	if (sizes[0] != World_Size() ||
		sizes[1] != sizeof(TmnfEnvSnapshot) ||
		sizes[2] != sizeof(TmnfStepResult) ||
		sizes[3] != sizeof(TmnfObservation) ||
		sizes[4] != sizeof(TmnfTrack) ||
		sizes[5] != sizeof(TmnfRoute) ||
		sizes[6] != sizeof(TMNFRaceInputs) ||
		sizes[7] != sizeof(TmnfWorldLinkSources) ||
		sizes[8] != World_ColdSize()) {
		cuda_fail("host and device struct layouts differ");
	}
}

/* Fills a zeroed env; any device error unwinds to TmnfCudaVecEnv_Create. */
static void create_into(
	TmnfCudaVecEnv *env,
	const TmnfTrack *track,
	const TmnfWorld *template_world,
	const TmnfRoute *route,
	uint32_t count,
	const TmnfVecEnvConfig *config,
	const TmnfCudaVecEnvLimits *limits)
{
	DeviceEnv *device = &env->device;
	device->count = count;
	env->threads_per_block = limits->threads_per_block;
	device->config = *config;

	/* Immutable data. */
	env->track = upload_track(track);
	uint32_t blob_size;
	const uint8_t *blob = World_GetVehicleBlob(template_world, &blob_size);
	env->vehicle_blob = (uint8_t *)device_upload(blob, blob_size);
	uint32_t bound_count;
	const float *bounds = World_GetCurveBounds(template_world, &bound_count);
	env->curve_bounds = (float *)device_upload(
		bounds, (size_t)bound_count * sizeof(float));
	env->fake_contact_mask = (uint8_t *)device_upload(
		TMNF_FAKE_CONTACT_MASK, sizeof(TMNF_FAKE_CONTACT_MASK));
	{
		float tables[12];
		memcpy(tables, TMNF_WATER_IMPULSE_POSITIONS, 4 * sizeof(float));
		memcpy(tables + 4, TMNF_WATER_IMPULSE_VERTICAL_VALUES,
			4 * sizeof(float));
		memcpy(tables + 8, TMNF_WATER_IMPULSE_HORIZONTAL_VALUES,
			4 * sizeof(float));
		env->water_tables = (float *)device_upload(tables, sizeof(tables));
	}
	memset(&device->link, 0, sizeof(device->link));
	device->link.vehicle_blob = env->vehicle_blob;
	device->link.track = env->track.track;
	device->link.curve_bounds = env->curve_bounds;
	device->link.fake_contact_mask = env->fake_contact_mask;
	device->link.water_impulse_positions = env->water_tables;
	device->link.water_impulse_vertical_values = env->water_tables + 4;
	device->link.water_impulse_horizontal_values = env->water_tables + 8;

	/* Per-environment worlds and scratch. */
	device->collision_capacity = limits->collision_capacity;
	device->contact_capacity = limits->contact_capacity;
	device->replacement_capacity = limits->replacement_capacity;
	device->world_size = World_Size();
	device->template_world =
		(const TmnfWorld *)device_upload(template_world, World_Size());
	device->cold = device_upload(World_GetCold(template_world), World_ColdSize());
	device->link.cold = device->cold;
	/* The physics step tests the route's triggers, so the template links
	 * the device route before any world is assembled from it. Without a
	 * route (plain replays) the fields stay NULL and no trigger is tested. */
	if (route != NULL) {
		env->route = upload_route(route);
		device->route = env->route.route;
		device->link.route = device->route;
	}
	{
		const TmnfPhysicsWorld *physics =
			World_GetPhysicsWorldConst(template_world);
		const TMNFVehicleContactContext *contact =
			physics->corpora->vehicle_compute->contact;
		for (uint32_t i = 0; i < TMNF_STADIUM_WHEEL_COUNT; ++i) {
			const CHmsResponseBody *body = contact->wheels[i].contact_body;
			if (body == NULL) {
				device->template_contact_refs[i] = TMNF_ENV_NO_CONTACT_BODY;
				continue;
			}
			if (physics->response_zone.resolve_body(
					physics->response_zone.resolver_user,
					body->corpus_ref) != body)
				cuda_fail("template contact body is not addressable by its token");
			device->template_contact_refs[i] = body->corpus_ref;
		}
	}
	device->linked_template = (TmnfWorld *)device_alloc(World_Size());
	device->template_base = (uintptr_t *)device_alloc(sizeof(uintptr_t));
	{
		TmnfWorldLinkSources source;
		World_GetLinkSources(template_world, &source);
		device->history_template = (const CSceneVehicleCarWheelHistory *)device_upload(
			source.wheel_history, 4 * sizeof(CSceneVehicleCarWheelHistory));
		device->wheel_history = (CSceneVehicleCarWheelHistory *)device_alloc(
			(size_t)count * 4 * sizeof(CSceneVehicleCarWheelHistory));
		device->link.wheel_history = device->wheel_history;
	}
	device->world_state = (uint4 *)device_alloc((size_t)count * World_Size());
	device->collision_records = (SHmsPhysicalCollision *)device_alloc(
		(size_t)count * device->collision_capacity *
			sizeof(SHmsPhysicalCollision));
	device->contact_records = (SHmsPhysicalCollision *)device_alloc(
		(size_t)count * TMNF_WORLD_CONTACT_BUFFER_COUNT *
			device->contact_capacity * sizeof(SHmsPhysicalCollision));
	device->replacements = (GmVec3 *)device_alloc(
		(size_t)count * device->replacement_capacity * sizeof(GmVec3));
	device->warp_workspace_stride =
		tmnf_dev_warp_workspace_bytes(device->contact_capacity);
	device->warp_workspace = (uint8_t *)device_alloc(
		(size_t)blocks_for(count, env->threads_per_block) *
			(env->threads_per_block / 32) * device->warp_workspace_stride);
	launch(env, KERNEL_INIT_TEMPLATE, 1, 0, NULL, NULL, NULL, "template kernel");
	launch(env, KERNEL_INIT_HISTORY, count, 0, NULL, NULL, NULL, "history initialization");

	/* Argument staging. */
	size_t action_bytes = sizeof(TMNFRaceInputs);
	if (action_bytes < sizeof(TmnfAnalogAction))
		action_bytes = sizeof(TmnfAnalogAction);
	env->device_actions = device_alloc((size_t)count * action_bytes);
	env->device_observations = (TmnfObservation *)device_alloc(
		(size_t)count * sizeof(TmnfObservation));
	env->device_indices = (uint32_t *)device_alloc((size_t)count * sizeof(uint32_t));
	env->device_mask = (uint8_t *)device_alloc(count);
	env->device_world = (TmnfWorld *)device_alloc(World_Size());

	if (route == NULL)
		return;

	/* RL layer. */
	env->has_route = 1;
	resolve_race_budget(&device->config, route);
	device->fell_floor_y = route_floor_y(route) - TMNF_FELL_MARGIN_METERS;
	validate_spawn(template_world, route);
	env->lookahead_meters = (float *)device_upload(
		TMNF_OBSERVATION_LOOKAHEAD_METERS,
		sizeof(float) * TMNF_OBSERVATION_LOOKAHEAD_COUNT);
	device->lookahead_meters = env->lookahead_meters;
	{
		uint32_t powers = device->config.max_race_ticks + 1;
		float *table = (float *)malloc((size_t)powers * sizeof(float));
		if (table == NULL)
			cuda_fail("out of memory");
		for (uint32_t k = 0; k < powers; ++k)
			table[k] = powf(device->config.discount_per_tick, (float)k);
		env->gamma_powers =
			(float *)device_upload(table, (size_t)powers * sizeof(float));
		free(table);
	}
	device->gamma_powers = env->gamma_powers;
	device->reset_snapshot =
		(TmnfEnvSnapshot *)device_alloc(sizeof(TmnfEnvSnapshot));
	device->race_states =
		(TmnfRaceState *)device_alloc((size_t)count * sizeof(TmnfRaceState));
	device->episode_ids =
		(uint64_t *)device_alloc((size_t)count * sizeof(uint64_t));
	device->episode_returns =
		(float *)device_alloc((size_t)count * sizeof(float));
	device->reset_pending = (uint8_t *)device_alloc(count);
	device->results = (TmnfStepResult *)device_alloc(
		(size_t)count * sizeof(TmnfStepResult));
	device->flat_observations = (float *)device_alloc(
		(size_t)count * TMNF_POLICY_OBSERVATION_WIDTH * sizeof(float));
	device->flat_final_observations = (float *)device_alloc(
		(size_t)count * TMNF_POLICY_OBSERVATION_WIDTH * sizeof(float));
	device->flat_transitions = (float *)device_alloc(
		(size_t)count * TMNF_POLICY_TRANSITION_WIDTH * sizeof(float));

	launch(env, KERNEL_CAPTURE_RESET, 1, 0, NULL, NULL, NULL,
		"reset snapshot capture");
	/* Creation used the legacy stream (uploads, clears); everything after
	 * this runs on env->stream, which may not be ordered behind it. */
	check(cudaDeviceSynchronize(), "create");
	TmnfCudaVecEnv_Reset(env, NULL, NULL);
}

static void destroy_env(TmnfCudaVecEnv *env, bool best_effort);

TmnfCudaVecEnv *TmnfCudaVecEnv_Create(
	const TmnfTrack *track,
	const TmnfWorld *template_world,
	const TmnfRoute *route,
	uint32_t count,
	const TmnfVecEnvConfig *config,
	const TmnfCudaVecEnvLimits *limits)
{
	if (track == NULL || template_world == NULL || count == 0 ||
		config == NULL || limits == NULL) {
		cuda_fail("create argument is null or empty");
	}
	if (World_GetTrack(template_world) != track)
		cuda_fail("template world was not built on this track");
	if (limits->collision_capacity == 0 || limits->contact_capacity == 0 ||
		limits->replacement_capacity == 0 ||
		limits->threads_per_block == 0 ||
		limits->threads_per_block % 32 != 0 ||
		limits->threads_per_block > 32 * TMNF_WARPS_PER_BLOCK_MAX ||
		limits->stack_bytes == 0) {
		cuda_fail("invalid limits");
	}
	validate_config(config);
	if (create_unwind != NULL)
		cuda_fail("create is not reentrant");
	TmnfCudaVecEnv *env = (TmnfCudaVecEnv *)calloc(1, sizeof(*env));
	if (env == NULL)
		cuda_fail("out of memory");
	jmp_buf unwind;
	if (setjmp(unwind) != 0) {
		create_unwind = NULL;
		fprintf(stderr, "tmnf cuda env: create failed: %s\n", last_error);
		destroy_env(env, true);
		return NULL;
	}
	create_unwind = &unwind;
	env->stream = limits->stream;
	check(cudaDeviceSetLimit(cudaLimitStackSize, limits->stack_bytes),
		"stack size limit");
	check_layouts();
	check_libm_pin();
	create_into(env, track, template_world, route, count, config, limits);
	create_unwind = NULL;
	return env;
}

static void destroy_env(TmnfCudaVecEnv *env, bool best_effort)
{
	if (env == NULL)
		return;
	void *blocks[] = {
		env->track.mapping, env->track.track,
		env->route.mapping, env->route.bvh, env->route.route,
		env->vehicle_blob, env->curve_bounds, env->fake_contact_mask,
		env->water_tables, env->lookahead_meters, env->gamma_powers,
		(void *)env->device.template_world, env->device.cold,
		env->device.linked_template,
		env->device.template_base, env->device.world_state,
		env->device.wheel_history, (void *)env->device.history_template,
		env->device.collision_records, env->device.contact_records,
		env->device.replacements, env->device.warp_workspace,
		env->device.reset_snapshot,
		env->device.race_states, env->device.episode_ids,
		env->device.episode_returns, env->device.reset_pending,
		env->device.results, env->device.flat_observations,
		env->device.flat_final_observations, env->device.flat_transitions,
		env->device_actions, env->device_snapshots,
		env->device_observations, env->device_indices, env->device_mask,
		env->device_world,
	};
	for (size_t i = 0; i < sizeof(blocks) / sizeof(blocks[0]); ++i) {
		if (blocks[i] != NULL) {
			cudaError_t error = cudaFree(blocks[i]);
			if (!best_effort) check(error, "cudaFree");
		}
	}
	for (uint32_t i = 0; i < env->track.block_count; ++i) {
		cudaError_t error = cudaFree(env->track.blocks[i]);
		if (!best_effort) check(error, "cudaFree");
	}
	free(env->track.blocks);
	free(env);
}

void TmnfCudaVecEnv_Destroy(TmnfCudaVecEnv *env)
{
	destroy_env(env, false);
}

uint32_t TmnfCudaVecEnv_Count(const TmnfCudaVecEnv *env)
{
	return env->device.count;
}

static void require_route(const TmnfCudaVecEnv *env)
{
	if (env == NULL)
		cuda_fail("environment is null");
	if (!env->has_route)
		cuda_fail("RL entry point used without a route");
}

void TmnfCudaVecEnv_Step(
	TmnfCudaVecEnv *env, const TMNFRaceInputs *inputs, uint32_t tick_ms,
	TmnfObservation *observations)
{
	if (env == NULL || inputs == NULL || tick_ms == 0)
		cuda_fail("physics step argument is invalid");
	uint32_t count = env->device.count;
	upload(env, env->device_actions, inputs,
		(size_t)count * sizeof(TMNFRaceInputs), "inputs upload");
	launch(env, KERNEL_STEP_RAW, count, tick_ms, NULL, env->device_actions,
		observations == NULL ? NULL : env->device_observations,
		"raw step kernel");
	if (observations != NULL) {
		download(env, observations, env->device_observations,
			(size_t)count * sizeof(TmnfObservation),
			"observations download");
	}
}

void TmnfCudaVecEnv_StepDevice(
	TmnfCudaVecEnv *env, const TMNFRaceInputs *device_inputs, uint32_t tick_ms)
{
	if (env == NULL || device_inputs == NULL || tick_ms == 0)
		cuda_fail("device physics step argument is invalid");
	launch(env, KERNEL_STEP_RAW, env->device.count, tick_ms, NULL,
		device_inputs, NULL, "raw device step kernel");
}

void TmnfCudaVecEnv_StepWithPreStateCopy(
	TmnfCudaVecEnv *env, const TMNFRaceInputs *inputs, uint32_t tick_ms,
	uint32_t index, TmnfWorld *destination)
{
	if (env == NULL || inputs == NULL || tick_ms == 0)
		cuda_fail("physics step argument is invalid");
	uint32_t count = env->device.count;
	upload(env, env->device_actions, inputs,
		(size_t)count * sizeof(TMNFRaceInputs), "inputs upload");
	launch(env, KERNEL_APPLY_INPUTS, count, 0, NULL, env->device_actions, NULL,
		"apply inputs kernel");
	TmnfCudaVecEnv_CopyWorldToHost(env, index, destination);
	launch(env, KERNEL_ADVANCE, count, tick_ms, NULL, NULL, NULL,
		"advance kernel");
}

void TmnfCudaVecEnv_Reset(
	TmnfCudaVecEnv *env, const uint8_t *mask, TmnfObservation *observations)
{
	require_route(env);
	uint32_t count = env->device.count;
	if (mask != NULL) {
		upload(env, env->device_mask, mask, count, "mask upload");
	}
	launch(env, KERNEL_RESET, count, 0, NULL,
		mask == NULL ? NULL : env->device_mask,
		observations == NULL ? NULL : env->device_observations,
		"reset kernel");
	if (observations != NULL) {
		download(env, observations, env->device_observations,
			(size_t)count * sizeof(TmnfObservation),
			"observations download");
	}
}

static void download_results(TmnfCudaVecEnv *env, TmnfStepResult *results)
{
	if (results == NULL)
		return;
	download(env, results, env->device.results,
		(size_t)env->device.count * sizeof(TmnfStepResult),
		"results download");
}

void TmnfCudaVecEnv_StepDiscreteDevice(
	TmnfCudaVecEnv *env, const uint8_t *device_actions,
	uint32_t action_repeat)
{
	require_route(env);
	if (device_actions == NULL || action_repeat == 0)
		cuda_fail("RL step argument is invalid");
	if (env->device.config.action_space != TMNF_ACTION_SPACE_DISCRETE)
		cuda_fail("discrete step used with analog action space");
	launch(env, KERNEL_STEP_DISCRETE, env->device.count, action_repeat, NULL,
		device_actions, NULL, "discrete step kernel");
}

void TmnfCudaVecEnv_StepDiscrete(
	TmnfCudaVecEnv *env, const uint8_t *actions, uint32_t action_repeat,
	TmnfStepResult *results)
{
	require_route(env);
	if (actions == NULL)
		cuda_fail("RL step argument is invalid");
	upload(env, env->device_actions, actions, env->device.count,
		"actions upload");
	TmnfCudaVecEnv_StepDiscreteDevice(
		env, (const uint8_t *)env->device_actions, action_repeat);
	download_results(env, results);
}

void TmnfCudaVecEnv_StepAnalogDevice(
	TmnfCudaVecEnv *env, const TmnfAnalogAction *device_actions,
	uint32_t action_repeat)
{
	require_route(env);
	if (device_actions == NULL || action_repeat == 0)
		cuda_fail("analog step argument is invalid");
	if (env->device.config.action_space != TMNF_ACTION_SPACE_ANALOG)
		cuda_fail("analog step used with discrete action space");
	launch(env, KERNEL_STEP_ANALOG, env->device.count, action_repeat, NULL,
		device_actions, NULL, "analog step kernel");
}

void TmnfCudaVecEnv_StepAnalog(
	TmnfCudaVecEnv *env, const TmnfAnalogAction *actions,
	uint32_t action_repeat, TmnfStepResult *results)
{
	require_route(env);
	if (actions == NULL)
		cuda_fail("analog step argument is invalid");
	for (uint32_t i = 0; i < env->device.count; ++i) {
		(void)TmnfVecEnv_QuantizeAnalogSteer(actions[i].steer);
		if (actions[i].gas > 1 || actions[i].brake > 1)
			cuda_fail("analog gas and brake must be binary");
	}
	upload(env, env->device_actions, actions,
		(size_t)env->device.count * sizeof(TmnfAnalogAction),
		"actions upload");
	TmnfCudaVecEnv_StepAnalogDevice(
		env, (const TmnfAnalogAction *)env->device_actions, action_repeat);
	download_results(env, results);
}

void TmnfCudaVecEnv_StepDiscreteDeviceWithGates(TmnfCudaVecEnv *env,
 const uint8_t *actions, uint32_t repeat,
 TmnfGateObservations *gates, TmnfGateObservations *final_gates)
{
 require_route(env);
 if (!actions || !repeat || !gates || !final_gates || gates == final_gates)
  cuda_fail("invalid device gate step arguments");
 if (env->device.config.action_space != TMNF_ACTION_SPACE_DISCRETE)
  cuda_fail("gate step action space mismatch");
 launch(env, KERNEL_STEP_DISCRETE, env->device.count, repeat, NULL,
  actions, NULL, "gate step kernel", gates, final_gates);
}

void TmnfCudaVecEnv_StepAnalogDeviceWithGates(TmnfCudaVecEnv *env,
 const TmnfAnalogAction *actions, uint32_t repeat,
 TmnfGateObservations *gates, TmnfGateObservations *final_gates)
{
 require_route(env);
 if (!actions || !repeat || !gates || !final_gates || gates == final_gates)
  cuda_fail("invalid device gate step arguments");
 if (env->device.config.action_space != TMNF_ACTION_SPACE_ANALOG)
  cuda_fail("gate step action space mismatch");
 launch(env, KERNEL_STEP_ANALOG, env->device.count, repeat, NULL,
  actions, NULL, "gate step kernel", gates, final_gates);
}

void TmnfCudaVecEnv_ObserveGatesDevice(TmnfCudaVecEnv *env,
 TmnfGateObservations *gates)
{
 require_route(env);
 if (!gates) cuda_fail("device gates are null");
 launch(env, KERNEL_OBSERVE_GATES, env->device.count, 0, NULL,
  NULL, NULL, "gate observation kernel", gates);
}

const TmnfStepResult *TmnfCudaVecEnv_DeviceResults(const TmnfCudaVecEnv *env)
{
	require_route(env);
	return env->device.results;
}

const float *TmnfCudaVecEnv_DeviceObservations(const TmnfCudaVecEnv *env)
{
	require_route(env);
	return env->device.flat_observations;
}

const float *TmnfCudaVecEnv_DeviceFinalObservations(
	const TmnfCudaVecEnv *env)
{
	require_route(env);
	return env->device.flat_final_observations;
}

const float *TmnfCudaVecEnv_DeviceTransitions(const TmnfCudaVecEnv *env)
{
	require_route(env);
	return env->device.flat_transitions;
}

static void validate_snapshot_indices(
	const TmnfCudaVecEnv *env, const uint32_t *indices, uint32_t count)
{
	if (indices == NULL || count == 0 || count > env->device.count)
		cuda_fail("snapshot indices are null, empty or too many");
	uint8_t *seen = (uint8_t *)calloc(env->device.count, 1);
	if (seen == NULL)
		cuda_fail("out of memory");
	for (uint32_t i = 0; i < count; ++i) {
		if (indices[i] >= env->device.count)
			cuda_fail("snapshot index is out of range");
		if (seen[indices[i]])
			cuda_fail("snapshot indices contain a duplicate");
		seen[indices[i]] = 1;
	}
	free(seen);
}

/* Snapshot staging grows only to the largest requested capture/restore.
 * Ordinary training does not reserve count * 4,976 bytes it never reads. */
static void reserve_snapshots(TmnfCudaVecEnv *env, uint32_t count)
{
	if (count <= env->snapshot_capacity)
		return;
	/* Previous uploads may still be using this buffer on the env stream. */
	check(cudaStreamSynchronize(env->stream), "snapshot staging resize");
	check(cudaFree(env->device_snapshots), "snapshot staging free");
	env->device_snapshots = NULL;
	env->snapshot_capacity = 0;
	env->device_snapshots = (TmnfEnvSnapshot *)device_alloc((size_t)count * sizeof(TmnfEnvSnapshot));
	env->snapshot_capacity = count;
}

void TmnfCudaVecEnv_CaptureIndices(
	TmnfCudaVecEnv *env, const uint32_t *indices, uint32_t count,
	TmnfEnvSnapshot *snapshots)
{
	require_route(env);
	if (snapshots == NULL)
		cuda_fail("capture argument is null");
	const uint32_t *device_indices = NULL;
	if (indices != NULL) {
		validate_snapshot_indices(env, indices, count);
		upload(env, env->device_indices, indices,
			(size_t)count * sizeof(uint32_t), "indices upload");
		device_indices = env->device_indices;
	} else {
		count = env->device.count;
	}
	reserve_snapshots(env, count);
	launch(env, KERNEL_CAPTURE, count, 0, device_indices, NULL,
		env->device_snapshots, "capture kernel");
	download(env, snapshots, env->device_snapshots,
		(size_t)count * sizeof(TmnfEnvSnapshot), "snapshots download");
}

void TmnfCudaVecEnv_Capture(TmnfCudaVecEnv *env, TmnfEnvSnapshot *snapshots)
{
	TmnfCudaVecEnv_CaptureIndices(env, NULL, 0, snapshots);
}

void TmnfCudaVecEnv_RestoreIndices(
	TmnfCudaVecEnv *env, const uint32_t *indices, uint32_t count,
	const TmnfEnvSnapshot *snapshots, TmnfObservation *observations)
{
	require_route(env);
	if (snapshots == NULL)
		cuda_fail("restore argument is null");
	const uint32_t *device_indices = NULL;
	if (indices != NULL) {
		validate_snapshot_indices(env, indices, count);
		upload(env, env->device_indices, indices,
			(size_t)count * sizeof(uint32_t), "indices upload");
		device_indices = env->device_indices;
	} else {
		count = env->device.count;
	}
	reserve_snapshots(env, count);
	upload(env, env->device_snapshots, snapshots,
		(size_t)count * sizeof(TmnfEnvSnapshot), "snapshots upload");
	launch(env, KERNEL_RESTORE, count, 0, device_indices, env->device_snapshots,
		observations == NULL ? NULL : env->device_observations,
		"restore kernel");
	if (observations != NULL) {
		download(env, observations, env->device_observations,
			(size_t)count * sizeof(TmnfObservation),
			"observations download");
	}
}

void TmnfCudaVecEnv_Restore(
	TmnfCudaVecEnv *env, const TmnfEnvSnapshot *snapshots)
{
	TmnfCudaVecEnv_RestoreIndices(env, NULL, 0, snapshots, NULL);
}

void TmnfCudaVecEnv_CopyWorldToHost(
	TmnfCudaVecEnv *env, uint32_t index, TmnfWorld *destination)
{
	if (env == NULL || destination == NULL || index >= env->device.count)
		cuda_fail("world copy argument is invalid");
	TmnfWorldLinkSources sources;
	World_GetLinkSources(destination, &sources);
	upload(env, env->device_indices, &index, sizeof(index), "index upload");
	launch(env, KERNEL_EXPORT_WORLD, 1, 0, env->device_indices, NULL,
		env->device_world, "world export kernel");
	download(env, destination, env->device_world, World_Size(),
		"world download");
	download(env, sources.wheel_history, env->device.wheel_history + (size_t)index * 4,
		4 * sizeof(CSceneVehicleCarWheelHistory), "wheel history download");
	World_LinkPointers(destination, &sources);
}
