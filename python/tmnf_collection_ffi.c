#include "tmnf_collection_ffi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <math.h>

#include "race.h"
#include "world.h"
#if TMNF_COLLECTION_CUDA
#include "cuda/tmnf_cuda_env.h"
#endif

typedef struct {
	TmnfTrack *track;
	TmnfRoute *route;
	TmnfTrack *trigger_geometry;
	char *vehicle_path;
	atomic_uint references;
} CollectionAssets;

struct TmnfCollection {
	CollectionAssets *assets;
	TmnfWorld *world;
	TmnfRaceState race;
	uint32_t ticks;
	uint32_t respawns;
	float replay_previous[3];
	int replay_has_previous;
	float sync_input_steer;
	uint32_t sync_contact_forward, previous_sync_contact_forward;
	float sync_contact_side, previous_sync_contact_side;
	int cuda;
	int initialize_from_graph;
	uint32_t validation_seed;
	int has_initial_position;
	int replay_mode;
	uint32_t lap_count; /* zero selects the immutable map default */
	float initial_position[3];
	uint32_t *checkpoint_times;
	uint32_t checkpoint_time_count;
	uint32_t checkpoint_time_capacity;
#if TMNF_COLLECTION_CUDA
	TmnfCudaVecEnv *device;
#endif
};

static _Thread_local char collection_error[256];
static void update_legacy_sync_contact(TmnfCollection *env);
static int append_checkpoint_times(TmnfCollection *env, uint32_t accepted,
	uint32_t race_time_ms);

static int error(const char *message)
{
	snprintf(collection_error, sizeof(collection_error), "%s", message);
	return 0;
}

const char *tmnf_collection_error(void) { return collection_error; }
size_t tmnf_collection_frame_size(void) { return sizeof(TmnfCollectionFrame); }

void tmnf_collection_destroy(TmnfCollection *env)
{
	if (!env) return;
#if TMNF_COLLECTION_CUDA
	if (env->device) TmnfCudaVecEnv_Destroy(env->device);
#endif
	if (env->world) World_Destroy(env->world);
	free(env->checkpoint_times);
	CollectionAssets *assets = env->assets;
	if (assets && atomic_fetch_sub(&assets->references, 1) == 1) {
		if (assets->route) TmnfRoute_Unload(assets->route);
		if (assets->track) TmnfTrack_Unload(assets->track);
		if (assets->trigger_geometry) TmnfTrack_Unload(assets->trigger_geometry);
		free(assets->vehicle_path);
		free(assets);
	}
	free(env);
}

int tmnf_collection_reset(TmnfCollection *env)
{
	if (!env) return error("null collection environment");
#if TMNF_COLLECTION_CUDA
	if (env->device) TmnfCudaVecEnv_Destroy(env->device);
	env->device = NULL;
#endif
	if (env->world) World_Destroy(env->world);
	env->ticks = env->respawns = 0;
	env->checkpoint_time_count = 0;
	uint32_t prestart_checkpoints = 0;
	GmIso4 start = env->assets->route->start->spawn;
	if (env->has_initial_position)
		memcpy(start.t, env->initial_position, sizeof(start.t));
	if (env->initialize_from_graph && !env->has_initial_position) {
		/* TMInterface validation's blocked countdown: the 259th state
		 * matches race time zero, including continuously moving contacts. */
		env->world = World_CreateRaceAfterCountdownSeed(env->assets->track,
			env->assets->vehicle_path, &start, env->validation_seed, 259,
			env->assets->route, &env->race, env->lap_count, &prestart_checkpoints);
	} else if (env->initialize_from_graph || env->validation_seed || env->has_initial_position) {
		env->world = World_CreateAtSpawnSeed(env->assets->track,
			env->assets->vehicle_path, &start, env->validation_seed);
	} else {
		env->world = World_Create(env->assets->track, env->assets->vehicle_path);
	}
	World_GetPhysicsWorld(env->world)->route = env->assets->route;
	if (!env->initialize_from_graph || env->has_initial_position)
		TmnfRace_Reset(env->assets->route, &env->race,
			World_GetPhysicsWorld(env->world)->corpora[0].collision_corpus->live_iso);
	if (env->lap_count) env->race.lap_limit = env->lap_count;
	if (!append_checkpoint_times(env, prestart_checkpoints, UINT32_MAX) ||
		!append_checkpoint_times(env,
			env->race.passed_race_checkpoints - prestart_checkpoints, 0)) return 0;
	env->replay_has_previous = 0;
	env->sync_input_steer = 0.0f;
	env->sync_contact_forward = env->previous_sync_contact_forward = 0;
	env->sync_contact_side = env->previous_sync_contact_side = 0.0f;
	update_legacy_sync_contact(env);
#if TMNF_COLLECTION_CUDA
	if (env->cuda) {
		TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
		config.use_world_spawn = 1;
		TmnfCudaVecEnvLimits limits = TmnfCudaVecEnv_DefaultLimits();
		env->device = TmnfCudaVecEnv_Create(env->assets->track, env->world,
			env->assets->route, 1, &config, &limits);
		if (!env->device) return error(TmnfCudaVecEnv_LastError());
		/* Preserve countdown callbacks and the recorded lap limit from
		 * host initialization instead of resetting to the map defaults. */
		TmnfEnvSnapshot snapshot;
		TmnfCudaVecEnv_Capture(env->device, &snapshot);
		snapshot.race_state = env->race;
		TmnfCudaVecEnv_Restore(env->device, &snapshot);
		TmnfCudaVecEnv_SetReplayMode(env->device, env->replay_mode);
	}
#endif
	return 1;
}

int tmnf_collection_reset_seed(TmnfCollection *env, uint32_t validation_seed)
{
	if (!env) return error("null collection environment");
	env->validation_seed = validation_seed;
	return tmnf_collection_reset(env);
}

int tmnf_collection_set_replay_mode(TmnfCollection *env, int enabled)
{
	if (!env) return error("null collection environment");
	env->replay_mode = enabled != 0;
#if TMNF_COLLECTION_CUDA
	if (env->device) TmnfCudaVecEnv_SetReplayMode(env->device, enabled);
#endif
	return 1;
}

int tmnf_collection_set_lap_count(TmnfCollection *env, uint32_t lap_count)
{
	if (!env || env->ticks)
		return error("recorded lap settings require an unstepped environment");
	const TmnfRouteMetadata *metadata = env->assets->route->metadata;
	uint32_t limit = lap_count ? lap_count : metadata->lap_count;
	if ((uint64_t)limit * (metadata->checkpoint_count + 1) > UINT32_MAX ||
		(limit > 1 && (uint64_t)metadata->checkpoint_count + metadata->finish_count > TMNF_CHECKPOINT_CAPACITY))
		return error("recorded lap setting exceeds supported checkpoint contacts/count");
	int changed = env->lap_count != lap_count;
	env->lap_count = lap_count;
	/* A time-zero finish contact must use the recorded lap limit too. */
	if (changed && (env->cuda || (env->initialize_from_graph && !env->has_initial_position)))
		return tmnf_collection_reset(env);
	env->race.lap_limit = limit;
	return 1;
}

int tmnf_collection_reset_position(TmnfCollection *env, uint32_t validation_seed,
	const float position[3])
{
	if (!env) return error("null collection environment");
	if (position) {
		for (int i = 0; i < 3; ++i)
			if (!isfinite(position[i])) return error("initial position must be finite");
		memcpy(env->initial_position, position, sizeof(env->initial_position));
	}
	env->has_initial_position = position != NULL;
	env->validation_seed = validation_seed;
	return tmnf_collection_reset(env);
}

uint32_t tmnf_collection_checkpoint_times(TmnfCollection *env, const uint32_t **times)
{
	if (!env || !times) return 0;
	*times = env->checkpoint_times;
	return env->checkpoint_time_count;
}

static int append_checkpoint_times(TmnfCollection *env, uint32_t accepted,
	uint32_t race_time_ms)
{
	if (!accepted) return 1;
	if (accepted > UINT32_MAX - env->checkpoint_time_count)
		return error("checkpoint time count overflow");
	uint32_t wanted = env->checkpoint_time_count + accepted;
	if (wanted > env->checkpoint_time_capacity) {
		uint32_t capacity = env->checkpoint_time_capacity ? env->checkpoint_time_capacity : 32;
		while (capacity < wanted) {
			if (capacity > UINT32_MAX / 2) { capacity = wanted; break; }
			capacity *= 2;
		}
		uint32_t *times = realloc(env->checkpoint_times, (size_t)capacity * sizeof(*times));
		if (!times) return error("cannot allocate checkpoint times");
		env->checkpoint_times = times;
		env->checkpoint_time_capacity = capacity;
	}
	while (env->checkpoint_time_count < wanted)
		env->checkpoint_times[env->checkpoint_time_count++] = race_time_ms;
	return 1;
}

static TmnfCollection *create_collection(const char *track, const char *vehicle,
	const char *route, const uint8_t sha256[32], int cuda, int initialize_from_graph,
	int build_grid, uint32_t validation_seed, const char *triggers)
{
	collection_error[0] = 0;
	if (!track || !vehicle || !route || !sha256 || !World_HasLocalAssets()) {
		error("collection requires track, vehicle, route, hash, and local game mask");
		return NULL;
	}
#if !TMNF_COLLECTION_CUDA
	if (cuda) {
		error("this library has no CUDA collection support; select a CUDA build");
		return NULL;
	}
#endif
	TmnfCollection *env = calloc(1, sizeof(*env));
	if (!env) { error("cannot allocate collection environment"); return NULL; }
	env->assets = calloc(1, sizeof(*env->assets));
	if (!env->assets) {
		error("cannot allocate collection assets");
		free(env);
		return NULL;
	}
	atomic_init(&env->assets->references, 1);
	env->assets->vehicle_path = malloc(strlen(vehicle) + 1);
	if (!env->assets->vehicle_path) {
		error("cannot allocate vehicle path");
		tmnf_collection_destroy(env);
		return NULL;
	}
	strcpy(env->assets->vehicle_path, vehicle);
	env->cuda = cuda != 0;
	env->initialize_from_graph = initialize_from_graph;
	env->validation_seed = validation_seed;
	env->assets->track = TmnfTrack_LoadWithGrid(track, sha256, build_grid);
	env->assets->route = TmnfRoute_Load(route, sha256);
	if (triggers) {
		env->assets->trigger_geometry = TmnfTrack_LoadWithGrid(triggers, sha256, 0);
		const TmnfTrack *geometry = env->assets->trigger_geometry;
		if (geometry->entry_count < 2 || geometry->entries[0].tree_flags != 0 ||
			geometry->entries[0].skip_count != geometry->entry_count) {
			error("trigger geometry lacks the game collision tree; regenerate the fixture");
			tmnf_collection_destroy(env);
			return NULL;
		}
		uint32_t expected = env->assets->route->metadata->checkpoint_count +
			env->assets->route->metadata->finish_count;
		if (env->assets->trigger_geometry->corpus_count != expected) {
			error("trigger geometry does not match route waypoint count");
			tmnf_collection_destroy(env);
			return NULL;
		}
		env->assets->route->trigger_geometry = env->assets->trigger_geometry;
	}
	if (!tmnf_collection_reset(env)) {
		tmnf_collection_destroy(env);
		return NULL;
	}
	return env;
}

TmnfCollection *tmnf_collection_create(const char *track, const char *vehicle,
	const char *route, const uint8_t sha256[32], int cuda)
{
	return create_collection(track, vehicle, route, sha256, cuda, 0, 1, 0, NULL);
}

TmnfCollection *tmnf_collection_create_graph(const char *track, const char *vehicle,
	const char *route, const uint8_t sha256[32])
{
	return create_collection(track, vehicle, route, sha256, 0, 1, 1, 0, NULL);
}

TmnfCollection *tmnf_collection_create_config(const char *track, const char *vehicle,
	const char *route, const uint8_t sha256[32], int cuda, int initialize_from_graph,
	int build_grid, uint32_t validation_seed)
{
	if (cuda && !build_grid) {
		error("tree collision mode requires CPU simulation");
		return NULL;
	}
	return create_collection(track, vehicle, route, sha256, cuda, initialize_from_graph,
		build_grid, validation_seed, NULL);
}

TmnfCollection *tmnf_collection_create_geometry(const char *track, const char *vehicle,
	const char *route, const char *triggers, const uint8_t sha256[32], int cuda,
	int initialize_from_graph, int build_grid, uint32_t validation_seed)
{
	return create_collection(track, vehicle, route, sha256, cuda, initialize_from_graph,
		build_grid, validation_seed, triggers);
}

TmnfCollection *tmnf_collection_clone(TmnfCollection *source)
{
	if (!source || source->cuda) {
		error("collection cloning requires a CPU source environment");
		return NULL;
	}
	TmnfCollection *env = calloc(1, sizeof(*env));
	if (!env) { error("cannot allocate collection environment"); return NULL; }
	env->assets = source->assets;
	env->initialize_from_graph = source->initialize_from_graph;
	env->validation_seed = source->validation_seed;
	env->has_initial_position = source->has_initial_position;
	env->replay_mode = source->replay_mode;
	env->lap_count = source->lap_count;
	memcpy(env->initial_position, source->initial_position, sizeof(env->initial_position));
	atomic_fetch_add(&env->assets->references, 1);
	if (!tmnf_collection_reset(env)) {
		tmnf_collection_destroy(env);
		return NULL;
	}
	return env;
}

int tmnf_collection_observe(TmnfCollection *env, TmnfCollectionFrame *frame)
{
	if (!env || !frame) return error("null collection observation argument");
	memset(frame, 0, sizeof(*frame));
	/* Larger routes pack checkpoint flags into the existing fixed-size export.
	 * Small routes retain byte-identical version-2 observations. */
	frame->version = env->assets->route->metadata->checkpoint_count < 64 ? 2 : 3;
	frame->time_ms = (env->ticks + 1) * 10;
	frame->finished = env->race.finished;
	frame->finish_time_ms = env->race.finish_time_ms;
	frame->checkpoint_count = env->assets->route->metadata->checkpoint_count + 1;
	for (uint32_t i = 0; i < frame->checkpoint_count; ++i) {
		uint32_t taken = i + 1 == frame->checkpoint_count ? env->race.finished
			: TmnfCheckpointSet_Test(&env->race.visited, i);
		if (frame->version == 2) frame->checkpoint_states[i] = taken;
		else frame->checkpoint_states[i / 32] |= taken << (i % 32);
	}
	frame->completed_laps = env->race.completed_laps;
	frame->respawn_count = env->respawns;
	frame->dyna = *World_GetPlayerState(env->world);
	World_WritePlayerGameState(env->world, frame->car, frame->wheels);
	/* UpdateSync precedes the next input update. Legacy is_turbo features
	 * accidentally read these steering bits, so retain their original timing. */
	memcpy(frame->car + 0x3a8, &env->sync_input_steer, 4);
	memcpy(frame->car + 0x394, &env->previous_sync_contact_forward, 4);
	memcpy(frame->car + 0x398, &env->previous_sync_contact_side, 4);
	return 1;
}

static void update_legacy_sync_contact(TmnfCollection *env)
{
	/* The old speed fields land on the previous sync state's last three
	 * fields. UpdateSync (0x007C0836..0x007C0ACA) stores a sticky contact
	 * direction bit and a contact angle there, rather than vehicle speeds. */
	env->previous_sync_contact_forward = env->sync_contact_forward;
	env->previous_sync_contact_side = env->sync_contact_side;
	GmVec3 normal;
	if (!World_GetPlayerBodyContactNormal(env->world, &normal)) {
		env->sync_contact_side = 0.0f;
		return;
	}
	float length2 = x87_add(x87_add(x87_mul(normal.y, normal.y),
		x87_mul(normal.x, normal.x)), x87_mul(normal.z, normal.z));
	if (length2 > 0x1.b7cdfcp-34f) {
		float inverse = x87_rcp(x87_sqrt(length2));
		normal.x = x87_mul(normal.x, inverse);
		normal.y = x87_mul(normal.y, inverse);
		normal.z = x87_mul(normal.z, inverse);
	}
	normal.x = -normal.x;
	normal.y = -normal.y;
	normal.z = -normal.z;
	length2 = x87_add(x87_add(x87_mul(normal.x, normal.x),
		x87_mul(normal.y, normal.y)), 0.0f);
	if (length2 > 0x1.b7cdfcp-34f) {
		float inverse = x87_rcp(x87_sqrt(length2));
		normal.x = x87_mul(normal.x, inverse);
		normal.y = x87_mul(normal.y, inverse);
	}
	float angle = x87_atan2(fabsf(normal.x), -normal.y);
	env->sync_contact_side = x87_div(fabsf(angle), 0x1.921fb6p+1f);
	env->sync_contact_forward = 0.0f < normal.z;
}

static int collection_step(TmnfCollection *env, const TMNFRaceInputs *inputs,
	TmnfCollectionFrame *frame)
{
	if (!env || !inputs) return error("null collection tick argument");
	if (env->race.finished) return error("race has finished; reset before stepping");
	if (env->ticks >= UINT32_MAX / 10 - 1) return error("collection clock overflow");
	if (inputs->respawn && !env->race.respawn_available && !env->replay_mode)
		return error("restart respawns before a checkpoint require a new episode");
	uint32_t before = env->race.passed_race_checkpoints;
#if TMNF_COLLECTION_CUDA
	if (env->cuda) {
		TmnfCudaVecEnv_StepWithPreStateCopy(env->device, inputs, 10, 0, env->world);
		if (frame) tmnf_collection_observe(env, frame);
		TmnfCudaVecEnv_CopyWorldToHost(env->device, 0, env->world);
		TmnfCudaVecEnv_CopyRaceToHost(env->device, 0, &env->race);
	} else
#endif
	{
		/* SmallRespawn (0x00472700) bypasses the live-player restart gate
		 * for recorded playback, retaining the clock and previous spawn. */
		if (inputs->respawn) World_Respawn(env->world, &env->race.respawn_location);
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(inputs,
			World_GetPlayerVehicle(env->world));
		if (frame) tmnf_collection_observe(env, frame);
		World_GetPhysicsWorld(env->world)->race_visited =
			env->race.visited;
		World_GetPhysicsWorld(env->world)->race_state = &env->race;
		World_AdvanceTimer(env->world, 10);
		CHmsZoneDynamic_PhysicsStep2(World_GetPhysicsWorld(env->world), 10);
		TmnfPhysicsWorld *physics = World_GetPhysicsWorld(env->world);
		TmnfRace_EndStep(env->assets->route, &env->race,
			physics->corpora[0].collision_corpus->live_iso, physics->race_step_result);
	}
	env->sync_input_steer = World_GetPlayerVehicle(env->world)->input_steer;
	update_legacy_sync_contact(env);
	++env->ticks;
	env->respawns += inputs->respawn != 0;
	return append_checkpoint_times(env, env->race.passed_race_checkpoints - before,
		env->ticks * 10);
}

int tmnf_collection_tick(TmnfCollection *env, const TMNFRaceInputs *inputs,
	TmnfCollectionFrame *frame)
{
	if (!frame) return error("null collection tick frame");
	return collection_step(env, inputs, frame);
}

int tmnf_collection_replay_samples(TmnfCollection *env, const TMNFRaceInputs *inputs,
	uint32_t count, int every_tick, int last_chunk, const uint8_t *sample_mask,
	TmnfCollectionFrame *frames,
	uint32_t *indices, uint32_t capacity, uint32_t *selected, uint32_t *executed,
	TmnfCollectionFrame *final)
{
	if (!env || !inputs || !frames || !indices || !selected || !executed || !final ||
		!count || capacity < count)
		return error("invalid replay collection buffers");
	*selected = *executed = 0;
	while (*executed < count && !env->race.finished) {
		uint32_t i = *executed;
		TmnfCollectionFrame *frame = &frames[*selected];
		if (!collection_step(env, &inputs[i], frame)) return 0;
		++*executed;
		float applied[3];
		memcpy(applied, frame->car + 0x50, sizeof(applied));
		int changed = !env->replay_has_previous;
		for (int j = 0; j < 3; ++j)
			changed |= applied[j] != env->replay_previous[j];
		memcpy(env->replay_previous, applied, sizeof(applied));
		env->replay_has_previous = 1;
		if (every_tick || (sample_mask && sample_mask[i]) ||
			frame->time_ms == 10 || frame->time_ms % 100 == 0 ||
			changed || (last_chunk && *executed == count) || env->race.finished)
			indices[(*selected)++] = i;
	}
	return tmnf_collection_observe(env, final);
}

int tmnf_collection_replay(TmnfCollection *env, const TMNFRaceInputs *inputs,
	uint32_t count, int every_tick, int last_chunk, TmnfCollectionFrame *frames,
	uint32_t *indices, uint32_t capacity, uint32_t *selected, uint32_t *executed,
	TmnfCollectionFrame *final)
{
	return tmnf_collection_replay_samples(env, inputs, count, every_tick, last_chunk,
		NULL, frames, indices, capacity, selected, executed, final);
}

int tmnf_collection_advance(TmnfCollection *env, const TMNFRaceInputs *inputs,
	uint32_t count, uint32_t timeout_ms, TmnfCollectionFrame *frame, uint32_t *executed)
{
	if (!env || !inputs || !frame || !executed || count == 0 || count > 10)
		return error("invalid collection action arguments");
	*executed = 0;
	while (*executed < count && !env->race.finished && (env->ticks + 1) * 10 < timeout_ms) {
		if (!collection_step(env, &inputs[*executed], NULL)) return 0;
		++*executed;
	}
	return tmnf_collection_observe(env, frame);
}

int tmnf_collection_action(TmnfCollection *env, uint32_t controls,
	uint32_t count, uint32_t timeout_ms, TmnfCollectionFrame *frame,
	TMNFRaceInputs records[10], uint32_t *executed)
{
	if (!env || !records || count == 0 || count > 10 || controls > 15)
		return error("invalid digital collection action");
	memset(records, 0, count * sizeof(*records));
	for (uint32_t i = 0; i < count; ++i) {
		uint32_t time_ms = (env->ticks + 1 + i) * 10;
		records[i].steer_left_time = records[i].steer_right_time = time_ms;
		records[i].accelerate_time = records[i].brake_time = time_ms;
		records[i].brake = controls & 1;
		records[i].accelerate = (controls >> 1) & 1;
		records[i].steer_right = (controls >> 2) & 1;
		records[i].steer_left = (controls >> 3) & 1;
	}
	return tmnf_collection_advance(env, records, count, timeout_ms, frame, executed);
}

int tmnf_collection_batch_action(TmnfCollection *const *envs,
	const uint32_t *controls, const uint32_t *ticks, uint32_t count,
	uint32_t threads, uint32_t timeout_ms, TmnfCollectionFrame *frames,
	TMNFRaceInputs *records, uint32_t *executed)
{
	if (!envs || !controls || !ticks || !frames || !records || !executed || !count || !threads)
		return error("invalid batched collection action");
#ifndef _OPENMP
	if (threads != 1) return error("parallel collection requires an OpenMP build");
#endif
	for (uint32_t i = 0; i < count; ++i)
		if (!envs[i] || envs[i]->cuda || controls[i] > 15 || ticks[i] == 0 || ticks[i] > 10)
			return error("batched actions require CPU worlds and valid digital controls");
	int success = 1;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) reduction(&:success) if(threads > 1)
#endif
	for (uint32_t i = 0; i < count; ++i)
		success &= tmnf_collection_action(envs[i], controls[i], ticks[i], timeout_ms,
			&frames[i], &records[i * 10], &executed[i]);
	return success ? 1 : error("a native collection batch action failed");
}
