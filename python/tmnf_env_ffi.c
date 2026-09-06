#include "tmnf_env_ffi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "route.h"
#include "track.h"
#include "vec_env.h"
#include "world.h"

struct TmnfPythonEnv {
	TmnfTrack *track;
	TmnfRoute *route;
	TmnfWorld **owners;
	TmnfPhysicsWorld **worlds;
	uint32_t *player_indices;
	void *actions;
	TmnfObservation *reset_observations;
	TmnfStepResult *results;
	uint32_t count;
	TmnfActionSpace action_space;
	TmnfVecEnv vec_env;
};

static _Thread_local char creation_error[256];

const char *tmnf_env_creation_error(void)
{
	return creation_error;
}

static void ffi_fail(const char *message)
{
	fprintf(stderr, "tmnf python ffi: %s\n", message);
	abort();
}

static void *ffi_allocate(size_t count, size_t size)
{
	if (count == 0 || size == 0 || count > SIZE_MAX / size)
		ffi_fail("invalid allocation size");
	void *memory = calloc(count, size);
	if (memory == NULL)
		ffi_fail("out of memory");
	return memory;
}

TmnfPythonEnv *tmnf_env_create_with_budget(
	const char *track_path,
	const char *vehicle_path,
	const char *route_path,
	const uint8_t track_sha256[32],
	uint32_t environment_count,
	uint32_t thread_count,
	uint32_t action_space,
	uint32_t max_race_ticks,
	uint32_t horizon_ticks,
	uint32_t off_track_grace_ticks,
	uint32_t stuck_grace_ticks,
	float discount_per_tick,
	float reference_speed,
	float stuck_progress_epsilon,
	uint32_t respawn_action,
	float budget_reference_speed)
{
	creation_error[0] = '\0';
	if (!World_HasLocalAssets()) {
		snprintf(creation_error, sizeof(creation_error), "local game assets missing; see docs/LOCAL_ASSETS.md");
		return NULL;
	}
	if (track_path == NULL || vehicle_path == NULL ||
		route_path == NULL || track_sha256 == NULL ||
		environment_count == 0 ||
		thread_count == 0 ||
		action_space > TMNF_ACTION_SPACE_ANALOG ||
		off_track_grace_ticks == 0 ||
		stuck_grace_ticks == 0 ||
		!(discount_per_tick > 0.0f) ||
		!(discount_per_tick <= 1.0f) ||
		!(reference_speed > 0.0f) ||
		!(stuck_progress_epsilon > 0.0f) ||
		!isfinite(discount_per_tick) ||
		!isfinite(reference_speed) ||
		!isfinite(stuck_progress_epsilon) ||
		!(budget_reference_speed >= 0.0f) ||
		!isfinite(budget_reference_speed)) {
		snprintf(creation_error, sizeof(creation_error), "invalid creation argument");
		return NULL;
	}

	/* Reject bad budgets before allocating worlds. The core retains its fatal
	 * invariant checks; Python callers get a recoverable creation failure. */
	TmnfRoute *route = TmnfRoute_Load(route_path, track_sha256);
	double length = (double)TmnfRoute_GetReferenceLength(route) * route->metadata->lap_count;
	double minimum = fmax(1.0, ceil(length / TMNF_MAX_LINEAR_SPEED_MPS /
		(TMNF_RACE_TICK_MS / 1000.0)));
	double pace = ceil(TMNF_HORIZON_PACE_FACTOR * length / reference_speed /
		(TMNF_RACE_TICK_MS / 1000.0));
	/* The core computes the reward-reference pace even with explicit limits. */
	if (!isfinite(pace) || pace > UINT32_MAX) {
		snprintf(creation_error, sizeof(creation_error), "reference_speed derives an overflowing race budget");
		TmnfRoute_Unload(route);
		return NULL;
	}
	if (budget_reference_speed > 0.0f && (max_race_ticks == 0 || horizon_ticks == 0))
		pace = ceil(TMNF_HORIZON_PACE_FACTOR * length / budget_reference_speed /
			(TMNF_RACE_TICK_MS / 1000.0));
	if (!isfinite(pace) || pace > UINT32_MAX) {
		snprintf(creation_error, sizeof(creation_error), "budget_reference_speed derives an overflowing race budget");
		TmnfRoute_Unload(route);
		return NULL;
	}
	double race_ticks = max_race_ticks ? max_race_ticks : pace;
	double collection_ticks = horizon_ticks ? horizon_ticks : pace;
	if (race_ticks < minimum || collection_ticks < minimum) {
		snprintf(creation_error, sizeof(creation_error),
			"%s is below the route budget minimum of %.0f ticks (%.0f m reference race length)",
			race_ticks < minimum ? "max_race_ticks" : "horizon_ticks", minimum, length);
		TmnfRoute_Unload(route);
		return NULL;
	}

	TmnfPythonEnv *env = ffi_allocate(1, sizeof(*env));
	env->route = route;
	env->count = environment_count;
	env->action_space = (TmnfActionSpace)action_space;
	env->owners = ffi_allocate(environment_count, sizeof(*env->owners));
	env->worlds = ffi_allocate(environment_count, sizeof(*env->worlds));
	env->player_indices =
		ffi_allocate(environment_count, sizeof(*env->player_indices));
	env->actions = ffi_allocate(
		environment_count,
		env->action_space == TMNF_ACTION_SPACE_DISCRETE
			? sizeof(uint8_t)
			: sizeof(TmnfAnalogAction));
	env->reset_observations =
		ffi_allocate(environment_count, sizeof(*env->reset_observations));
	env->results = ffi_allocate(environment_count, sizeof(*env->results));

	env->track = TmnfTrack_Load(track_path, track_sha256);
	for (uint32_t i = 0; i < environment_count; ++i) {
		env->owners[i] = World_Create(env->track, vehicle_path);
		env->worlds[i] = World_GetPhysicsWorld(env->owners[i]);
	}

	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	config.max_race_ticks = max_race_ticks;
	config.horizon_ticks = horizon_ticks;
	if (budget_reference_speed > 0.0f &&
		(max_race_ticks == 0 || horizon_ticks == 0)) {
		uint32_t budget = TmnfVecEnv_DerivePaceTicks(env->route, budget_reference_speed);
		if (max_race_ticks == 0)
			config.max_race_ticks = budget;
		if (horizon_ticks == 0)
			config.horizon_ticks = budget;
	}
	config.off_track_grace_ticks = off_track_grace_ticks;
	config.stuck_grace_ticks = stuck_grace_ticks;
	config.thread_count = thread_count;
	config.discount_per_tick = discount_per_tick;
	config.reference_speed = reference_speed;
	config.stuck_progress_epsilon = stuck_progress_epsilon;
	config.autoreset_mode = TMNF_AUTORESET_SAME_STEP;
	config.action_space = env->action_space;
	config.respawn_action = respawn_action != 0;
	TmnfVecEnv_Init(
		&env->vec_env,
		env->worlds,
		env->player_indices,
		environment_count,
		env->route,
		&config);
	return env;
}

/* Preserve the existing FFI entry point and its argument layout. */
TmnfPythonEnv *tmnf_env_create(
	const char *track_path, const char *vehicle_path, const char *route_path,
	const uint8_t track_sha256[32], uint32_t environment_count,
	uint32_t thread_count, uint32_t action_space, uint32_t max_race_ticks,
	uint32_t horizon_ticks, uint32_t off_track_grace_ticks,
	uint32_t stuck_grace_ticks, float discount_per_tick, float reference_speed,
	float stuck_progress_epsilon, uint32_t respawn_action)
{
	return tmnf_env_create_with_budget(track_path, vehicle_path, route_path,
		track_sha256, environment_count, thread_count, action_space,
		max_race_ticks, horizon_ticks, off_track_grace_ticks, stuck_grace_ticks,
		discount_per_tick, reference_speed, stuck_progress_epsilon, respawn_action, 0.0f);
}

void tmnf_env_destroy(TmnfPythonEnv *env)
{
	if (env == NULL)
		return;
	TmnfVecEnv_Destroy(&env->vec_env);
	TmnfRoute_Unload(env->route);
	for (uint32_t i = 0; i < env->count; ++i)
		World_Destroy(env->owners[i]);
	TmnfTrack_Unload(env->track);
	free(env->results);
	free(env->reset_observations);
	free(env->actions);
	free(env->player_indices);
	free(env->worlds);
	free(env->owners);
	free(env);
}

void tmnf_env_reset(TmnfPythonEnv *env)
{
	if (env == NULL)
		ffi_fail("reset argument is null");
	TmnfVecEnv_Reset(
		&env->vec_env, NULL, env->reset_observations);
	for (uint32_t i = 0; i < env->count; ++i) {
		memset(&env->results[i], 0, sizeof(env->results[i]));
		env->results[i].observation = env->reset_observations[i];
		env->results[i].transition_discount = 1.0f;
		env->results[i].episode_id = env->vec_env.episode_ids[i];
	}
}

void tmnf_env_step_discrete(
	TmnfPythonEnv *env, uint32_t action_repeat)
{
	if (env == NULL || action_repeat == 0)
		ffi_fail("discrete step argument is invalid");
	TmnfVecEnv_StepDiscrete(
		&env->vec_env, env->actions, action_repeat, env->results);
}

void tmnf_env_step_analog(
	TmnfPythonEnv *env, uint32_t action_repeat)
{
	if (env == NULL || action_repeat == 0)
		ffi_fail("analog step argument is invalid");
	TmnfVecEnv_StepAnalog(
		&env->vec_env, env->actions, action_repeat, env->results);
}

size_t tmnf_gate_observations_size(void) { return sizeof(TmnfGateObservations); }
void tmnf_env_observe_gates(TmnfPythonEnv *env, void *gates)
{
 if (!env) ffi_fail("gate query environment is null");
 TmnfVecEnv_ObserveGates(&env->vec_env, gates);
}
void tmnf_env_step_with_gates(TmnfPythonEnv *env, uint32_t repeat,
 void *gates, void *final_gates)
{
 if (!env) ffi_fail("gate step environment is null");
 if (env->vec_env.config.action_space == TMNF_ACTION_SPACE_ANALOG)
  TmnfVecEnv_StepAnalogWithGates(&env->vec_env, env->actions, repeat,
   env->results, gates, final_gates);
 else
  TmnfVecEnv_StepDiscreteWithGates(&env->vec_env, env->actions, repeat,
   env->results, gates, final_gates);
}

void tmnf_env_capture(
	TmnfPythonEnv *env, const uint32_t *indices, uint32_t count,
	void *snapshots)
{
	if (env == NULL || indices == NULL || count == 0 ||
		snapshots == NULL) {
		ffi_fail("capture argument is null or empty");
	}
	TmnfVecEnv_CaptureIndices(
		&env->vec_env, indices, count, snapshots);
}

void tmnf_env_restore(
	TmnfPythonEnv *env, const uint32_t *indices, uint32_t count,
	const void *snapshots)
{
	if (env == NULL || indices == NULL || count == 0 ||
		snapshots == NULL) {
		ffi_fail("restore argument is null or empty");
	}
	TmnfVecEnv_RestoreIndices(
		&env->vec_env, indices, count, snapshots,
		env->reset_observations);
	for (uint32_t i = 0; i < count; ++i) {
		uint32_t index = indices[i];
		memset(&env->results[index], 0, sizeof(env->results[index]));
		env->results[index].observation = env->reset_observations[i];
		env->results[index].transition_discount = 1.0f;
		env->results[index].episode_id =
			env->vec_env.episode_ids[index];
	}
}

uint8_t *tmnf_env_discrete_actions(TmnfPythonEnv *env)
{
	if (env == NULL)
		ffi_fail("discrete action buffer requested from null environment");
	if (env->action_space != TMNF_ACTION_SPACE_DISCRETE)
		ffi_fail("discrete action buffer requested from analog environment");
	return env->actions;
}

void *tmnf_env_analog_actions(TmnfPythonEnv *env)
{
	if (env == NULL)
		ffi_fail("analog action buffer requested from null environment");
	if (env->action_space != TMNF_ACTION_SPACE_ANALOG)
		ffi_fail("analog action buffer requested from discrete environment");
	return env->actions;
}

void *tmnf_env_results(TmnfPythonEnv *env)
{
	if (env == NULL)
		ffi_fail("result buffer requested from null environment");
	return env->results;
}

uint32_t tmnf_env_count(const TmnfPythonEnv *env)
{
	if (env == NULL)
		ffi_fail("count requested from null environment");
	return env->count;
}

float tmnf_env_route_length(const TmnfPythonEnv *env)
{
	if (env == NULL)
		ffi_fail("route length requested from null environment");
	return TmnfRoute_GetReferenceLength(env->route);
}

uint32_t tmnf_env_checkpoint_count(const TmnfPythonEnv *env)
{
	if (env == NULL)
		ffi_fail("checkpoint count requested from null environment");
	return env->route->metadata->checkpoint_count;
}

uint32_t tmnf_env_lap_count(const TmnfPythonEnv *env)
{
	if (env == NULL)
		ffi_fail("lap count requested from null environment");
	return env->route->metadata->lap_count;
}

/* Resolved collection horizon: the derived per-track value when 0 was passed. */
uint32_t tmnf_env_horizon_ticks(const TmnfPythonEnv *env)
{
	if (env == NULL)
		ffi_fail("horizon requested from null environment");
	return env->vec_env.config.horizon_ticks;
}

/* Resolved race timeout: the derived per-track value when 0 was passed. */
uint32_t tmnf_env_max_race_ticks(const TmnfPythonEnv *env)
{
	if (env == NULL)
		ffi_fail("race timeout requested from null environment");
	return env->vec_env.config.max_race_ticks;
}

uint32_t tmnf_observation_version(void)
{
	return TMNF_OBSERVATION_VERSION;
}

size_t tmnf_observation_size(void)
{
	return sizeof(TmnfObservation);
}

size_t tmnf_step_result_size(void)
{
	return sizeof(TmnfStepResult);
}

size_t tmnf_env_snapshot_size(void)
{
	return sizeof(TmnfEnvSnapshot);
}
