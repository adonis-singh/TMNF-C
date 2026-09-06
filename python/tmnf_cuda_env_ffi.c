#include "tmnf_cuda_env_ffi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "route.h"
#include "tmnf_cuda_env.h"
#include "track.h"
#include "vec_env.h"
#include "world.h"

struct TmnfPythonCudaEnv {
	TmnfTrack *track;
	TmnfRoute *route;
	TmnfWorld *template_world;
	TmnfCudaVecEnv *env;
	uint32_t count;
	TmnfActionSpace action_space;
	/* Host scratch for the flatten of reset/restore observations through
	 * the public TmnfVecEnv_FlattenStepResults: count step results whose
	 * observation is filled, plus the two outputs the caller does not want. */
	TmnfObservation *observations;
	TmnfStepResult *results;
	float *flat_final;
	float *flat_transitions;
};

static void ffi_fail(const char *message)
{
	fprintf(stderr, "tmnf cuda python ffi: %s\n", message);
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

TmnfPythonCudaEnv *tmnf_cuda_env_create(
	const char *track_path,
	const char *vehicle_path,
	const char *route_path,
	const uint8_t track_sha256[32],
	uint32_t environment_count,
	uint32_t action_space,
	uint32_t max_race_ticks,
	uint32_t horizon_ticks,
	uint32_t off_track_grace_ticks,
	uint32_t stuck_grace_ticks,
	float discount_per_tick,
	float reference_speed,
	float stuck_progress_epsilon,
	uint32_t respawn_action,
	uint32_t stack_bytes,
	void *stream)
{
	if (!World_HasLocalAssets()) {
		fprintf(stderr, "local game assets missing; see docs/LOCAL_ASSETS.md\n");
		return NULL;
	}
	if (track_path == NULL || vehicle_path == NULL ||
		route_path == NULL || track_sha256 == NULL ||
		environment_count == 0 ||
		action_space > TMNF_ACTION_SPACE_ANALOG ||
		max_race_ticks == 0 || horizon_ticks == 0 ||
		off_track_grace_ticks == 0 ||
		stuck_grace_ticks == 0 ||
		!(discount_per_tick > 0.0f) ||
		!(discount_per_tick <= 1.0f) ||
		!(reference_speed > 0.0f) ||
		!(stuck_progress_epsilon > 0.0f) ||
		!isfinite(discount_per_tick) ||
		!isfinite(reference_speed) ||
		!isfinite(stuck_progress_epsilon)) {
		ffi_fail("invalid creation argument");
	}

	TmnfPythonCudaEnv *env = ffi_allocate(1, sizeof(*env));
	env->count = environment_count;
	env->action_space = (TmnfActionSpace)action_space;
	env->observations =
		ffi_allocate(environment_count, sizeof(*env->observations));
	env->results = ffi_allocate(environment_count, sizeof(*env->results));
	env->flat_final = ffi_allocate(
		environment_count, TMNF_POLICY_OBSERVATION_WIDTH * sizeof(float));
	env->flat_transitions = ffi_allocate(
		environment_count, TMNF_POLICY_TRANSITION_WIDTH * sizeof(float));

	env->track = TmnfTrack_Load(track_path, track_sha256);
	env->template_world = World_Create(env->track, vehicle_path);
	env->route = TmnfRoute_Load(route_path, track_sha256);

	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	config.max_race_ticks = max_race_ticks;
	config.horizon_ticks = horizon_ticks;
	config.off_track_grace_ticks = off_track_grace_ticks;
	config.stuck_grace_ticks = stuck_grace_ticks;
	config.discount_per_tick = discount_per_tick;
	config.reference_speed = reference_speed;
	config.stuck_progress_epsilon = stuck_progress_epsilon;
	config.autoreset_mode = TMNF_AUTORESET_SAME_STEP;
	config.action_space = env->action_space;
	config.respawn_action = respawn_action != 0;
	TmnfCudaVecEnvLimits limits = TmnfCudaVecEnv_DefaultLimits();
	limits.stream = (struct CUstream_st *)stream;
	/* Per-thread device stack: the kernel's frame depends on the target
	 * architecture (14,160 B on sm_120, 39,072 B on sm_86 per cuobjdump
	 * -res-usage); the caller passes the size for its device. */
	if (stack_bytes == 0)
		ffi_fail("stack_bytes must be positive");
	limits.stack_bytes = stack_bytes;
	env->env = TmnfCudaVecEnv_Create(
		env->track, env->template_world, env->route, environment_count,
		&config, &limits);
	if (env->env == NULL) {
		TmnfRoute_Unload(env->route);
		World_Destroy(env->template_world);
		TmnfTrack_Unload(env->track);
		free(env->flat_transitions);
		free(env->flat_final);
		free(env->results);
		free(env->observations);
		free(env);
		return NULL;
	}
	return env;
}

const char *tmnf_cuda_env_last_error(void)
{
	return TmnfCudaVecEnv_LastError();
}

void tmnf_cuda_env_destroy(TmnfPythonCudaEnv *env)
{
	if (env == NULL)
		return;
	TmnfCudaVecEnv_Destroy(env->env);
	TmnfRoute_Unload(env->route);
	World_Destroy(env->template_world);
	TmnfTrack_Unload(env->track);
	free(env->flat_transitions);
	free(env->flat_final);
	free(env->results);
	free(env->observations);
	free(env);
}

/* Flattens count host observations into the caller's rows with the public
 * flatten, so the rows are the CPU env's bytes for the same state. */
static void flatten_rows(
	TmnfPythonCudaEnv *env, uint32_t count, float *flat_observations)
{
	if (flat_observations == NULL)
		ffi_fail("flat observation output is null");
	for (uint32_t i = 0; i < count; ++i) {
		memset(&env->results[i], 0, sizeof(env->results[i]));
		env->results[i].observation = env->observations[i];
	}
	TmnfVecEnv_FlattenStepResults(
		env->results, count, flat_observations, env->flat_final,
		env->flat_transitions);
}

void tmnf_cuda_env_reset(TmnfPythonCudaEnv *env, float *flat_observations)
{
	if (env == NULL)
		ffi_fail("reset argument is null");
	TmnfCudaVecEnv_Reset(env->env, NULL, env->observations);
	flatten_rows(env, env->count, flat_observations);
}

void tmnf_cuda_env_observe_gates(TmnfPythonCudaEnv *env, void *gates)
{
 if (!env) ffi_fail("gate query environment is null");
 TmnfCudaVecEnv_ObserveGatesDevice(env->env, gates);
}
void tmnf_cuda_env_step_with_gates(TmnfPythonCudaEnv *env,
 const void *actions, uint32_t repeat, void *gates, void *final_gates)
{
 if (!env) ffi_fail("gate step environment is null");
 if (env->action_space == TMNF_ACTION_SPACE_ANALOG)
  TmnfCudaVecEnv_StepAnalogDeviceWithGates(env->env, actions, repeat, gates, final_gates);
 else
  TmnfCudaVecEnv_StepDiscreteDeviceWithGates(env->env, actions, repeat, gates, final_gates);
}

void tmnf_cuda_env_step_discrete(
	TmnfPythonCudaEnv *env, const void *device_actions, uint32_t action_repeat)
{
	if (env == NULL || device_actions == NULL || action_repeat == 0)
		ffi_fail("discrete step argument is invalid");
	if (env->action_space != TMNF_ACTION_SPACE_DISCRETE)
		ffi_fail("discrete step on an analog environment");
	TmnfCudaVecEnv_StepDiscreteDevice(
		env->env, (const uint8_t *)device_actions, action_repeat);
}

void tmnf_cuda_env_step_analog(
	TmnfPythonCudaEnv *env, const void *device_actions, uint32_t action_repeat)
{
	if (env == NULL || device_actions == NULL || action_repeat == 0)
		ffi_fail("analog step argument is invalid");
	if (env->action_space != TMNF_ACTION_SPACE_ANALOG)
		ffi_fail("analog step on a discrete environment");
	TmnfCudaVecEnv_StepAnalogDevice(
		env->env, (const TmnfAnalogAction *)device_actions, action_repeat);
}

const void *tmnf_cuda_env_device_results(const TmnfPythonCudaEnv *env)
{
	if (env == NULL)
		ffi_fail("device results requested from null environment");
	return TmnfCudaVecEnv_DeviceResults(env->env);
}

const float *tmnf_cuda_env_device_observations(const TmnfPythonCudaEnv *env)
{
	if (env == NULL)
		ffi_fail("device observations requested from null environment");
	return TmnfCudaVecEnv_DeviceObservations(env->env);
}

const float *tmnf_cuda_env_device_final_observations(
	const TmnfPythonCudaEnv *env)
{
	if (env == NULL)
		ffi_fail("device final observations requested from null environment");
	return TmnfCudaVecEnv_DeviceFinalObservations(env->env);
}

const float *tmnf_cuda_env_device_transitions(const TmnfPythonCudaEnv *env)
{
	if (env == NULL)
		ffi_fail("device transitions requested from null environment");
	return TmnfCudaVecEnv_DeviceTransitions(env->env);
}

void tmnf_cuda_env_capture(
	TmnfPythonCudaEnv *env, const uint32_t *indices, uint32_t count,
	void *snapshots)
{
	if (env == NULL || indices == NULL || count == 0 || snapshots == NULL)
		ffi_fail("capture argument is null or empty");
	TmnfCudaVecEnv_CaptureIndices(
		env->env, indices, count, (TmnfEnvSnapshot *)snapshots);
}

void tmnf_cuda_env_restore(
	TmnfPythonCudaEnv *env, const uint32_t *indices, uint32_t count,
	const void *snapshots, float *flat_observations)
{
	if (env == NULL || indices == NULL || count == 0 || snapshots == NULL)
		ffi_fail("restore argument is null or empty");
	TmnfCudaVecEnv_RestoreIndices(
		env->env, indices, count, (const TmnfEnvSnapshot *)snapshots,
		env->observations);
	flatten_rows(env, count, flat_observations);
}

uint32_t tmnf_cuda_env_count(const TmnfPythonCudaEnv *env)
{
	if (env == NULL)
		ffi_fail("count requested from null environment");
	return env->count;
}

float tmnf_cuda_env_route_length(const TmnfPythonCudaEnv *env)
{
	if (env == NULL)
		ffi_fail("route length requested from null environment");
	return TmnfRoute_GetReferenceLength(env->route);
}

uint32_t tmnf_cuda_env_checkpoint_count(const TmnfPythonCudaEnv *env)
{
	if (env == NULL)
		ffi_fail("checkpoint count requested from null environment");
	return env->route->metadata->checkpoint_count;
}

uint32_t tmnf_cuda_env_lap_count(const TmnfPythonCudaEnv *env)
{
	if (env == NULL)
		ffi_fail("lap count requested from null environment");
	return env->route->metadata->lap_count;
}
