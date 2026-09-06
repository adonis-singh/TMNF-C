#ifndef TMNF_ENV_FFI_H
#define TMNF_ENV_FFI_H

#include <stddef.h>
#include <stdint.h>

typedef struct TmnfPythonEnv TmnfPythonEnv;

/* Thread-local creation diagnostic; cleared by each create call. Invalid
 * arguments/budgets return NULL. Invalid fixture data still follows loader
 * fatal-error semantics. */
const char *tmnf_env_creation_error(void);

/* Zero retains legacy derivation from reward reference_speed. A positive
 * budget_reference_speed resolves only budgets whose explicit value is zero. */
TmnfPythonEnv *tmnf_env_create_with_budget(
	const char *track_path, const char *vehicle_path, const char *route_path,
	const uint8_t track_sha256[32], uint32_t environment_count,
	uint32_t thread_count, uint32_t action_space, uint32_t max_race_ticks,
	uint32_t horizon_ticks, uint32_t off_track_grace_ticks,
	uint32_t stuck_grace_ticks, float discount_per_tick, float reference_speed,
	float stuck_progress_epsilon, uint32_t respawn_action,
	float budget_reference_speed);

TmnfPythonEnv *tmnf_env_create(
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
	uint32_t respawn_action);
void tmnf_env_destroy(TmnfPythonEnv *env);

void tmnf_env_reset(TmnfPythonEnv *env);
void tmnf_env_step_discrete(
	TmnfPythonEnv *env, uint32_t action_repeat);
void tmnf_env_step_analog(
	TmnfPythonEnv *env, uint32_t action_repeat);
size_t tmnf_gate_observations_size(void);
void tmnf_env_observe_gates(TmnfPythonEnv *env, void *gates);
void tmnf_env_step_with_gates(TmnfPythonEnv *env, uint32_t repeat,
 void *gates, void *final_gates);
void tmnf_env_capture(
	TmnfPythonEnv *env, const uint32_t *indices, uint32_t count,
	void *snapshots);
void tmnf_env_restore(
	TmnfPythonEnv *env, const uint32_t *indices, uint32_t count,
	const void *snapshots);

uint8_t *tmnf_env_discrete_actions(TmnfPythonEnv *env);
void *tmnf_env_analog_actions(TmnfPythonEnv *env);
void *tmnf_env_results(TmnfPythonEnv *env);
uint32_t tmnf_env_count(const TmnfPythonEnv *env);
float tmnf_env_route_length(const TmnfPythonEnv *env);
uint32_t tmnf_env_checkpoint_count(const TmnfPythonEnv *env);
uint32_t tmnf_env_lap_count(const TmnfPythonEnv *env);

uint32_t tmnf_observation_version(void);
size_t tmnf_observation_size(void);
size_t tmnf_step_result_size(void);
size_t tmnf_env_snapshot_size(void);

#endif
