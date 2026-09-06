/* C ABI over TmnfCudaVecEnv for the Python trainer (tmnf_rl.cuda_env).
 *
 * The RL data path stays on the device: actions arrive as device pointers,
 * observations, final observations, transitions and step results are read
 * through the Device* accessors (PyTorch wraps them zero-copy). The host
 * sees snapshots (capture/restore) and the flattened observation rows that
 * a reset or restore produces, because the device env writes those into its
 * observation scratch and not into the flat policy buffer; the Python side
 * scatters the rows back into the device buffer, which is what the CPU FFI
 * does when it patches results[i].observation after a restore.
 */
#ifndef TMNF_CUDA_ENV_FFI_H
#define TMNF_CUDA_ENV_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TmnfPythonCudaEnv TmnfPythonCudaEnv;

/* Returns NULL, with the reason in tmnf_cuda_env_last_error, when the device
 * refuses the stack reservation or an allocation; every other failure aborts.
 * max_race_ticks and horizon_ticks must be the resolved per-track values (a
 * CPU TmnfVectorEnv resolves 0 to them); stream is the cudaStream_t every
 * launch and copy is ordered on (the caller's PyTorch stream). */
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
	void *stream);
const char *tmnf_cuda_env_last_error(void);
void tmnf_cuda_env_destroy(TmnfPythonCudaEnv *env);

/* Resets every environment; flat_observations receives the count x 81 float
 * policy rows of the reset states. */
void tmnf_cuda_env_reset(TmnfPythonCudaEnv *env, float *flat_observations);

/* Steps with actions resident on the device: count bytes (discrete) or count
 * TmnfAnalogAction (analog). */
void tmnf_cuda_env_observe_gates(TmnfPythonCudaEnv *env, void *gates);
void tmnf_cuda_env_step_with_gates(TmnfPythonCudaEnv *env,
 const void *actions, uint32_t repeat, void *gates, void *final_gates);
void tmnf_cuda_env_step_discrete(
	TmnfPythonCudaEnv *env, const void *device_actions, uint32_t action_repeat);
void tmnf_cuda_env_step_analog(
	TmnfPythonCudaEnv *env, const void *device_actions, uint32_t action_repeat);

/* Device pointers, valid until destroy, overwritten by the next step. */
const void *tmnf_cuda_env_device_results(const TmnfPythonCudaEnv *env);
const float *tmnf_cuda_env_device_observations(const TmnfPythonCudaEnv *env);
const float *tmnf_cuda_env_device_final_observations(const TmnfPythonCudaEnv *env);
const float *tmnf_cuda_env_device_transitions(const TmnfPythonCudaEnv *env);

/* Snapshots (TmnfEnvSnapshot, host memory, count entries). restore also
 * writes the count x 81 flat policy rows of the restored states. */
void tmnf_cuda_env_capture(
	TmnfPythonCudaEnv *env, const uint32_t *indices, uint32_t count,
	void *snapshots);
void tmnf_cuda_env_restore(
	TmnfPythonCudaEnv *env, const uint32_t *indices, uint32_t count,
	const void *snapshots, float *flat_observations);

uint32_t tmnf_cuda_env_count(const TmnfPythonCudaEnv *env);
float tmnf_cuda_env_route_length(const TmnfPythonCudaEnv *env);
uint32_t tmnf_cuda_env_checkpoint_count(const TmnfPythonCudaEnv *env);
uint32_t tmnf_cuda_env_lap_count(const TmnfPythonCudaEnv *env);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_CUDA_ENV_FFI_H */
