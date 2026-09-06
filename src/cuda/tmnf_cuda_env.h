/* CUDA vectorised environment: TmnfVecEnv semantics, one GPU thread per
 * environment, byte-exact with the CPU engine (docs/CUDA.md).
 *
 * Every environment is a byte copy of one host TmnfWorld relinked on the
 * device against a shared immutable track, vehicle blob and route plus
 * fixed-capacity per-environment scratch. Snapshots are TmnfEnvSnapshot v6,
 * interchangeable with TmnfVecEnv. Host-pointer arguments are copied across;
 * the Device* accessors expose the resident result buffers for zero-copy
 * consumers (a PyTorch tensor built from the raw device pointer).
 *
 * Streams and buffers. Every launch and copy runs on the stream given at
 * Create (TmnfCudaVecEnvLimits.stream, legacy default stream if NULL) and
 * every call returns after cudaStreamSynchronize on it; nothing else on the
 * device is waited for. The Device* buffers (results, observations,
 * transitions) are single-buffered: each is one allocation owned by the env,
 * valid until Destroy, and overwritten in place by the next step or reset,
 * so a consumer must have read (or copied) them before the next call. Actions
 * passed by device pointer are read by the kernel during the call only.
 *
 * Failures. Create returns NULL with TmnfCudaVecEnv_LastError when the device
 * refuses the stack reservation or an allocation (out of memory on a shared
 * card). Everything else fails fast: a CUDA error or a device-side capacity
 * overrun (tmnf_fail -> __trap) aborts the process with a message; a trap
 * leaves the CUDA context unusable, so a long-lived host needs a supervisor.
 * Create also sets the process-wide device stack limit (stack_bytes).
 */
#ifndef TMNF_CUDA_ENV_H
#define TMNF_CUDA_ENV_H

#include <stdint.h>

#include "route.h"
#include "track.h"
#include "vec_env.h"
#include "world.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TmnfCudaVecEnv TmnfCudaVecEnv;

/* Per-environment scratch caps. The game's initial CFastBuffer capacity is
 * 0x32 records; the host grows on demand, the device traps. */
typedef struct {
	uint32_t collision_capacity;    /* zone collision records per env */
	uint32_t contact_capacity;      /* per-sphere contact records per child */
	uint32_t replacement_capacity;  /* penetration corrections per env */
	uint32_t threads_per_block;
	/* Device stack per thread (cudaLimitStackSize), a process-wide setting
	 * the driver backs with resident memory for every thread the card can
	 * hold: about 240 MB per KB on an RTX 5090. The default is the measured
	 * requirement of the kernel with a margin; a smaller value fails a launch. */
	uint32_t stack_bytes;
	/* cudaStream_t every launch and host <-> device copy is ordered on and
	 * waited for (cudaStreamSynchronize) before a call returns; NULL is the
	 * legacy default stream. The caller owns the stream and must keep it
	 * alive until Destroy. A consumer that produces device_actions or reads
	 * the Device* buffers on a stream of its own passes that stream here;
	 * work on other streams is not ordered against the env. */
	struct CUstream_st *stream;
} TmnfCudaVecEnvLimits;

TmnfCudaVecEnvLimits TmnfCudaVecEnv_DefaultLimits(void);

/*
 * Builds count environments from template_world (a host world whose spawn
 * tick has run, as World_Create leaves it). route may be NULL for raw physics
 * stepping only; the RL entry points then abort. config follows
 * TmnfVecEnv_DefaultConfig; thread_count is ignored.
 */
/* Returns NULL, with the reason in TmnfCudaVecEnv_LastError, when the device
 * refuses the stack reservation or an allocation (typically out of memory
 * on a shared card); every other failure, before and after Create, aborts. */
TmnfCudaVecEnv *TmnfCudaVecEnv_Create(
	const TmnfTrack *track,
	const TmnfWorld *template_world,
	const TmnfRoute *route,
	uint32_t count,
	const TmnfVecEnvConfig *config,
	const TmnfCudaVecEnvLimits *limits);
void TmnfCudaVecEnv_Destroy(TmnfCudaVecEnv *env);
const char *TmnfCudaVecEnv_LastError(void);
uint32_t TmnfCudaVecEnv_Count(const TmnfCudaVecEnv *env);

/* Raw physics: TmnfVecEnv_Step. inputs has count entries; observations may
 * be NULL. */
void TmnfCudaVecEnv_Step(
	TmnfCudaVecEnv *env, const TMNFRaceInputs *inputs, uint32_t tick_ms,
	TmnfObservation *observations);

/* Raw physics with count inputs already on the device. No observations,
 * race rules, or transfers; completion follows the same stream contract. */
void TmnfCudaVecEnv_StepDevice(
	TmnfCudaVecEnv *env, const TMNFRaceInputs *device_inputs, uint32_t tick_ms);

/* RL layer: TmnfVecEnv_Reset / StepDiscrete / StepAnalog. Host results may
 * be NULL when the caller reads the device buffers instead. */
void TmnfCudaVecEnv_Reset(
	TmnfCudaVecEnv *env, const uint8_t *mask,
	TmnfObservation *observations);
void TmnfCudaVecEnv_StepDiscrete(
	TmnfCudaVecEnv *env, const uint8_t *actions, uint32_t action_repeat,
	TmnfStepResult *results);
void TmnfCudaVecEnv_StepAnalog(
	TmnfCudaVecEnv *env, const TmnfAnalogAction *actions,
	uint32_t action_repeat, TmnfStepResult *results);

/* Same steps with actions already resident on the device (count entries). */
void TmnfCudaVecEnv_StepDiscreteDevice(
	TmnfCudaVecEnv *env, const uint8_t *device_actions,
	uint32_t action_repeat);
void TmnfCudaVecEnv_StepAnalogDevice(
	TmnfCudaVecEnv *env, const TmnfAnalogAction *device_actions,
	uint32_t action_repeat);

/* Optional caller-owned device sidecars, count entries each, on the env
 * stream. Semantics match CPU WithGates. No persistent allocation is added.
 * Arrays must be distinct and remain valid through stream completion. */
void TmnfCudaVecEnv_StepDiscreteDeviceWithGates(TmnfCudaVecEnv *env,
 const uint8_t *actions, uint32_t repeat,
 TmnfGateObservations *gates, TmnfGateObservations *final_gates);
void TmnfCudaVecEnv_StepAnalogDeviceWithGates(TmnfCudaVecEnv *env,
 const TmnfAnalogAction *actions, uint32_t repeat,
 TmnfGateObservations *gates, TmnfGateObservations *final_gates);
void TmnfCudaVecEnv_ObserveGatesDevice(TmnfCudaVecEnv *env,
 TmnfGateObservations *gates);

/* Device-resident outputs of the last RL step, count entries each. The flat
 * arrays follow TmnfVecEnv_FlattenStepResults: observations and final
 * observations are [count][TMNF_POLICY_OBSERVATION_WIDTH] float32,
 * transitions are [count][TMNF_POLICY_TRANSITION_WIDTH]. */
const TmnfStepResult *TmnfCudaVecEnv_DeviceResults(const TmnfCudaVecEnv *env);
const float *TmnfCudaVecEnv_DeviceObservations(const TmnfCudaVecEnv *env);
const float *TmnfCudaVecEnv_DeviceFinalObservations(
	const TmnfCudaVecEnv *env);
const float *TmnfCudaVecEnv_DeviceTransitions(const TmnfCudaVecEnv *env);

/* Snapshots: TmnfEnvSnapshot v6, byte-compatible with TmnfVecEnv. */
void TmnfCudaVecEnv_Capture(TmnfCudaVecEnv *env, TmnfEnvSnapshot *snapshots);
void TmnfCudaVecEnv_Restore(
	TmnfCudaVecEnv *env, const TmnfEnvSnapshot *snapshots);
void TmnfCudaVecEnv_CaptureIndices(
	TmnfCudaVecEnv *env, const uint32_t *indices, uint32_t count,
	TmnfEnvSnapshot *snapshots);
void TmnfCudaVecEnv_RestoreIndices(
	TmnfCudaVecEnv *env, const uint32_t *indices, uint32_t count,
	const TmnfEnvSnapshot *snapshots, TmnfObservation *observations);

/*
 * Copies environment index's world bytes over destination, a host world built
 * on the same track and vehicle snapshot, and relinks it against the
 * destination's own allocations. Used by the replay harness to read the
 * device state through the host accessors.
 */
void TmnfCudaVecEnv_CopyWorldToHost(
	TmnfCudaVecEnv *env, uint32_t index, TmnfWorld *destination);

/* Raw step split for the replay harness: applies the inputs, copies world
 * index to destination (the record the game exposes between input and
 * step), then advances the timer and steps. Same end state as _Step. */
void TmnfCudaVecEnv_StepWithPreStateCopy(
	TmnfCudaVecEnv *env, const TMNFRaceInputs *inputs, uint32_t tick_ms,
	uint32_t index, TmnfWorld *destination);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_CUDA_ENV_H */
