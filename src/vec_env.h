/* Parallel batched RL-facing wrapper over the single-world bit-exact physics
 * path. A persistent worker pool advances fixed contiguous environment shards.
 * Worlds own independent mutable state and may share immutable track
 * geometry/tuning.
 */
#ifndef TMNF_VEC_ENV_H
#define TMNF_VEC_ENV_H

#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>

#include "physics.h"
#include "race.h"
#include "race_observation.h"
#include "vehicle_model6.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
	TMNF_STADIUM_WHEEL_COUNT = 4,
	TMNF_DISCRETE_ACTION_COUNT = 12,
	/* Optional respawn modifier on a discrete action byte
	 * (TmnfVecEnvConfig.respawn_action): the Enter press on the first tick of
	 * the repeat, the low bits' controls throughout. */
	TMNF_DISCRETE_RESPAWN_FLAG = 0x80,
	TMNF_OBSERVATION_LOOKAHEAD_COUNT = 8,
	TMNF_OBSERVATION_VERSION = 2,
	TMNF_POLICY_OBSERVATION_WIDTH = 81,
	TMNF_POLICY_TRANSITION_WIDTH = 5,
	/* 6: TmnfRaceState carries the respawn location and its armed flag. */
	TMNF_ENV_SNAPSHOT_VERSION = 6,
	/* contact_body_refs value for a wheel without a contact body. */
	TMNF_ENV_NO_CONTACT_BODY = 0xFFFFFFFFu,
};

extern const float TMNF_OBSERVATION_LOOKAHEAD_METERS[
	TMNF_OBSERVATION_LOOKAHEAD_COUNT];

typedef struct {
	GmVec3 position;
	float half_width;
} TmnfCenterlineSample;

typedef struct {
	GmVec3 position;
	GmQuat rotation;
	GmVec3 linear_speed;
	GmVec3 angular_speed;
	float wheel_speed[TMNF_STADIUM_WHEEL_COUNT];
	float wheel_damper[TMNF_STADIUM_WHEEL_COUNT];
	float wheel_contact[TMNF_STADIUM_WHEEL_COUNT];
	float wheel_sliding[TMNF_STADIUM_WHEEL_COUNT];
	float wheel_material[TMNF_STADIUM_WHEEL_COUNT];
	float engine_rpm;
	int32_t gear;
	float input_steer;
	float input_gas;
	float input_brake;
	float arc_length;
	float unwrapped_progress;
	float lateral_offset;
	float track_half_width;
	float remaining_distance;
	float elapsed_fraction;
	float next_checkpoint_fraction;
	float completed_lap_fraction;
	float turbo_active;
	float turbo_type;
	float turbo_remaining_progress;
	TmnfCenterlineSample
		centerline_lookahead[TMNF_OBSERVATION_LOOKAHEAD_COUNT];
} TmnfObservation;

_Static_assert(offsetof(TmnfObservation, gear) == 136,
	"TmnfObservation gear offset");
_Static_assert(offsetof(TmnfObservation, input_steer) == 140,
	"TmnfObservation race offset");
_Static_assert(offsetof(TmnfObservation, turbo_active) == 184,
	"TmnfObservation turbo offset");
_Static_assert(offsetof(TmnfObservation, centerline_lookahead) == 196,
	"TmnfObservation centerline offset");
_Static_assert(sizeof(TmnfObservation) == 324,
	"TmnfObservation size");

/*
 * A snapshot is a pure state transfer: it holds no address of the source
 * environment. Every pointer field inside the copied structs is zero in the
 * snapshot and re-derived from the destination world on restore; the one
 * pointer that is state, the wheel's contact body, is stored as the body's
 * corpus_ref token and resolved through the destination's response zone.
 *
 * dyna_params is state, not tuning: the vehicle compute stage rewrites
 * forceFieldScale and dragLinear every tick from the grounded flag, and the
 * next tick's gravity and drag read them before anything recomputes them.
 */
typedef struct {
	uint32_t version;
	CHmsStateDyna live_state;
	CHmsStateDyna committed_state;
	CHmsStateDyna temp_state;
	CHmsDynaParams dyna_params;
	CSceneVehicleCar car;
	CSceneVehicleCarWheelSnapshot wheels[TMNF_STADIUM_WHEEL_COUNT];
	CSceneVehicleCarAuxContext aux;
	CSceneVehicleCarWheelAux aux_wheels[TMNF_STADIUM_WHEEL_COUNT];
	TMNFVehicleContactContext contact;
	TMNFVehicleContactWheelState
		contact_wheels[TMNF_STADIUM_WHEEL_COUNT];
	uint32_t contact_body_refs[TMNF_STADIUM_WHEEL_COUNT];
	CSceneVehicleCarModel6State model6_state;
	TMNFVehicleComputeState compute_state;
	uint32_t contact_timer_tick;
	uint64_t physics_tick_time;
	int32_t dyna_dirty;
	TmnfRaceState race_state;
	uint64_t episode_id;
	float episode_return;
	uint8_t race_state_valid;
	uint8_t reset_pending;
	uint8_t vehicle_context_valid;
	uint8_t reserved;
} TmnfEnvSnapshot;

typedef enum {
	TMNF_AUTORESET_SAME_STEP = 0,
	TMNF_AUTORESET_NEXT_STEP = 1,
} TmnfAutoresetMode;

typedef enum {
	TMNF_ACTION_SPACE_DISCRETE = 0,
	TMNF_ACTION_SPACE_ANALOG = 1,
} TmnfActionSpace;

/* respawn is honoured only with TmnfVecEnvConfig.respawn_action; a nonzero
 * byte without it aborts. */
typedef struct {
	float steer;
	uint8_t gas;
	uint8_t brake;
	uint8_t respawn;
	uint8_t reserved;
} TmnfAnalogAction;

_Static_assert(offsetof(TmnfAnalogAction, steer) == 0,
	"TmnfAnalogAction steer offset");
_Static_assert(offsetof(TmnfAnalogAction, gas) == 4,
	"TmnfAnalogAction gas offset");
_Static_assert(offsetof(TmnfAnalogAction, brake) == 5,
	"TmnfAnalogAction brake offset");
_Static_assert(offsetof(TmnfAnalogAction, respawn) == 6,
	"TmnfAnalogAction respawn offset");
_Static_assert(sizeof(TmnfAnalogAction) == 8,
	"TmnfAnalogAction size");

typedef enum {
	TMNF_TERMINATION_NONE = 0,
	TMNF_TERMINATION_FINISH = 1,
	TMNF_TERMINATION_TIMEOUT = 2,
	TMNF_TERMINATION_OFF_TRACK = 3,
	TMNF_TERMINATION_STUCK = 4,
	/* Car below the lowest route sample by more than one block (16 m). */
	TMNF_TERMINATION_FELL = 5,
	/* Respawn pressed before any respawnable checkpoint: the game restarts
	 * the race (0x00472700 SmallRespawn -> CGameRace::SetStatus(3)), so the
	 * episode ends as a failure. */
	TMNF_TERMINATION_RESTART = 6,
} TmnfTerminationReason;

/* World floor: this far below the lowest centerline sample is off the map. */
#define TMNF_FELL_MARGIN_METERS 16.0f

/* Game rigid-body speed cap (scene_mobil.max_linear_speed, 1,000 km/h). */
#define TMNF_MAX_LINEAR_SPEED_MPS 277.77777099609375
/* Derived horizon: this many times the route at reference_speed. */
#define TMNF_HORIZON_PACE_FACTOR 2.0

/* Resolve a learning budget independently of reward normalization. Use the
 * result for either zero/automatic budget before initialization. */
uint32_t TmnfVecEnv_DerivePaceTicks(const TmnfRoute *route, float reference_speed);

/*
 * max_race_ticks == 0 and horizon_ticks == 0 each derive from the route:
 * TMNF_HORIZON_PACE_FACTOR * laps * route length / reference_speed. There is
 * no clamp: a fixed cap made long tracks unfinishable (D01 needed a 546 km/h
 * average inside 12,000 ticks). Explicit values are accepted. Both budgets
 * must allow a finish at the game's speed cap or initialization aborts.
 */
typedef struct {
	uint32_t max_race_ticks;
	uint32_t horizon_ticks;
	uint32_t off_track_grace_ticks;
	uint32_t stuck_grace_ticks;
	uint32_t thread_count;
	float discount_per_tick;
	float reference_speed;
	/* Stuck: |progress delta| <= epsilon for stuck_grace_ticks consecutive
	 * ticks, whatever the speed. */
	float stuck_progress_epsilon;
	TmnfAutoresetMode autoreset_mode;
	TmnfActionSpace action_space;
	/* Nonzero enables the respawn press as an action: the
	 * TMNF_DISCRETE_RESPAWN_FLAG bit and TmnfAnalogAction.respawn. Zero keeps
	 * the twelve-action and steer/gas/brake sets unchanged. */
	uint32_t respawn_action;
} TmnfVecEnvConfig;

typedef struct {
	TmnfObservation observation;
	TmnfObservation final_observation;
	float reward;
	float transition_discount;
	float completed_episode_return;
	uint32_t completed_episode_ticks;
	uint32_t race_time_ms;
	uint32_t executed_ticks;
	uint64_t episode_id;
	uint8_t terminated;
	uint8_t truncated;
	uint8_t final_observation_valid;
	uint8_t reset_only;
	TmnfTerminationReason termination_reason;
} TmnfStepResult;

_Static_assert(offsetof(TmnfStepResult, final_observation) == 324,
	"TmnfStepResult final observation offset");
_Static_assert(offsetof(TmnfStepResult, reward) == 648,
	"TmnfStepResult reward offset");
_Static_assert(offsetof(TmnfStepResult, episode_id) == 672,
	"TmnfStepResult episode offset");
_Static_assert(sizeof(TmnfStepResult) == 688,
	"TmnfStepResult size");

typedef struct {
	TmnfPhysicsWorld **worlds;
	uint32_t *player_corpus_indices;
	uint32_t count;
	const TmnfRoute *route;
	TmnfRaceState *race_states;
	TmnfEnvSnapshot *reset_snapshots;
	uint64_t *episode_ids;
	float *episode_returns;
	uint8_t *reset_pending;
	TmnfVecEnvConfig config;
	/* Lowest centerline sample minus TMNF_FELL_MARGIN_METERS. */
	float fell_floor_y;
	void *worker_pool;
	_Atomic uint32_t active_operation;
} TmnfVecEnv;

/*
 * Initializes the RL layer over caller-owned worlds and immediately resets all
 * environments to route->start. Destroy releases only vec-env allocations.
 *
 * One external caller may operate on an environment at a time. Reset, step,
 * capture, restore, thread-count changes, empty dispatch, and destroy must not
 * overlap on the same TmnfVecEnv. Concurrent use is a caller bug and aborts.
 */
void TmnfVecEnv_Init(
	TmnfVecEnv *env,
	TmnfPhysicsWorld **worlds,
	uint32_t *player_corpus_indices,
	uint32_t count,
	const TmnfRoute *route,
	const TmnfVecEnvConfig *config);
void TmnfVecEnv_Destroy(TmnfVecEnv *env);

/*
 * SAME_STEP is the default convention in TmnfVecEnv_DefaultConfig. On an
 * ending step, observation is the reset observation and final_observation is
 * the transition's true next observation.
 *
 * NEXT_STEP returns the terminal observation on the ending step. The following
 * call ignores that environment's action and returns the eagerly buffered
 * reset state with reward 0 and both end flags clear.
 *
 * Both modes reset eagerly and always publish final_observation on the ending
 * step. Consumers must use final_observation for termination and truncation
 * targets whenever final_observation_valid is set.
 */
TmnfVecEnvConfig TmnfVecEnv_DefaultConfig(void);
/* Builds or rebuilds the persistent pool. Call only while no other env
 * operation is active. thread_count includes the caller and must be positive.
 * TmnfVecEnv_Init calls this automatically; manually assembled raw physics
 * batches must call it before TmnfVecEnv_Step. */
void TmnfVecEnv_SetThreadCount(TmnfVecEnv *env, uint32_t thread_count);
void TmnfVecEnv_Reset(
	TmnfVecEnv *env,
	const uint8_t *mask,
	TmnfObservation *observations);

/*
 * Actions are [longitudinal * 3 + steering]: longitudinal is coast, gas,
 * brake, or gas+brake; steering is left, neutral, or right. With
 * config.respawn_action, TMNF_DISCRETE_RESPAWN_FLAG on the byte respawns the
 * car on the first repeated tick (analysis/respawn.md). One call holds the
 * action for action_repeat canonical 10 ms physics ticks. Reward is discounted
 * across repeated ticks and transition_discount is gamma^executed_ticks; use
 * that value for the transition's bootstrap term.
 */
void TmnfVecEnv_StepDiscrete(
	TmnfVecEnv *env,
	const uint8_t *actions,
	uint32_t action_repeat,
	TmnfStepResult *results);

/*
 * Analog steering is quantized to the game's signed integer range, then
 * applied through the native analog input branch. Gas and brake remain binary.
 */
int32_t TmnfVecEnv_QuantizeAnalogSteer(float steer);
void TmnfVecEnv_StepAnalog(
	TmnfVecEnv *env,
	const TmnfAnalogAction *actions,
	uint32_t action_repeat,
	TmnfStepResult *results);

/* Optional gate sidecars have env->count entries. Both arrays are required.
 * Current gates follow the returned observation (including NEXT_STEP terminal
 * observations); final gates are zero unless final_observation_valid is set.
 * No sidecar is retained internally and legacy step/snapshot layouts are unchanged.
 */
void TmnfVecEnv_StepDiscreteWithGates(TmnfVecEnv *env,
 const uint8_t *actions, uint32_t action_repeat, TmnfStepResult *results,
 TmnfGateObservations *gates, TmnfGateObservations *final_gates);
void TmnfVecEnv_StepAnalogWithGates(TmnfVecEnv *env,
 const TmnfAnalogAction *actions, uint32_t action_repeat, TmnfStepResult *results,
 TmnfGateObservations *gates, TmnfGateObservations *final_gates);
/* Query actual current state after Reset/Restore; NEXT_STEP eagerly resets,
 * so this query intentionally differs from that ending step's observation. */
void TmnfVecEnv_ObserveGates(TmnfVecEnv *env, TmnfGateObservations *gates);

/* Snapshot version 6 captures/restores every mutable field represented by the
 * current native car/dynamics port (including the per-tick CHmsDynaParams
 * fields) plus race progression (with the corridor and respawn state), reward,
 * episode, and autoreset state.
 * Immutable route/track geometry, tuning, and pointer topology remain owned by
 * the caller. */
void TmnfVecEnv_Capture(
	TmnfVecEnv *env, TmnfEnvSnapshot *snapshots);
void TmnfVecEnv_Restore(
	TmnfVecEnv *env, const TmnfEnvSnapshot *snapshots);
/* Indexed variants pack snapshots and optional observations in index order.
 * Indices must be unique and less than env->count. */
void TmnfVecEnv_CaptureIndices(
	TmnfVecEnv *env, const uint32_t *indices, uint32_t count,
	TmnfEnvSnapshot *snapshots);
void TmnfVecEnv_RestoreIndices(
	TmnfVecEnv *env, const uint32_t *indices, uint32_t count,
	const TmnfEnvSnapshot *snapshots, TmnfObservation *observations);

/* Advances the corpus' game timer (CMwTimerAdapter tick read by the vehicle
 * contact, compute, and model6 stages) by tick_ms. Every physics tick must be
 * preceded by exactly one advance; World_AdvanceTimer and the vec-env step
 * paths both go through here. */
void TmnfPhysicsCorpus_AdvanceTimer(
	TmnfPhysicsCorpus *corpus, uint32_t tick_ms);

/* Applies one exact TMNF input packet and advances each world by tick_ms.
 * Arrays contain env->count elements. A packet with respawn set respawns the
 * car first (the race layer must be initialized and a checkpoint passed). */
void TmnfVecEnv_Step(
	TmnfVecEnv *env, const TMNFRaceInputs *inputs,
	uint32_t tick_ms, TmnfObservation *observations);

/* Dispatches one empty job through the persistent worker pool. This isolates
 * pool wakeup and completion latency for benchmark diagnostics. */
void TmnfVecEnv_DispatchEmpty(TmnfVecEnv *env);

/* Packs strided result records into contiguous float32 policy inputs. The
 * transition columns are reward, discount, terminated, truncated, and
 * executed physics ticks. */
void TmnfVecEnv_FlattenStepResults(
	const TmnfStepResult *results,
	uint32_t count,
	float *observations,
	float *final_observations,
	float *transitions);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_VEC_ENV_H */
