#ifndef TMNF_VEHICLE_H
#define TMNF_VEHICLE_H

#include "tmnf_hd.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "gm.h"
#include "hms_state.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
	TMNF_CSCENE_VEHICLE_CAR_GAME_SIZE = 0x878,
	TMNF_CSCENE_VEHICLE_CAR_TUNING_GAME_SIZE = 0x3ac,
	TMNF_CSCENE_VEHICLE_CAR_WHEEL_GAME_SIZE = 0x2fc,

	TMNF_CAR_GAME_HMS_ITEM_OFFSET = 0x028,
	TMNF_CAR_GAME_INPUT_GAS_OFFSET = 0x050,
	TMNF_CAR_GAME_INPUT_BRAKE_OFFSET = 0x054,
	TMNF_CAR_GAME_INPUT_STEER_OFFSET = 0x058,
	TMNF_CAR_GAME_TUNING_OFFSET = 0x060,
	TMNF_CAR_GAME_WHEEL_BUFFER_OFFSET = 0x2e8,
	TMNF_CAR_GAME_ENGINE_OFFSET = 0x59c,
	TMNF_CAR_GAME_CURRENT_LOCAL_SPEED_OFFSET = 0x70c,
	TMNF_CAR_GAME_FORCE_SUM_OFFSET = 0x818,
	TMNF_CAR_GAME_IMPULSE_SUM_OFFSET = 0x824,

	TMNF_TUNING_GAME_ACCEL_CURVE_OFFSET = 0x034,
	TMNF_TUNING_GAME_DAMPER_MAX_OFFSET = 0x11c,
	TMNF_TUNING_GAME_DAMPER_MIN_OFFSET = 0x120,
	TMNF_TUNING_GAME_DAMPER_CURVE_OFFSET = 0x224,
	TMNF_TUNING_GAME_GEAR_RATIOS_OFFSET = 0x2c4,
};

/* 0x48-byte input packet consumed by the static 0x004FE500 mapper. The
 * mapper reads the eight timestamp/value words only; word +0x40 is never
 * read by the game and carries the race-level respawn press (Enter,
 * CTrackManiaRace::OnInputEvent -> SmallRespawn, analysis/respawn.md):
 * nonzero on the tick the press edge applies. */
typedef struct {
	uint32_t steer_left_time;       /* 0x00 */
	uint32_t reserved04;
	int32_t steer_left;             /* 0x08 */
	uint32_t steer_right_time;      /* 0x0c */
	uint32_t reserved10;
	int32_t steer_right;            /* 0x14 */
	uint32_t steer_analog_time;     /* 0x18 */
	uint32_t reserved1c;
	float steer_analog;             /* 0x20 */
	uint32_t accelerate_time;       /* 0x24 */
	uint32_t reserved28;
	int32_t accelerate;             /* 0x2c */
	uint32_t brake_time;            /* 0x30 */
	uint32_t reserved34;
	int32_t brake;                  /* 0x38 */
	uint32_t gas_analog_time;       /* 0x3c */
	uint32_t respawn;               /* 0x40, engine-defined */
	float gas_analog;               /* 0x44 */
} TMNFRaceInputs;

_Static_assert(sizeof(TMNFRaceInputs) == 0x48, "TMNFRaceInputs size");
_Static_assert(offsetof(TMNFRaceInputs, steer_left) == 0x08, "steer left");
_Static_assert(offsetof(TMNFRaceInputs, steer_right_time) == 0x0c,
	"steer right time");
_Static_assert(offsetof(TMNFRaceInputs, steer_right) == 0x14, "steer right");
_Static_assert(offsetof(TMNFRaceInputs, steer_analog_time) == 0x18,
	"steer analog time");
_Static_assert(offsetof(TMNFRaceInputs, steer_analog) == 0x20, "steer analog");
_Static_assert(offsetof(TMNFRaceInputs, accelerate_time) == 0x24,
	"accelerate time");
_Static_assert(offsetof(TMNFRaceInputs, accelerate) == 0x2c, "accelerate");
_Static_assert(offsetof(TMNFRaceInputs, brake_time) == 0x30, "brake time");
_Static_assert(offsetof(TMNFRaceInputs, brake) == 0x38, "brake");
_Static_assert(offsetof(TMNFRaceInputs, gas_analog_time) == 0x3c,
	"gas analog time");
_Static_assert(offsetof(TMNFRaceInputs, respawn) == 0x40, "respawn");
_Static_assert(offsetof(TMNFRaceInputs, gas_analog) == 0x44, "gas analog");

/* Pure-data engine state embedded at CSceneVehicleCar+0x59C. */
typedef struct {
	float max_rpm;                  /* 0x00 */
	uint8_t reserved04[0x10];
	float braking_factor;           /* 0x14 */
	float rpm;                      /* 0x18 */
	float target_rpm;               /* 0x1c */
	float clutch;                   /* 0x20 */
	float shift_timer;              /* 0x24 */
	int32_t reverse;                /* 0x28 */
	int32_t gear;                   /* 0x2c */
} CSceneVehicleCarEngine;

_Static_assert(sizeof(CSceneVehicleCarEngine) == 0x30, "engine size");
_Static_assert(offsetof(CSceneVehicleCarEngine, braking_factor) == 0x14,
	"engine braking factor");
_Static_assert(offsetof(CSceneVehicleCarEngine, reverse) == 0x28,
	"engine reverse");
_Static_assert(offsetof(CSceneVehicleCarEngine, gear) == 0x2c, "engine gear");

/* Pure-data real-time wheel state embedded at wheel+0xB4. */
typedef struct {
	float damper_absorb;            /* 0x00 */
	float field04;
	float field08;
	GmMat3 basis0;                  /* 0x0c */
	GmMat3 basis1;                  /* 0x30 */
	GmVec3 field54;                 /* 0x54 */
	uint8_t reserved60[0x0c];
	float field6c;
	int32_t has_ground_contact;     /* 0x70 */
	int32_t contact_material_id;    /* 0x74 */
	int32_t is_sliding;             /* 0x78 */
	GmVec3 relative_rotz_axis;      /* 0x7c */
	uint32_t contact_body;          /* 0x88, game 32-bit pointer */
	int32_t ground_contact_count;   /* 0x8c */
	GmVec3 field90;
	float rotation_phase;           /* 0x9c */
	float blend_value;              /* 0xa0 */
	float blend_target;             /* 0xa4 */
} CSceneVehicleCarWheelRealTimeState;

_Static_assert(sizeof(CSceneVehicleCarWheelRealTimeState) == 0xa8,
	"wheel real-time state size");
_Static_assert(offsetof(CSceneVehicleCarWheelRealTimeState, basis0) == 0x0c,
	"wheel basis0");
_Static_assert(offsetof(CSceneVehicleCarWheelRealTimeState, basis1) == 0x30,
	"wheel basis1");
_Static_assert(offsetof(CSceneVehicleCarWheelRealTimeState, has_ground_contact) == 0x70,
	"wheel ground contact");
_Static_assert(offsetof(CSceneVehicleCarWheelRealTimeState, relative_rotz_axis) == 0x7c,
	"wheel relative rotation");
_Static_assert(offsetof(CSceneVehicleCarWheelRealTimeState, contact_body) == 0x88,
	"wheel contact body");
_Static_assert(offsetof(CSceneVehicleCarWheelRealTimeState, ground_contact_count) == 0x8c,
	"wheel contact count");
_Static_assert(offsetof(CSceneVehicleCarWheelRealTimeState, rotation_phase) == 0x9c,
	"wheel rotation phase");
_Static_assert(offsetof(CSceneVehicleCarWheelRealTimeState, blend_target) == 0xa4,
	"wheel blend target");

typedef struct {
	uint8_t bytes[0x64];
} CSceneVehicleCarWheelState;

_Static_assert(sizeof(CSceneVehicleCarWheelState) == 0x64, "wheel state size");

typedef struct {
	float stiffness;                /* 0x00 */
	float damping;                  /* 0x04 */
	float value;                    /* 0x08 */
	float target;                   /* 0x0c */
	float velocity;                 /* 0x10 */
} GmSpringFloat;

_Static_assert(sizeof(GmSpringFloat) == 0x14, "spring size");
_Static_assert(offsetof(GmSpringFloat, velocity) == 0x10, "spring velocity");

/*
 * Native pointer-bearing wheel representation. Game offsets are:
 * +0x00 active, +0x04 steerable, +0x08 field08, +0x0C surface handler,
 * +0x70 field70, +0xA0/+0xA4 scalar state, +0xA8 vehicle offset,
 * +0xB4 real-time state, +0x15C field15c, +0x160 contact-relative distance,
 * and four 0x64-byte wheel states at +0x16C, +0x1D0, +0x234, +0x298.
 */
typedef struct {
	CSceneVehicleCarWheelState previous_sync;
	CSceneVehicleCarWheelState sync;
	CSceneVehicleCarWheelState field234;
	CSceneVehicleCarWheelState async_state;
	uint8_t snapshot_padding[4]; /* preserve v6 trailing bytes on restore */
} CSceneVehicleCarWheelHistory;

typedef struct {
	int32_t active;
	int32_t steerable;
	float radius;
	void *surface_handler;
	float field70[12];
	int32_t fielda0;
	int32_t fielda4;
	GmVec3 offset_from_vehicle;
	CSceneVehicleCarWheelRealTimeState real_time;
	int32_t field15c;
	GmVec3 contact_relative_local_distance;
	/* Synchronization history is read only by capture/export and written by
	 * restore/respawn. Keep its 400 bytes outside the active physics state. */
	CSceneVehicleCarWheelHistory *history;
} CSceneVehicleCarWheel;

/* Snapshot v6 retains the original 680-byte wheel layout, independently of
 * the internal simulation layout. The prefix is byte-identical. */
typedef struct {
	int32_t active;
	int32_t steerable;
	float radius;
	void *surface_handler;
	float field70[12];
	int32_t fielda0;
	int32_t fielda4;
	GmVec3 offset_from_vehicle;
	CSceneVehicleCarWheelRealTimeState real_time;
	int32_t field15c;
	GmVec3 contact_relative_local_distance;
	CSceneVehicleCarWheelState previous_sync;
	CSceneVehicleCarWheelState sync;
	CSceneVehicleCarWheelState field234;
	CSceneVehicleCarWheelState async_state;
} CSceneVehicleCarWheelSnapshot;

_Static_assert(offsetof(CSceneVehicleCarWheel, contact_relative_local_distance) ==
	offsetof(CSceneVehicleCarWheelSnapshot, contact_relative_local_distance),
	"wheel snapshot prefix");
_Static_assert(sizeof(CSceneVehicleCarWheelSnapshot) == 680, "v6 wheel size");
_Static_assert(offsetof(CSceneVehicleCarWheelSnapshot, previous_sync) +
	sizeof(CSceneVehicleCarWheelHistory) == sizeof(CSceneVehicleCarWheelSnapshot),
	"wheel snapshot history and padding");

TMNF_HD static inline void CSceneVehicleCarWheel_Capture(
	const CSceneVehicleCarWheel *wheel, CSceneVehicleCarWheelSnapshot *snapshot)
{
	memcpy(snapshot, wheel, offsetof(CSceneVehicleCarWheelSnapshot, previous_sync));
	snapshot->surface_handler = NULL;
	if (wheel->history != NULL)
		memcpy(&snapshot->previous_sync, wheel->history, sizeof(*wheel->history));
	else
		memset(&snapshot->previous_sync, 0, sizeof(CSceneVehicleCarWheelHistory));
}

TMNF_HD static inline void CSceneVehicleCarWheel_Restore(
	CSceneVehicleCarWheel *wheel, const CSceneVehicleCarWheelSnapshot *snapshot)
{
	void *surface = wheel->surface_handler;
	memcpy(wheel, snapshot, offsetof(CSceneVehicleCarWheelSnapshot, previous_sync));
	wheel->surface_handler = surface;
	if (wheel->history != NULL)
		memcpy(wheel->history, &snapshot->previous_sync, sizeof(*wheel->history));
}


/*
 * CSceneVehicleCarTuning is a 0x3AC-byte game object containing 32-bit curve
 * pointers and dynamic buffers. This native form preserves typed fields, not
 * game byte offsets. The corresponding game offsets are enumerated above.
 */
typedef struct {
	void *accel_curve;
	float damper_max;
	float damper_min;
	void *damper_modulation_curve;
	const float *gear_ratios;
	const float *gear_upshift;
	const float *gear_downshift;
	const float *gear_aux;
	uint32_t gear_ratio_count;
	int32_t engine_model;
	float forward_speed_limit_scale; /* game +0x02c */
	float engine_rpm_accel;
	float engine_rpm_decel;
	float engine_rpm_reverse_accel;
	float engine_rpm_high_decel;
	float engine_rpm_low_accel;
	float engine_rpm_follow_accel;
	float engine_rpm_turbo_decel;
	float speed_32c;
	float speed_330;
	float speed_334;
	float speed_338;
	int32_t suspension_model;
	float suspension_stiffness;
	float suspension_damping;
	float suspension_rest_length;
	float suspension_scale;
} CSceneVehicleCarTuning;

/*
 * CSceneVehicleCar is a 0x878-byte game object with 32-bit pointers. The replay
 * harness reconstructs this native object from the documented game offsets.
 */
typedef struct {
	void *hms_item;
	CHmsStateDyna *dyna_state;
	CHmsDynaParams *dyna_params;
	CSceneVehicleCarTuning *tuning;
	float input_gas;
	float input_brake;
	float input_steer;
	CSceneVehicleCarWheel *wheels;
	uint32_t wheel_count;
	CSceneVehicleCarEngine engine;
	GmVec3 current_local_speed;
	GmVec3 total_force_added;
	GmVec3 total_impulse_added;
	int32_t engine_mode;             /* game +0x2e4 */
	int32_t turbo_active;            /* game +0x628 */
	int32_t drive_mode;              /* game +0x69c */
	int32_t engine_limit_flag;       /* game +0x744 */
	int32_t gear_downshift_flag;     /* game +0x748 */
	int32_t force_wheel_speed;       /* game +0x6a0 */
	int32_t flag_60c;                /* game +0x60c */
	int32_t block_wheel_speed;       /* game +0x73c */
	float forced_wheel_speed;        /* tuning-selected source +0x2bc */
} CSceneVehicleCar;

/* 0x007BFD80 */ TMNF_HD void CSceneVehicle_VehicleInputSteerSet(
	CSceneVehicleCar *self, float value);
/* 0x007BFD90 */ TMNF_HD void CSceneVehicle_VehicleInputGasSet(
	CSceneVehicleCar *self, float value);
/* 0x007BFDA0 */ TMNF_HD void CSceneVehicle_VehicleInputBrakeSet(
	CSceneVehicleCar *self, float value);

/* 0x004FE500 */
TMNF_HD void CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
	const TMNFRaceInputs *inputs, CSceneVehicleCar *vehicle);

/* 0x007C1060 */
TMNF_HD void CSceneVehicleCarWheelRealTimeState_Integrate(
	CSceneVehicleCarWheelRealTimeState *self, float dt);

/* 0x007C0EC0 */
TMNF_HD void CSceneVehicleCar_WheelUpdateSpeedFromVehicleSpeed(
	CSceneVehicleCar *self, CSceneVehicleCarWheel *wheel,
	float vehicle_speed, float dt);

/* 0x008F46C0 */
TMNF_HD void GmSpringFloat_Integrate(GmSpringFloat *self, float dt);

/* 0x007BD090 */
TMNF_HD void SDynaMath_ComputeImpulse(
	float mass, const GmMat3 *inverse_inertia, float restitution,
	const GmVec3 *relative_speed, const GmVec3 *normal,
	const GmVec3 *lever_arm, GmVec3 *impulse);

/* 0x007BE310 */
TMNF_HD void CSceneVehicleCar_AddVehicleCentralForce(
	CSceneVehicleCar *self, const GmVec3 *force);

/* 0x007BE360 */
TMNF_HD void CSceneVehicleCar_AddVehicleTorque(
	CSceneVehicleCar *self, const GmVec3 *torque);

/* 0x007BE2C0 */
TMNF_HD void CSceneVehicleCar_AddVehicleForce(
	CSceneVehicleCar *self, const GmVec3 *force, const GmVec3 *point);

/* 0x007BD700, A01 engine model 5 */
TMNF_HD void CSceneVehicleCar_EngineIntegrate(
	CSceneVehicleCar *self, float throttle, float dt);

/* 0x007C1810 */
TMNF_HD void CSceneVehicleCar_WheelAddForceToVehicle(
	CSceneVehicleCar *self, const CSceneVehicleCarWheel *wheel);

#ifdef __cplusplus
}
#endif

#endif
