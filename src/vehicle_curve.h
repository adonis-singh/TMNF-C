#ifndef TMNF_VEHICLE_CURVE_H
#define TMNF_VEHICLE_CURVE_H

#include "tmnf_hd.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Native immutable curve views. Positions and values each contain `count`
 * float32 elements.
 */
typedef struct {
	uint32_t count;
	const float *positions;
	/* positions[i] -/+ the key epsilon under PC=24 (0x005914C0 recomputes
	 * them per visited key); filled by CFuncKeys_Compile. */
	const float *lower_bounds;
	const float *upper_bounds;
} CFuncKeys;

/* Precomputes lower_bounds/upper_bounds from positions (heap allocated). */
void CFuncKeys_Compile(CFuncKeys *keys);
/* Fills the 2 * count floats lower_bounds already points at (world.c links
 * every curve's bounds into one table before compiling). */
void CFuncKeys_CompileInto(CFuncKeys *keys);
void CFuncKeys_Release(CFuncKeys *keys);

typedef struct {
	CFuncKeys keys;
	const float *values;
	int32_t interpolation;
} CFuncKeysReal;

/*
 * Native view of the curve-backed CSceneVehicleCarTuning fields used by the
 * vehicle force models. This is intentionally independent of the 32-bit game
 * object's byte layout below.
 */
typedef struct {
	const CFuncKeysReal *accel_from_speed;
	const CFuncKeysReal *rollover_lateral_from_speed;
	const CFuncKeysReal *max_side_friction_from_speed;
	const CFuncKeysReal *lateral_contact_slowdown_from_speed;
	const CFuncKeysReal *steer_slowdown_from_speed;
	const CFuncKeysReal *rollover_lateral_coef_from_angle;
	const CFuncKeysReal *steer_drive_torque_from_speed;
	const CFuncKeysReal *m4_steer_radius_from_speed;
	const CFuncKeysReal *m4_max_friction_force_from_speed;
	const CFuncKeysReal *m5_slipping_accel_from_speed;
	float m5_slipping_accel_scale;
	const CFuncKeysReal *water_friction_from_speed;
	float damper_max;
	float damper_min;
	const CFuncKeysReal *m6_damper_modulation;
	const CFuncKeysReal *m6_rear_gear_accel_from_speed;
	const CFuncKeysReal *m6_rollover_lateral_from_speed_ratio;
	const CFuncKeysReal *m6_burnout_radius_from_speed;
	const CFuncKeysReal *m6_lateral_speed_from_burnout_radius;
	const CFuncKeysReal *m6_donut_rollover_from_speed;
	const CFuncKeysReal *m6_burnout_rollover_from_speed;
} CSceneVehicleCarTuningCurveSet;

/* Exact 32-bit layouts recovered from the exported decompilation. */
typedef struct {
	uint32_t count;
	uint32_t data;
} TMNFFastBufferGame32;

typedef struct {
	uint8_t reserved00[0x14];
	TMNFFastBufferGame32 positions;
	uint32_t reserved1c;
} CFuncKeysGame32;

typedef struct {
	CFuncKeysGame32 base;
	TMNFFastBufferGame32 values;
	int32_t interpolation;
} CFuncKeysRealGame32;

typedef struct {
	uint8_t reserved000[0x034];
	uint32_t accel_from_speed;
	uint8_t reserved038[0x030];
	uint32_t lateral_contact_slowdown_from_speed;
	uint8_t reserved06c[0x00c];
	uint32_t steer_slowdown_from_speed;
	uint8_t reserved07c[0x024];
	uint32_t steer_drive_torque_from_speed;
	uint8_t reserved0a4[0x008];
	uint32_t max_side_friction_from_speed;
	uint8_t reserved0b0[0x008];
	uint32_t rollover_lateral_from_speed;
	uint32_t rollover_lateral_coef_from_angle;
	uint8_t reserved0c0[0x05c];
	float damper_max;
	float damper_min;
	uint8_t reserved124[0x090];
	uint32_t m4_steer_radius_from_speed;
	uint8_t reserved1b8[0x004];
	uint32_t m4_max_friction_force_from_speed;
	uint8_t reserved1c0[0x020];
	uint32_t m5_slipping_accel_from_speed;
	float m5_slipping_accel_scale;
	uint8_t reserved1e8[0x030];
	uint32_t water_friction_from_speed;
	uint8_t reserved21c[0x008];
	uint32_t m6_damper_modulation;
	uint8_t reserved228[0x008];
	uint32_t m6_rear_gear_accel_from_speed;
	uint8_t reserved234[0x01c];
	uint32_t m6_rollover_lateral_from_speed_ratio;
	uint8_t reserved254[0x008];
	uint32_t m6_burnout_radius_from_speed;
	uint32_t m6_lateral_speed_from_burnout_radius;
	uint8_t reserved264[0x024];
	uint32_t m6_donut_rollover_from_speed;
	uint8_t reserved28c[0x018];
	uint32_t m6_burnout_rollover_from_speed;
	uint8_t reserved2a8[0x104];
} CSceneVehicleCarTuningGame32;

_Static_assert(sizeof(TMNFFastBufferGame32) == 0x08, "game fast buffer size");
_Static_assert(offsetof(TMNFFastBufferGame32, count) == 0x00,
	"game fast buffer count");
_Static_assert(offsetof(TMNFFastBufferGame32, data) == 0x04,
	"game fast buffer data");

_Static_assert(sizeof(CFuncKeysGame32) == 0x20, "CFuncKeys game size");
_Static_assert(offsetof(CFuncKeysGame32, positions) == 0x14,
	"CFuncKeys positions");

_Static_assert(sizeof(CFuncKeysRealGame32) == 0x2c,
	"CFuncKeysReal game size");
_Static_assert(offsetof(CFuncKeysRealGame32, values) == 0x20,
	"CFuncKeysReal values");
_Static_assert(offsetof(CFuncKeysRealGame32, interpolation) == 0x28,
	"CFuncKeysReal interpolation");

_Static_assert(sizeof(CSceneVehicleCarTuningGame32) == 0x3ac,
	"CSceneVehicleCarTuning game size");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32, accel_from_speed) == 0x034,
	"tuning accel curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	lateral_contact_slowdown_from_speed) == 0x068,
	"tuning lateral slowdown curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	steer_slowdown_from_speed) == 0x078, "tuning steer slowdown curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	steer_drive_torque_from_speed) == 0x0a0,
	"tuning steer torque curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	max_side_friction_from_speed) == 0x0ac,
	"tuning side friction curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	rollover_lateral_from_speed) == 0x0b8,
	"tuning rollover speed curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	rollover_lateral_coef_from_angle) == 0x0bc,
	"tuning rollover angle curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32, damper_max) == 0x11c,
	"tuning damper max");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32, damper_min) == 0x120,
	"tuning damper min");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	m4_steer_radius_from_speed) == 0x1b4,
	"tuning model 4 steer curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	m4_max_friction_force_from_speed) == 0x1bc,
	"tuning model 4 friction curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	m5_slipping_accel_from_speed) == 0x1e0,
	"tuning model 5 slipping curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	m5_slipping_accel_scale) == 0x1e4,
	"tuning model 5 slipping scale");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	water_friction_from_speed) == 0x218,
	"tuning water friction curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	m6_damper_modulation) == 0x224,
	"tuning damper modulation curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	m6_rear_gear_accel_from_speed) == 0x230,
	"tuning rear gear curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	m6_rollover_lateral_from_speed_ratio) == 0x250,
	"tuning model 6 rollover ratio curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	m6_burnout_radius_from_speed) == 0x25c,
	"tuning burnout radius curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	m6_lateral_speed_from_burnout_radius) == 0x260,
	"tuning burnout lateral speed curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	m6_donut_rollover_from_speed) == 0x288,
	"tuning donut rollover curve");
_Static_assert(offsetof(CSceneVehicleCarTuningGame32,
	m6_burnout_rollover_from_speed) == 0x2a4,
	"tuning burnout rollover curve");

/* 0x005914C0 */
TMNF_HD void CFuncKeys_GetBoundingIndices(
	const CFuncKeys *self, float position, uint32_t *lower_index,
	uint32_t *upper_index, int forward);

/* 0x00591670 */
TMNF_HD int CFuncKeys_ComputeBlendCoef(
	const CFuncKeys *self, float position, uint32_t *lower_index,
	uint32_t *upper_index, float *blend, int forward);

/* 0x00585E70 */
TMNF_HD void CFuncKeysReal_GetRealAt(
	const CFuncKeysReal *self, float position, float *value,
	uint32_t *lower_index, uint32_t *upper_index, float *blend,
	int32_t interpolation, int forward);

/* 0x00586200 */
TMNF_HD void CFuncKeysReal_GetValueOut(
	const CFuncKeysReal *self, float position, float *value,
	uint32_t *lower_index);

/* 0x00586240 */
TMNF_HD float CFuncKeysReal_GetValue(
	const CFuncKeysReal *self, float position, uint32_t *lower_index);

/* 0x007F3BE0 */
TMNF_HD float CSceneVehicleCarTuning_GetAccelFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3C30 */
TMNF_HD float CSceneVehicleCarTuning_GetRolloverLateralFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3C70 */
TMNF_HD float CSceneVehicleCarTuning_GetMaxSideFrictionFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3CB0 */
TMNF_HD float CSceneVehicleCarTuning_GetLateralContactSlowDownFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3D00 */
TMNF_HD float CSceneVehicleCarTuning_GetSteerSlowDownFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3D50 */
TMNF_HD float CSceneVehicleCarTuning_GetRolloverLateralCoefFromAngle(
	const CSceneVehicleCarTuningCurveSet *self, float angle);
/* 0x007F3D80 */
TMNF_HD float CSceneVehicleCarTuning_GetSteerDriveTorqueFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3DC0 */
TMNF_HD float CSceneVehicleCarTuning_M4GetSteerRadiusFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3E00 */
TMNF_HD float CSceneVehicleCarTuning_M4GetMaxFrictionForceFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3E40 */
TMNF_HD float CSceneVehicleCarTuning_M5GetAccelFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3E80 */
TMNF_HD float CSceneVehicleCarTuning_M5GetSlippingAccelFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3ED0 */
TMNF_HD float CSceneVehicleCarTuning_M5GetSteerSlowDownFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3F10 */
TMNF_HD float CSceneVehicleCarTuning_M5GetLateralContactSlowDownFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3F40 */
TMNF_HD float CSceneVehicleCarTuning_GetWaterFrictionFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F3F80 */
TMNF_HD float CSceneVehicleCarTuning_M6GetModulationFromDamperAbsorbVal(
	const CSceneVehicleCarTuningCurveSet *self, float damper_absorb);
/* 0x007F3FF0 */
TMNF_HD float CSceneVehicleCarTuning_M6GetRearGearAccelFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F4030 */
TMNF_HD float CSceneVehicleCarTuning_M6GetBurnoutRadiusFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F4070 */
TMNF_HD float CSceneVehicleCarTuning_M6GetLateralSpeedFromBurnoutRadius(
	const CSceneVehicleCarTuningCurveSet *self, float radius);
/* 0x007F40B0 */
TMNF_HD float CSceneVehicleCarTuning_M6GetBurnoutRolloverFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F40F0 */
TMNF_HD float CSceneVehicleCarTuning_M6GetDonutRolloverFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed);
/* 0x007F4130 */
TMNF_HD float CSceneVehicleCarTuning_M6GetRolloverLateralFromSpeedRatio(
	const CSceneVehicleCarTuningCurveSet *self, float speed_ratio);

#ifdef __cplusplus
}
#endif

#endif
