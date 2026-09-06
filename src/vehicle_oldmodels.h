/*
 * CSceneVehicleCar force models 3, 4 and 5 (sceneengine:
 * SceneVehicleCar_OldModels.obj). These are the non-Stadium car physics that
 * TrackMania United Forever selects through CSceneVehicleCarTuning+0x354
 * (reflection enum Steer01..Steer06, strings at 0x007A33C3..0x007A33EC):
 *
 *   +0x354 value  enum     0x007C69E0 dispatch          port
 *   0             Steer01  ComputeForcesModel3 (0x007FA770), wheel forces only
 *   1             Steer02  ComputeForcesModel3 (0x007FA770), full model
 *   2             Steer03  ComputeForcesModel3 (0x007FA770), wheel forces only
 *   3             Steer04  ComputeForcesModel4 (0x007FB5F0)
 *   4             Steer05  ComputeForcesModel5 (0x007FC170)
 *   5             Steer06  ComputeForcesModel6 (0x007C3E80), vehicle_model6.c
 *
 * PORTED, NO ORACLE TRACE YET. Every function in this unit is a transcription
 * of the disassembly with per-operation PC=24 rounding (tools/x87trace.py
 * recipes, docs/PORTING.md). No TMUF capture has been replayed against it.
 * Constants: 0x00B313B8 double 0.5, 0x00B2C178 double 0.0, 0x00B36110 float
 * pi as double, 0x00B362C0 float 0.1 as double, 0x00BA38FC float 1e-5,
 * 0x00B2C060 float -1.0, 0x00D0AC60 float 1e-10 (static .data initialiser,
 * same value Model 6 replays with).
 */
#ifndef TMNF_VEHICLE_OLDMODELS_H
#define TMNF_VEHICLE_OLDMODELS_H

#include "tmnf_hd.h"
#include <stdint.h>

#include "vehicle_aux.h"
#include "vehicle_contact.h"
#include "vehicle_curve.h"
#include "vehicle_model6.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Scalar CSceneVehicleCarTuning fields read by 0x007FA770, 0x007FB5F0 and
 * 0x007FC170. Offsets shared with Model 6 keep the vehicle_model6.h names.
 *
 *  game    field                          readers
 *  +02c    forward_speed_limit_scale      3 4 5
 *  +030    reverse_speed_limit_scale      3 4 5
 *  +040    brake_base                     3 4 5
 *  +044    brake_speed_scale              3 4 5
 *  +048    brake_limit_sliding            3 4 5
 *  +04c    brake_limit                    3 4 5
 *  +060    speed_limit_force              3 4 5
 *  +064    vertical_force_scale           3 5
 *  +074    wheel_steer_sine_limit         3 5
 *  +07c    steer_slowdown_scale           3 5
 *  +080    m5_steer_gate_needs_sliding    5   (int)
 *  +084    m5_steer_gate_sliding_ticks    5   (uint)
 *  +098    wheel_torque_scale             3 5
 *  +09c    sliding_steer_torque_scale     3 5
 *  +0a4    lateral_force_scale            3 5
 *  +0b0    sliding_lateral_limit_scale    3 5
 *  +0b4    lateral_overflow_blend         3 5
 *  +0c0    longitudinal_torque_scale      3 4 5
 *  +0e4    wheel_overflow_blend           3 5
 *  +160    vertical_force_divisor         3 5
 *  +19c    m4_steer_torque_speed_scale    4
 *  +1a0    m4_yaw_damping_linear          4
 *  +1a4    m4_yaw_damping_quadratic       4
 *  +1b0    m4_drift_exit_lateral_speed    4
 *  +1b8    m4_drift_steer_radius_scale    4
 *  +1cc    m4_drift_angle_rate            4
 *  +1d0    m4_drift_angle_radius_scale    4
 *  +1d8    m4_drift_angle_limit           4
 *  +1dc    m4_drift_entry_angle_scale     4
 *  +1f4    m5_longitudinal_torque_limit   5
 *  +1fc    m5_steer_gate_ticks            5   (uint)
 *  +200    traction_loss_scale            5
 */
typedef struct {
	float forward_speed_limit_scale;
	float reverse_speed_limit_scale;
	float brake_base;
	float brake_speed_scale;
	float brake_limit_sliding;
	float brake_limit;
	float speed_limit_force;
	float vertical_force_scale;
	float wheel_steer_sine_limit;
	float steer_slowdown_scale;
	int32_t m5_steer_gate_needs_sliding;
	uint32_t m5_steer_gate_sliding_ticks;
	float wheel_torque_scale;
	float sliding_steer_torque_scale;
	float lateral_force_scale;
	float sliding_lateral_limit_scale;
	float lateral_overflow_blend;
	float longitudinal_torque_scale;
	float wheel_overflow_blend;
	float vertical_force_divisor;
	float m4_steer_torque_speed_scale;
	float m4_yaw_damping_linear;
	float m4_yaw_damping_quadratic;
	float m4_drift_exit_lateral_speed;
	float m4_drift_steer_radius_scale;
	float m4_drift_angle_rate;
	float m4_drift_angle_radius_scale;
	float m4_drift_angle_limit;
	float m4_drift_entry_angle_scale;
	float m5_longitudinal_torque_limit;
	uint32_t m5_steer_gate_ticks;
	float traction_loss_scale;
} CSceneVehicleCarOldModelsTuning;

/*
 * Car fields the old models own. Fields shared with Model 6 (car +0x5c4
 * reverse_mode, +0x5cc reverse_speed_threshold, +0x62c/+0x630/+0x634 sliding
 * ticks, +0x840 axle_width) live in CSceneVehicleCarModel6State.
 *
 *  game   field
 *  +638   m4_drift_angle
 *  +63c   m4_drift_steer_sign
 *  +640   m4_drift_state (0 none, 1 entering, 2 holding)
 *  +648   m5_last_steer_tick
 *  +64c   m5_last_steer_sliding
 */
typedef struct {
	float m4_drift_angle;
	float m4_drift_steer_sign;
	int32_t m4_drift_state;
	uint32_t m5_last_steer_tick;
	int32_t m5_last_steer_sliding;
} CSceneVehicleCarOldModelsState;

typedef struct TMNFVehicleOldModelsContext {
	CSceneVehicleCar *vehicle;
	CSceneVehicleCarAuxContext *aux;
	TMNFVehicleContactContext *contact;
	const CSceneVehicleCarTuningCurveSet *curves;
	const CSceneVehicleCarOldModelsTuning *tuning;
	CSceneVehicleCarModel6State *shared;
	CSceneVehicleCarOldModelsState *state;
} CSceneVehicleCarOldModelsContext;

/* Shared regions. Each caller's disassembly has the same x87 schedule for
 * the region it uses; the VA ranges are listed at the definitions. */

/* Wheel lateral force and rollover torque for one wheel (Model 3
 * 0x007FA7F2..0x007FAC19, Model 5 0x007FC1C2..0x007FC5D2). */
TMNF_HD void VehicleOldModels_WheelLateral(
	CSceneVehicleCarOldModelsContext *context, CSceneVehicleCarWheel *wheel,
	float lateral_force_factor, const GmVec3 *local_speed,
	float steering_angle, int *sliding);

/* Reverse latch on car +0x5c4 (Model 3 0x007FAC96, Model 4 0x007FB694,
 * Model 5 0x007FC63E). */
TMNF_HD void VehicleOldModels_UpdateReverseLatch(
	CSceneVehicleCarOldModelsContext *context, float speed_length);

/* Longitudinal acceleration from the accel curve value, material, inputs
 * and turbo (Model 3 0x007FB169, Model 4 0x007FBC0D, Model 5 0x007FCBCE
 * with its own traction blend). */
TMNF_HD float VehicleOldModels_InputAcceleration(
	CSceneVehicleCarOldModelsContext *context,
	const CSceneVehicleMaterialBlendableVals *material,
	float accel_curve_value, float accel_blend, float steer_slowdown);

/* Brake force from the forward speed (Model 3 0x007FB20E..0x007FB44A,
 * Model 4 0x007FBCA6..0x007FBE3E, Model 5 0x007FCC60..0x007FCE6B).
 * `mark_wheels` sets every wheel sliding when the limit clips (Models 3 and
 * 5); `any_sliding` is raised on clip when non-NULL (Models 4 and 5). */
TMNF_HD float VehicleOldModels_BrakeForce(
	CSceneVehicleCarOldModelsContext *context,
	const CSceneVehicleMaterialBlendableVals *material,
	float forward_speed, int sliding, int mark_wheels, int *any_sliding);

/* Speed-limit override of the net longitudinal value (Model 3 0x007FB45C,
 * Model 4 0x007FC006, Model 5 0x007FCE7D). */
TMNF_HD float VehicleOldModels_LimitLongitudinal(
	CSceneVehicleCarOldModelsContext *context,
	const CSceneVehicleMaterialBlendableVals *material,
	float forward_speed, float net);

/* Central force from the model's vertical term (Model 3 0x007FB57A,
 * Model 5 0x007FCFEE). */
TMNF_HD void VehicleOldModels_AddVerticalForce(
	CSceneVehicleCarOldModelsContext *context, const GmVec3 *existing_force);

/* 0x007FA770 */
TMNF_HD void CSceneVehicleCar_ComputeForcesModel3(
	CSceneVehicleCarOldModelsContext *context, float dt,
	const GmVec3 *existing_force, float lateral_force_factor,
	float longitudinal_force_factor, const GmVec3 *local_speed,
	const GmVec3 *local_angular_speed, float steering_angle,
	int grounded, const CSceneVehicleMaterialBlendableVals *material,
	int *sliding, float *brake_force);

/* 0x007FB5F0 */
TMNF_HD void CSceneVehicleCar_ComputeForcesModel4(
	CSceneVehicleCarOldModelsContext *context, float dt,
	const GmVec3 *existing_force, float lateral_force_factor,
	float longitudinal_force_factor, const GmVec3 *local_speed,
	const GmVec3 *local_angular_speed, float steering_angle,
	int grounded, const CSceneVehicleMaterialBlendableVals *material,
	int *sliding, float *brake_force);

/* 0x007FC170 */
TMNF_HD void CSceneVehicleCar_ComputeForcesModel5(
	CSceneVehicleCarOldModelsContext *context, float dt,
	const GmVec3 *existing_force, float lateral_force_factor,
	float longitudinal_force_factor, const GmVec3 *local_speed,
	const GmVec3 *local_angular_speed, float steering_angle,
	int grounded, const CSceneVehicleMaterialBlendableVals *material,
	int *sliding, float *brake_force);

#ifdef __cplusplus
}
#endif

#endif
