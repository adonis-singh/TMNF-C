/* Shared state boundary for independently ported regions of the 3,003-
 * instruction CSceneVehicleCar::ComputeForcesModel6 (0x007C3E80).
 *
 * Fields are grouped by lifetime: persistent state mirrors car offsets that
 * survive calls, inputs mirror the original ABI, and scratch carries only
 * values crossing the seven recovered control-flow regions.
 */
#ifndef TMNF_VEHICLE_MODEL6_COMMON_H
#define TMNF_VEHICLE_MODEL6_COMMON_H

#include "tmnf_hd.h"
#include <stdint.h>

#include "vehicle.h"
#include "vehicle_aux.h"
#include "vehicle_contact.h"
#include "vehicle_curve.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	int32_t reverse_latch;          /* car +0x5C4 */
	float reverse_speed_threshold; /* car +0x5CC */
	int32_t contact_counter;        /* car +0x5D8 */
	int32_t lateral_contact;        /* car +0x5DC */
	int32_t water_forces_applied;   /* car +0x5E4 */
	float smoothed_steer;           /* car +0x5E8 */
	float turbo_factor;             /* car +0x5F4 */
	int32_t turbo_type;             /* car +0x600 */
	int32_t special_physics;        /* car +0x60C */
	int32_t sliding;                /* car +0x628 */
	uint32_t slip_tick;             /* car +0x62C */
	uint32_t slip_begin_tick;       /* car +0x630 */
	uint32_t slip_duration;         /* car +0x634 */
	int32_t burnout_state;          /* car +0x69C */
	int32_t force_wheel_speed;      /* car +0x6A0 */
	GmIso4 car_iso;                 /* car +0x6A4 */
	GmVec3 normalized_force;        /* car +0x6D4 */
	GmVec3 burnout_center;          /* car +0x6E0 */
	float burnout_initial_radius;   /* car +0x6EC */
	float burnout_target_radius;    /* car +0x6F0 */
	uint32_t burnout_start_tick;    /* car +0x6F4 */
	uint32_t burnout_end_tick;      /* car +0x6F8 */
	GmVec3 burnout_axis;            /* car +0x6FC */
	float burnout_steer_sign;       /* car +0x708 */
	GmVec3 reference_position;      /* car +0x1DC */
	GmVec3 reference_axis;          /* car +0x1E8 */
	float axle_half_span_source;    /* car +0x840 */
} VehicleModel6PersistentState;

/* Scalar tuning fields read directly by Model 6. Names retain source offsets
 * until their engine semantics are established by traces. Curves use the
 * typed CSceneVehicleCarTuningCurveSet instead. */
typedef struct {
	float s02c, s030;
	float s040, s044, s048, s04c;
	float s060, s064;
	float s074, s07c;
	float s098, s09c, s0a4;
	float s0b0, s0b4;
	float s0e4;
	float s200;
	float s228, s22c;
	float s234, s238, s23c, s240, s244, s248, s24c;
	float s254, s258;
	float s264, s268, s26c, s270, s274, s278, s27c, s280;
	float s284, s28c, s290, s294, s298, s29c, s2a0;
	uint32_t ticks298;
	uint32_t ticks2a8;
	float s2a8, s2ac, s2b8;
	float s33c, s340, s344, s348;
} VehicleModel6TuningScalars;

typedef struct {
	float dt;
	const GmVec3 *force_before_model;
	float slope_adherence;
	float longitudinal_scale;
	const GmVec3 *local_speed;
	const GmVec3 *local_angular_speed;
	float steer_angle;
	int32_t valid_ground_material;
	const CSceneVehicleMaterialBlendableVals *material;
	int32_t *sliding_out;
	float *braking_out;
} VehicleModel6Inputs;

typedef struct {
	GmIso4 car_iso;
	const GmVec3 *body_center_local;
	uint32_t wheel_count;
	uint32_t tick;
	int32_t water_forces_applied;
	int32_t all_wheels_material6;
	int32_t sliding_at_entry;
	int32_t slip_activity;
	int32_t valid_ground_path;
	float local_speed_magnitude;
	float side_force_requested_sum;
	float side_force_limit_sum;
	float traction_blend;
} VehicleModel6Scratch;

typedef struct {
	CSceneVehicleCar *car;
	CSceneVehicleCarAuxContext *aux;
	TMNFVehicleContactContext *contact;
	const CSceneVehicleCarTuningCurveSet *curves;
	VehicleModel6PersistentState *state;
	const VehicleModel6TuningScalars *scalars;
	const GmIso4 *resolved_car_iso;
	const GmVec3 *body_center_local;
	uint32_t tick;
	int32_t track_has_water;
} VehicleModel6Context;

typedef enum {
	VEHICLE_M6_CONTINUE = 0,
	VEHICLE_M6_SKIP_TO_ACTIVITY_COMMIT = 1,
	VEHICLE_M6_SKIP_TO_FINALIZE = 2,
} VehicleModel6Flow;

#ifdef __cplusplus
}
#endif

#endif /* TMNF_VEHICLE_MODEL6_COMMON_H */
