#ifndef TMNF_VEHICLE_MODEL6_H
#define TMNF_VEHICLE_MODEL6_H

#include "tmnf_hd.h"
#include <stdint.h>

#include "vehicle_aux.h"
#include "vehicle_contact.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Scalar Model 6 tuning view. The curve-backed members are supplied through
 * CSceneVehicleCarTuningCurveSet. This table maps every other tuning byte
 * offset read by 0x007C3E80 to its typed native field.
 *
 *  game    field
 *  +02c    forward_speed_limit_scale
 *  +030    reverse_speed_limit_scale
 *  +040    brake_base
 *  +044    brake_speed_scale
 *  +048    forward_brake_limit_sliding
 *  +04c    forward_brake_limit
 *  +060    speed_limit_force
 *  +064    vertical_force_scale
 *  +074    wheel_steer_sine_limit
 *  +07c    steer_slowdown_scale
 *  +098    wheel_torque_scale
 *  +09c    sliding_steer_torque_scale
 *  +0a4    lateral_force_scale
 *  +0b0    sliding_lateral_limit_scale
 *  +0b4    lateral_overflow_blend
 *  +0e4    wheel_overflow_blend
 *  +160    vertical_force_divisor
 *  +200    traction_loss_scale
 *  +228    burnout_trigger_scale
 *  +22c    burnout_trigger_limit
 *  +234    rollover_axis_min_length
 *  +238    rollover_torque_x_scale
 *  +23c    rollover_torque_z_scale
 *  +240    sliding_brake_scale
 *  +244    braking_lateral_limit_scale
 *  +248    reverse_brake_limit_sliding
 *  +24c    reverse_brake_limit
 *  +254    burnout_speed_max
 *  +258    burnout_speed_min
 *  +264    donut_lateral_force_scale
 *  +268    donut_yaw_angle_scale
 *  +26c    donut_steer_linear
 *  +270    donut_steer_quadratic
 *  +274    donut_countersteer_scale
 *  +278    donut_radius_exponent
 *  +27c    donut_radial_speed_exponent
 *  +280    donut_radius_min
 *  +284    donut_lateral_speed_limit
 *  +28c    donut_normal_angle_limit
 *  +290    donut_angle_positive_limit
 *  +294    donut_angle_negative_limit
 *  +298    burnout_enter_ticks
 *  +29c    burnout_enter_accel_scale
 *  +2a0    burnout_enter_lateral_scale
 *  +2a8    burnout_exit_ticks
 *  +2ac    burnout_exit_accel_scale
 *  +2b8    burnout_exit_extra_accel
 *  +33c    material6_longitudinal_scale
 *  +340    material6_gas_denominator
 *  +344    material6_vertical_shape
 *  +348    material6_vertical_scale
 */
typedef struct {
	float forward_speed_limit_scale;
	float reverse_speed_limit_scale;
	float brake_base;
	float brake_speed_scale;
	float forward_brake_limit_sliding;
	float forward_brake_limit;
	float speed_limit_force;
	float vertical_force_scale;
	float wheel_steer_sine_limit;
	float steer_slowdown_scale;
	float wheel_torque_scale;
	float sliding_steer_torque_scale;
	float lateral_force_scale;
	float sliding_lateral_limit_scale;
	float lateral_overflow_blend;
	float wheel_overflow_blend;
	float vertical_force_divisor;
	float traction_loss_scale;
	float burnout_trigger_scale;
	float burnout_trigger_limit;
	float rollover_axis_min_length;
	float rollover_torque_x_scale;
	float rollover_torque_z_scale;
	float sliding_brake_scale;
	float braking_lateral_limit_scale;
	float reverse_brake_limit_sliding;
	float reverse_brake_limit;
	float burnout_speed_max;
	float burnout_speed_min;
	float donut_lateral_force_scale;
	float donut_yaw_angle_scale;
	float donut_steer_linear;
	float donut_steer_quadratic;
	float donut_countersteer_scale;
	float donut_radius_exponent;
	float donut_radial_speed_exponent;
	float donut_radius_min;
	float donut_lateral_speed_limit;
	float donut_normal_angle_limit;
	float donut_angle_positive_limit;
	float donut_angle_negative_limit;
	uint32_t burnout_enter_ticks;
	float burnout_enter_accel_scale;
	float burnout_enter_lateral_scale;
	uint32_t burnout_exit_ticks;
	float burnout_exit_accel_scale;
	float burnout_exit_extra_accel;
	float material6_longitudinal_scale;
	float material6_gas_denominator;
	float material6_vertical_shape;
	float material6_vertical_scale;
} CSceneVehicleCarModel6Tuning;

/*
 * Native storage for Model 6 car fields absent from CSceneVehicleCar.
 *
 *  game       field
 *  +1dc..1e4 pivot_position
 *  +1e8..1f0 pivot_axis
 *  +5c4      reverse_mode
 *  +5cc      reverse_speed_threshold
 *  +5d8      contact_block_count
 *  +5dc      side_contact
 *  +62c      last_sliding_tick
 *  +630      sliding_start_tick
 *  +634      sliding_elapsed_ticks
 *  +6a4..6d0 model_iso
 *  +6d4..6dc rollover_axis
 *  +6e0..6e8 orbit_center
 *  +6ec      orbit_initial_radius
 *  +6f0      orbit_radius
 *  +6f4      burnout_start_tick
 *  +6f8      burnout_transition_tick
 *  +6fc..704 orbit_axis
 *  +708      orbit_sign
 *  +840      axle_width
 */
typedef struct {
	GmVec3 pivot_position;
	GmVec3 pivot_axis;
	int32_t reverse_mode;
	float reverse_speed_threshold;
	int32_t contact_block_count;
	int32_t side_contact;
	uint32_t last_sliding_tick;
	uint32_t sliding_start_tick;
	uint32_t sliding_elapsed_ticks;
	GmIso4 model_iso;
	GmVec3 rollover_axis;
	GmVec3 orbit_center;
	float orbit_initial_radius;
	float orbit_radius;
	uint32_t burnout_start_tick;
	uint32_t burnout_transition_tick;
	GmVec3 orbit_axis;
	float orbit_sign;
	float axle_width;
} CSceneVehicleCarModel6State;

typedef struct TMNFVehicleModel6Context {
	CSceneVehicleCar *vehicle;
	CSceneVehicleCarAuxContext *aux;
	TMNFVehicleContactContext *contact;
	const CSceneVehicleCarTuningCurveSet *curves;
	const CSceneVehicleCarModel6Tuning *tuning;
	CSceneVehicleCarModel6State *state;

	/* Source of the virtual location copied to car+0x6A4 at function entry. */
	const GmIso4 *model_iso_source;
	/* *(this->hms_item->field14)+0x50 in the original object graph. */
	const GmVec3 *body_reference_position;
} CSceneVehicleCarModel6Context;

/*
 * 0x007C3E80
 *
 * Stack/local mapping used by the mechanical translation:
 *  fStack_100/fc/f8 -> wheel_axis or orbit_tangent
 *  fStack_ec/e8/e4  -> temporary force/vector
 *  fStack_c0/bc/b8  -> temporary torque/delta
 *  fStack_a4/a0/9c  -> central force/torque
 *  fStack_128       -> requested lateral or longitudinal force
 *  fStack_120/118   -> requested/limited traction totals or brake product
 *  fStack_110       -> traction fraction
 *  fStack_114       -> final longitudinal acceleration
 *  fStack_38        -> model_iso.m[4]
 *  fStack_f4        -> any-sliding flag
 */
TMNF_HD void CSceneVehicleCar_ComputeForcesModel6(
	CSceneVehicleCarModel6Context *context, float model_value,
	const GmVec3 *existing_force, float lateral_force_factor,
	float longitudinal_force_factor, const GmVec3 *local_speed,
	const GmVec3 *local_angular_speed, float steering_angle,
	int grounded, const CSceneVehicleMaterialBlendableVals *material,
	int *sliding, float *brake_force);

#ifdef __cplusplus
}
#endif

#endif
