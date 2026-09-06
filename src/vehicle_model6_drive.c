#include <math.h>
#include <stdint.h>

#include "tmnf_fp.h"
#include "vehicle_model6_drive.h"

#define DRIVE_INPUT_THRESHOLD 0.1
#define DRIVE_AXIS_MIN_Y 0x1.8p-1f
#define DRIVE_STEER_EPSILON 0x1.4f8b58p-17f
#define DRIVE_LENGTH_SQUARED_EPSILON 0x1.b7cdfcp-34f
#define DRIVE_REVERSE_LATERAL_LIMIT 2.0f

TMNF_HD static float drive_length_squared_yx_z(const GmVec3 *value)
{
	float y_squared = x87_mul(value->y, value->y);
	float x_squared = x87_mul(value->x, value->x);
	float z_squared = x87_mul(value->z, value->z);

	return x87_add(x87_add(y_squared, x_squared), z_squared);
}

TMNF_HD static float drive_cross_length_squared(const GmVec3 *value)
{
	float x_squared = x87_mul(value->x, value->x);
	float y_squared = x87_mul(value->y, value->y);
	float z_squared = x87_mul(value->z, value->z);

	return x87_add(z_squared, x87_add(x_squared, y_squared));
}

TMNF_HD static float drive_sign_of_negated(float value)
{
	union {
		float f;
		uint32_t u;
	} bits = { .f = value };

	return (bits.u & UINT32_C(0x80000000)) == 0 ? -1.0f : 1.0f;
}

TMNF_HD static void drive_set_up_axis(GmVec3 *axis)
{
	axis->x = 0.0f;
	axis->y = 1.0f;
	axis->z = 0.0f;
}

TMNF_HD static void drive_normalize_burnout_axis(
	VehicleModel6PersistentState *state)
{
	float length_squared = drive_length_squared_yx_z(&state->burnout_axis);

	if (F(DRIVE_LENGTH_SQUARED_EPSILON) < F(length_squared)) {
		float length = x87_sqrt(length_squared);
		float inverse_length = x87_rcp(length);

		state->burnout_axis.x =
			x87_mul(inverse_length, state->burnout_axis.x);
		state->burnout_axis.y =
			x87_mul(state->burnout_axis.y, inverse_length);
		state->burnout_axis.z =
			x87_mul(inverse_length, state->burnout_axis.z);
		return;
	}

	state->burnout_state = 0;
	drive_set_up_axis(&state->burnout_axis);
}

TMNF_HD static GmVec3 drive_compute_burnout_cross(
	const GmVec3 *axis, const GmVec3 *local_speed, float steer_angle)
{
	GmVec3 cross;
	float steer_sign = GmFunc_Sign(steer_angle);
	float length_squared;

	cross.x = x87_sub(
		x87_mul(axis->y, local_speed->z),
		x87_mul(axis->z, local_speed->y));
	cross.y = x87_sub(
		x87_mul(axis->z, local_speed->x),
		x87_mul(axis->x, local_speed->z));
	cross.z = x87_sub(
		x87_mul(axis->x, local_speed->y),
		x87_mul(axis->y, local_speed->x));
	cross.x = x87_mul(steer_sign, cross.x);
	cross.y = x87_mul(steer_sign, cross.y);
	cross.z = x87_mul(steer_sign, cross.z);

	length_squared = drive_cross_length_squared(&cross);
	if (F(DRIVE_LENGTH_SQUARED_EPSILON) < F(length_squared)) {
		float length = x87_sqrt(length_squared);
		float inverse_length = x87_rcp(length);

		cross.x = x87_mul(inverse_length, cross.x);
		cross.y = x87_mul(cross.y, inverse_length);
		cross.z = x87_mul(inverse_length, cross.z);
	}
	return cross;
}

TMNF_HD static void drive_initialize_burnout(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	VehicleModel6Scratch *scratch)
{
	VehicleModel6PersistentState *state = context->state;
	const GmVec3 *local_speed = inputs->local_speed;
	const GmVec3 *body_center = scratch->body_center_local;
	GmVec3 radius_offset;
	GmVec3 burnout_cross;
	GmVec3 world_axis;
	GmVec3 world_center;
	GmVec3 reference = { 0.0f, 0.0f, 1.0f };
	float reference_scale;
	float reference_zero;
	float signed_angle;
	float target_radius;
	uint32_t wheel_index;

	state->burnout_state = 2;
	state->burnout_steer_sign = GmFunc_Sign(inputs->steer_angle);
	state->burnout_axis.x = 0.0f;
	state->burnout_axis.y = 0.0f;
	state->burnout_axis.z = 0.0f;
	for (wheel_index = 0; wheel_index < scratch->wheel_count; wheel_index++) {
		const CSceneVehicleCarWheel *wheel =
			&context->car->wheels[wheel_index];

		if (wheel->real_time.has_ground_contact == 0)
			continue;
		state->burnout_axis.x = x87_add(
			wheel->real_time.field90.x, state->burnout_axis.x);
		state->burnout_axis.y = x87_add(
			wheel->real_time.field90.y, state->burnout_axis.y);
		state->burnout_axis.z = x87_add(
			wheel->real_time.field90.z, state->burnout_axis.z);
	}
	drive_normalize_burnout_axis(state);

	radius_offset.x = x87_sub(state->reference_position.x, body_center->x);
	radius_offset.y = x87_sub(state->reference_position.y, body_center->y);
	radius_offset.z = x87_sub(state->reference_position.z, body_center->z);
	reference_scale = x87_add(
		x87_add(
			x87_mul(state->reference_axis.y, 0.0f),
			x87_mul(state->reference_axis.x, 0.0f)),
		state->reference_axis.z);
	reference_zero = x87_mul(0.0f, reference_scale);
	radius_offset.x = x87_add(reference_zero, radius_offset.x);
	radius_offset.y = x87_add(reference_zero, radius_offset.y);
	radius_offset.z = x87_add(reference_scale, radius_offset.z);
	state->burnout_initial_radius =
		x87_sqrt(drive_length_squared_yx_z(&radius_offset));

	burnout_cross = drive_compute_burnout_cross(
		&state->burnout_axis, local_speed, inputs->steer_angle);
	GmVec3_Mult_Mat3(
		&state->burnout_axis, (const GmMat3 *)&state->car_iso);
	if (F(state->burnout_axis.y) < F(DRIVE_AXIS_MIN_Y)) {
		state->burnout_state = 0;
		drive_set_up_axis(&state->burnout_axis);
	}

	signed_angle = x87_mul(
		GmVec3_GetAngle(&reference, &burnout_cross),
		GmFunc_Sign(inputs->steer_angle));
	if (F(context->scalars->s290) < F(signed_angle) ||
	    F(signed_angle) < -F(context->scalars->s294)) {
		state->burnout_state = 0;
	} else {
		GmVec3_SetMult_Mat3(
			&world_axis, &burnout_cross,
			(const GmMat3 *)&state->car_iso);
		GmVec3_SetMult_Iso4(
			&world_center, body_center, &state->car_iso);
		target_radius = x87_add(
			CSceneVehicleCarTuning_M6GetBurnoutRadiusFromSpeed(
				context->curves, local_speed->z),
			state->burnout_initial_radius);
		state->burnout_target_radius = target_radius;
		state->burnout_center.x = x87_add(
			world_center.x, x87_mul(target_radius, world_axis.x));
		state->burnout_center.y = x87_add(
			x87_mul(target_radius, world_axis.y), world_center.y);
		state->burnout_center.z = x87_add(
			world_center.z, x87_mul(target_radius, world_axis.z));
	}
	state->force_wheel_speed = state->burnout_state == 2;
}

TMNF_HD static void drive_update_burnout(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	VehicleModel6Scratch *scratch)
{
	VehicleModel6PersistentState *state = context->state;
	const GmVec3 *local_speed = inputs->local_speed;

	if (state->special_physics != 0)
		return;

	if (F(DRIVE_INPUT_THRESHOLD) < F(context->car->input_brake) &&
	    F(context->car->input_gas) < F(DRIVE_INPUT_THRESHOLD) &&
	    state->burnout_state == 1) {
		state->burnout_end_tick = scratch->tick;
		state->burnout_state = 3;
	}

	if (F(DRIVE_INPUT_THRESHOLD) < F(context->car->input_gas) &&
	    F(DRIVE_INPUT_THRESHOLD) < F(context->car->input_brake) &&
	    F(local_speed->z) < F(context->scalars->s258) &&
	    F(DRIVE_AXIS_MIN_Y) < F(inputs->slope_adherence)) {
		state->burnout_state = 1;
		state->force_wheel_speed = 1;
		state->burnout_start_tick = scratch->tick;
	}

	if (F(DRIVE_INPUT_THRESHOLD) < F(context->car->input_gas) &&
	    F(DRIVE_INPUT_THRESHOLD) < F(context->car->input_brake) &&
	    F(local_speed->z) < F(context->scalars->s254) &&
	    F(context->scalars->s258) < F(local_speed->z) &&
	    !(F(fabsf(inputs->steer_angle)) < F(DRIVE_STEER_EPSILON))) {
		drive_initialize_burnout(context, inputs, scratch);
	}
}

TMNF_HD static void drive_update_reverse_latch(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs)
{
	VehicleModel6PersistentState *state = context->state;
	const GmVec3 *local_speed = inputs->local_speed;

	if (state->burnout_state != 0) {
		state->reverse_latch = 0;
		return;
	}

	if (F(DRIVE_INPUT_THRESHOLD) < F(context->car->input_brake) &&
	    F(local_speed->z) < F(state->reverse_speed_threshold) &&
	    F(fabsf(local_speed->x)) < F(DRIVE_REVERSE_LATERAL_LIMIT)) {
		state->reverse_latch = 1;
	}
	if (F(DRIVE_INPUT_THRESHOLD) < F(context->car->input_gas) &&
	    (0.0 < F(local_speed->z) ||
	     F(DRIVE_REVERSE_LATERAL_LIMIT) < F(fabsf(local_speed->x)))) {
		state->reverse_latch = 0;
	}
	if (F(context->car->input_gas) < F(DRIVE_INPUT_THRESHOLD) &&
	    F(context->car->input_brake) < F(DRIVE_INPUT_THRESHOLD)) {
		if ((!isnan(F(local_speed->z)) && 0.0 <= F(local_speed->z)) ||
		    F(fabsf(local_speed->z)) <
			    F(DRIVE_REVERSE_LATERAL_LIMIT)) {
			state->reverse_latch = 0;
		} else {
			state->reverse_latch = 1;
		}
	}
	if (0.0 < F(local_speed->z) && state->turbo_type != 0)
		state->reverse_latch = 0;
}

TMNF_HD static void drive_add_rollover_torque(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs)
{
	const GmVec3 *local_speed = inputs->local_speed;
	float lateral_squared = x87_mul(local_speed->x, local_speed->x);
	float denominator = x87_add(fabsf(local_speed->z), 1.0f);
	float ratio = x87_div(lateral_squared, denominator);
	float torque_z = x87_mul(
		CSceneVehicleCarTuning_M6GetRolloverLateralFromSpeedRatio(
			context->curves, ratio),
		drive_sign_of_negated(local_speed->x));
	GmVec3 torque = { 0.0f, 0.0f, torque_z };

	CSceneVehicleCar_AddVehicleTorque(context->car, &torque);
}

TMNF_HD VehicleModel6Flow VehicleModel6_UpdateDriveState(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	VehicleModel6Scratch *scratch)
{
	VehicleModel6PersistentState *state = context->state;

	if (inputs->valid_ground_material == 0) {
		scratch->valid_ground_path = 0;
		if (state->burnout_state == 1) {
			state->burnout_end_tick = scratch->tick;
			state->burnout_state = 3;
		}
		return VEHICLE_M6_SKIP_TO_ACTIVITY_COMMIT;
	}

	scratch->valid_ground_path = 1;
	scratch->local_speed_magnitude =
		x87_sqrt(drive_length_squared_yx_z(inputs->local_speed));
	drive_update_burnout(context, inputs, scratch);
	drive_update_reverse_latch(context, inputs);
	drive_add_rollover_torque(context, inputs);
	return VEHICLE_M6_CONTINUE;
}
