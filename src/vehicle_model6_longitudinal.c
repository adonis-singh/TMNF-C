#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tmnf_fp.h"
#include "vehicle_model6_longitudinal.h"

#define PI_DOUBLE 0x1.921fb54442d18p+1
#define UINT32_RANGE_FLOAT 0x1p32f
#define BURNOUT_START_GAS 0.1
#define BURNOUT_START_UP 0x1.8p-1f

TMNF_HD static float u32_to_pc24(uint32_t value)
{
	int32_t signed_value = (int32_t)value;
	float result = (float)signed_value;

	if (signed_value < 0)
		result = x87_add(result, UINT32_RANGE_FLOAT);
	return result;
}

TMNF_HD static float sign_from_bits(float value)
{
	uint32_t bits;

	memcpy(&bits, &value, sizeof(bits));
	return (bits & UINT32_C(0x80000000)) != 0 ? -1.0f : 1.0f;
}

TMNF_HD static float burnout_sine(uint32_t elapsed, uint32_t duration)
{
	float elapsed_float = u32_to_pc24(elapsed);
	float duration_float = u32_to_pc24(duration);
	float phase = x87_r24(F(elapsed_float) * PI_DOUBLE);

	phase = x87_div(phase, duration_float);
	return x87_sin(phase);
}

TMNF_HD static float wheel_sliding_scale(
	const CSceneVehicleCar *car, uint32_t wheel_count, float scale)
{
	float result = 1.0f;
	uint32_t index;

	for (index = 0; index < wheel_count; ++index) {
		if (car->wheels[index].real_time.is_sliding != 0)
			result = x87_mul(scale, result);
	}
	return result;
}

TMNF_HD static void mark_all_wheels_sliding(
	CSceneVehicleCar *car, uint32_t wheel_count,
	VehicleModel6Scratch *scratch)
{
	uint32_t index;

	scratch->slip_activity = 1;
	for (index = 0; index < wheel_count; ++index)
		car->wheels[index].real_time.is_sliding = 1;
}

TMNF_HD static float compute_drive_acceleration(
	const VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	const VehicleModel6Scratch *scratch)
{
	const VehicleModel6PersistentState *state = context->state;
	const VehicleModel6TuningScalars *scalars = context->scalars;
	float slipping_acceleration;
	float curve_acceleration;
	float acceleration;
	float steer_slowdown;
	float steer_penalty;
	float burnout_multiplier = 1.0f;
	float burnout_offset = 0.0f;
	float reverse_brake_sign;
	float turbo_factor;
	float drive_direction;
	float throttle;

	slipping_acceleration =
		CSceneVehicleCarTuning_M5GetSlippingAccelFromSpeed(
			context->curves, inputs->local_speed->z);
	if (state->reverse_latch == 0) {
		curve_acceleration =
			CSceneVehicleCarTuning_M5GetAccelFromSpeed(
				context->curves, inputs->local_speed->z);
	} else {
		curve_acceleration =
			CSceneVehicleCarTuning_M6GetRearGearAccelFromSpeed(
				context->curves, inputs->local_speed->z);
	}

	if (context->car->engine_mode == 1) {
		acceleration = 0.0f;
	} else {
		float slipping_part = x87_mul(
			x87_sub(1.0f, scratch->traction_blend),
			slipping_acceleration);
		float traction_part = x87_mul(
			scratch->traction_blend, curve_acceleration);

		acceleration = x87_add(traction_part, slipping_part);
	}

	steer_slowdown = CSceneVehicleCarTuning_M5GetSteerSlowDownFromSpeed(
		context->curves, inputs->local_speed->z);
	steer_penalty = x87_mul(
		scalars->s07c, (float)fabs(F(state->smoothed_steer)));
	steer_penalty = x87_mul(steer_penalty, steer_slowdown);

	if (state->burnout_state == 1) {
		float sine = burnout_sine(
			scratch->tick - state->burnout_start_tick,
			scalars->ticks298);

		burnout_multiplier = x87_add(
			x87_mul(x87_sub(scalars->s29c, 1.0f), sine),
			1.0f);
	}
	if (state->burnout_state == 3) {
		uint32_t elapsed = scratch->tick - state->burnout_end_tick;
		float sine = burnout_sine(elapsed, scalars->ticks2a8);
		float cycle = u32_to_pc24(elapsed / scalars->ticks2a8);

		burnout_multiplier = x87_add(
			x87_mul(x87_sub(scalars->s2ac, 1.0f), sine),
			1.0f);
		cycle = x87_sub(cycle, 1.0f);
		cycle = x87_mul(cycle, cycle);
		burnout_offset = x87_mul(cycle, scalars->s2b8);
	}

	reverse_brake_sign = state->reverse_latch == 0 ? 0.0f : -1.0f;
	turbo_factor =
		state->turbo_type == TMNF_TURBO_NONE
		? 0.0f : state->turbo_factor;
	drive_direction = state->reverse_latch == 0 ? 1.0f : -1.0f;

	throttle = x87_mul(
		reverse_brake_sign, inputs->material->braking);
	throttle = x87_mul(throttle, context->car->input_brake);
	throttle = x87_add(
		x87_mul(context->car->input_gas, inputs->material->braking),
		throttle);
	acceleration = x87_mul(throttle, acceleration);
	acceleration = x87_add(
		acceleration, x87_mul(curve_acceleration, turbo_factor));
	acceleration = x87_mul(burnout_multiplier, acceleration);
	acceleration = x87_sub(
		acceleration, x87_mul(steer_penalty, drive_direction));
	acceleration = x87_add(burnout_offset, acceleration);

	if (scratch->water_forces_applied != 0)
		acceleration = x87_r24(F(acceleration) * 0.5);
	if (state->special_physics != 0)
		acceleration = x87_mul(curve_acceleration, turbo_factor);

	return acceleration;
}

TMNF_HD static float compute_braking(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	VehicleModel6Scratch *scratch, float drive_acceleration)
{
	CSceneVehicleCar *car = context->car;
	VehicleModel6PersistentState *state = context->state;
	const VehicleModel6TuningScalars *scalars = context->scalars;
	const float speed = inputs->local_speed->z;
	float braking = 0.0f;
	float wheel_scale;
	float maximum;

	if (0.0 < F(speed)) {
		wheel_scale = wheel_sliding_scale(
			car, scratch->wheel_count, scalars->s240);
		braking = x87_add(
			x87_mul(scalars->s044, speed), scalars->s040);
		braking = x87_mul(braking, car->input_brake);
		braking = x87_mul(braking, wheel_scale);
		maximum = x87_mul(
			inputs->material->steering,
			*inputs->sliding_out == 0
			? scalars->s04c : scalars->s048);
		if (F(maximum) < F(braking)) {
			braking = maximum;
			mark_all_wheels_sliding(
				car, scratch->wheel_count, scratch);
		}
	}

	if (F(speed) < 0.0 && BURNOUT_START_GAS < F(car->input_gas)) {
		if (state->special_physics == 0) {
			float trigger = x87_mul(
				x87_mul(-drive_acceleration, scalars->s228),
				speed);

			if (F(scalars->s22c) < F(trigger)
				&& F(BURNOUT_START_UP)
					< F(scratch->car_iso.m[4])) {
				state->burnout_start_tick = scratch->tick;
				state->burnout_state = 1;
				state->force_wheel_speed = 1;
			}
		}

		wheel_scale = wheel_sliding_scale(
			car, scratch->wheel_count, scalars->s240);
		braking = x87_sub(
			scalars->s040, x87_mul(scalars->s044, speed));
		braking = x87_mul(braking, car->input_gas);
		braking = x87_mul(braking, wheel_scale);
		maximum = x87_mul(
			inputs->material->steering,
			*inputs->sliding_out == 0
			? scalars->s24c : scalars->s248);
		if (F(maximum) < F(braking)) {
			braking = maximum;
			mark_all_wheels_sliding(
				car, scratch->wheel_count, scratch);
		}
	}

	return braking;
}

TMNF_HD void VehicleModel6_ApplyLongitudinalForces(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	VehicleModel6Scratch *scratch)
{
	const VehicleModel6LongitudinalScalars *longitudinal_scalars =
		(const VehicleModel6LongitudinalScalars *)
			(const void *)context->scalars;
	const VehicleModel6TuningScalars *scalars =
		&longitudinal_scalars->common;
	float drive_acceleration;
	float braking;
	float braking_direction;
	float longitudinal_force;
	float forward_speed_limit;
	float reverse_speed_limit;
	float vertical_force;
	GmVec3 force;

	drive_acceleration = compute_drive_acceleration(
		context, inputs, scratch);
	braking = compute_braking(
		context, inputs, scratch, drive_acceleration);
	*inputs->braking_out = braking;

	braking_direction = sign_from_bits(inputs->local_speed->z);
	longitudinal_force = x87_sub(
		drive_acceleration,
		x87_mul(braking_direction, braking));

	reverse_speed_limit = x87_mul(
		scalars->s030, inputs->material->acceleration);
	forward_speed_limit = x87_mul(
		scalars->s02c, inputs->material->acceleration);
	if (F(forward_speed_limit) < F(inputs->local_speed->z)) {
		if (!(F(longitudinal_force) < 0.0))
			longitudinal_force = -scalars->s060;
		else
			longitudinal_force = x87_sub(
				longitudinal_force, scalars->s060);
	}
	if (F(inputs->local_speed->z) < F(-reverse_speed_limit)) {
		if (!(0.0 < F(longitudinal_force)))
			longitudinal_force = scalars->s060;
		else
			longitudinal_force = x87_add(
				scalars->s060, longitudinal_force);
	}

	force.x = 0.0f;
	force.y = 0.0f;
	force.z = x87_mul(
		longitudinal_force, inputs->longitudinal_scale);
	CSceneVehicleCar_AddVehicleCentralForce(context->car, &force);

	vertical_force = x87_mul(
		-scalars->s064, inputs->force_before_model->z);
	vertical_force = x87_div(vertical_force, longitudinal_scalars->s160);
	force.x = 0.0f;
	force.y = 0.0f;
	force.z = vertical_force;
	CSceneVehicleCar_AddVehicleCentralForce(context->car, &force);
}
