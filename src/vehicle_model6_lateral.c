#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tmnf_fp.h"
#include "vehicle_model6_lateral.h"

#define MODEL6_STEER_RESPONSE_MIN_SPEED 0x1.666666p-1f
#define MODEL6_SIDE_SUM_EPSILON 0x1.4f8b58p-17f
#define TMNF_PI_DOUBLE 0x1.921fb54442d18p+1

TMNF_HD static float model6_steer_response(
	float speed_magnitude, const VehicleModel6TuningScalars *scalars)
{
	float absolute_speed = fabsf(speed_magnitude);
	float response;

	/* 0x007C5C68: unordered values take the response branch. */
	if (!(F(absolute_speed) < F(MODEL6_STEER_RESPONSE_MIN_SPEED))) {
		/* 0x007C5C88: unordered values take the sine branch. */
		if (!(F(scalars->s074) < F(speed_magnitude))) {
			float ratio =
				x87_div(speed_magnitude, scalars->s074);
			float angle = x87_r24(F(ratio) * TMNF_PI_DOUBLE);

			angle = x87_r24(F(angle) * 0.5);
			response = x87_sin(angle);
		} else {
			response = 1.0f;
		}
	} else {
		response = 0.0f;
	}
	return response;
}

TMNF_HD static float model6_clamp_unit_preserve_nan(float value)
{
	if (F(value) <= 0.0)
		return 0.0f;
	if (1.0 <= F(value))
		return 1.0f;
	return value;
}

TMNF_HD static void model6_add_lateral_torque(
	CSceneVehicleCar *car, float force_x, float wheel_offset)
{
	GmVec3 torque;
	float force_y = x87_mul(force_x, 0.0f);
	float force_z = force_y;
	float force_z_at_zero_y = x87_mul(force_z, 0.0f);
	float offset_force_y = x87_mul(wheel_offset, force_y);
	float offset_force_x;
	float force_y_at_zero_z;
	float force_x_at_zero_z;

	torque.x = x87_sub(force_z_at_zero_y, offset_force_y);
	offset_force_x = x87_mul(wheel_offset, force_x);
	torque.y = x87_sub(offset_force_x, force_z_at_zero_y);
	force_y_at_zero_z = x87_mul(force_y, 0.0f);
	force_x_at_zero_z = x87_mul(0.0f, force_x);
	torque.z = x87_sub(force_y_at_zero_z, force_x_at_zero_z);
	CSceneVehicleCar_AddVehicleTorque(car, &torque);
}

/*
 * CSceneVehicleCar::ComputeForcesModel6 region 4,
 * decompiled lines 660-789, [0x007C5BB3, 0x007C5FB8).
 */
TMNF_HD void VehicleModel6_ApplyLateralFriction(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	VehicleModel6Scratch *scratch)
{
	CSceneVehicleCar *car = context->car;
	VehicleModel6PersistentState *state = context->state;
	const VehicleModel6TuningScalars *scalars = context->scalars;
	const GmVec3 *local_speed = inputs->local_speed;
	const GmVec3 *local_angular_speed = inputs->local_angular_speed;
	uint32_t wheel_index;

	scratch->side_force_requested_sum = 0.0f;
	scratch->side_force_limit_sum = 0.0f;
	for (wheel_index = 0; wheel_index < car->wheel_count; ++wheel_index) {
		CSceneVehicleCarWheel *wheel = &car->wheels[wheel_index];
		GmVec3 wheel_velocity;
		float wheel_offset = wheel->steerable == 0
			? -state->axle_half_span_source
			: state->axle_half_span_source;
		float angular_lateral;
		float steer_response;
		float maximum;
		float requested_dot;
		float requested_scale;
		float requested;
		float requested_absolute;
		float lateral_limit;
		float force;

		wheel_offset = x87_r24(F(wheel_offset) * 0.5);
		angular_lateral =
			x87_mul(local_angular_speed->y, wheel_offset);
		wheel_velocity.x =
			x87_add(angular_lateral, local_speed->x);
		wheel_velocity.y = x87_add(local_speed->y, 0.0f);
		wheel_velocity.z = x87_add(0.0f, local_speed->z);
		steer_response =
			model6_steer_response(
				scratch->local_speed_magnitude, scalars);

		maximum =
			CSceneVehicleCarTuning_GetMaxSideFrictionFromSpeed(
				context->curves, local_speed->z);
		lateral_limit =
			x87_mul(maximum, inputs->material->lateral_grip);

		requested_dot = x87_mul(wheel_velocity.y, 0.0f);
		requested_dot =
			x87_add(requested_dot, wheel_velocity.x);
		requested_dot = x87_add(
			requested_dot,
			x87_mul(0.0f, wheel_velocity.z));
		requested_scale = -scalars->s0a4;
		requested_scale = x87_r24(F(requested_scale) * 0.5);
		requested = x87_mul(requested_dot, requested_scale);
		requested_absolute = fabsf(requested);
		force = requested;

		if (F(lateral_limit) < F(requested_absolute)) {
			uint32_t requested_bits;
			float one_minus_blend =
				x87_sub(1.0f, scalars->s0e4);
			float limit_part =
				x87_mul(lateral_limit, one_minus_blend);
			float requested_part =
				x87_mul(requested_absolute, scalars->s0e4);
			float blended_absolute =
				x87_add(requested_part, limit_part);
			float sign;

			scratch->side_force_limit_sum = x87_add(
				lateral_limit,
				scratch->side_force_limit_sum);
			scratch->side_force_requested_sum = x87_add(
				requested_absolute,
				scratch->side_force_requested_sum);
			memcpy(
				&requested_bits, &requested,
				sizeof(requested_bits));
			sign =
				(requested_bits & UINT32_C(0x80000000)) != 0
					? -1.0f
					: 1.0f;
			scratch->slip_activity = 1;
			force = x87_mul(sign, blended_absolute);
		}

		force = x87_mul(scalars->s098, force);
		if (wheel->steerable != 0) {
			float direction =
				state->reverse_latch == 0 ? 1.0f : -1.0f;
			float sliding_scale =
				wheel->real_time.is_sliding == 0
					? 1.0f
					: scalars->s09c;
			float steer_torque =
				CSceneVehicleCarTuning_GetSteerDriveTorqueFromSpeed(
					context->curves, local_speed->z);
			float steer_term =
				x87_mul(direction, steer_response);

			steer_term =
				x87_mul(steer_term, state->smoothed_steer);
			steer_term = x87_mul(steer_torque, steer_term);
			steer_term = x87_mul(steer_term, sliding_scale);
			force = x87_sub(force, steer_term);
		}
		model6_add_lateral_torque(car, force, wheel_offset);
	}

	if (scratch->slip_activity != 0) {
		state->slip_tick = scratch->tick;
		if (scratch->sliding_at_entry == 0)
			state->slip_begin_tick = scratch->tick;
		state->slip_duration =
			scratch->tick - state->slip_begin_tick;
	}

	scratch->traction_blend = 1.0f;
	if (scratch->tick == state->slip_tick
		&& F(MODEL6_SIDE_SUM_EPSILON)
			< F(scratch->side_force_limit_sum)) {
		float excess = x87_sub(
			scratch->side_force_requested_sum,
			scratch->side_force_limit_sum);
		float normalized = x87_div(excess, scratch->side_force_limit_sum);
		float ratio =
			x87_div(normalized, scalars->s200);
		float clamped = model6_clamp_unit_preserve_nan(ratio);

		scratch->traction_blend = x87_sub(1.0f, clamped);
	}
}
