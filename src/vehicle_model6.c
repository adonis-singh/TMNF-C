#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tmnf_fp.h"
#include "vehicle_compute.h"
#include "vehicle_model6.h"

#define MODEL6_LENGTH_EPSILON 0x1.b7cdfcp-34f
#define MODEL6_CONTROL_EPSILON 0x1.4f8b58p-17f
/* 0x00B362C0 / 0x00B362B8: float 0.1 promoted to double (+/-), the only
 * 0.1 constants referenced by 0x007C3E80..0x007C6A00. Not double 0.1. */
#define MODEL6_POINT_ONE 0x1.99999ap-4
#define MODEL6_POINT_SEVEN 0x1.666666p-1f
#define MODEL6_POINT_SEVEN_FIVE 0x1.8p-1f
/* 0x00B36110: float pi promoted to double, not mathematical double pi. */
#define MODEL6_PI 0x1.921fb6p+1
/* 0x00B3D2A8: float 3.6 promoted to double. */
#define MODEL6_METERS_PER_SECOND_TO_KMH 0x1.ccccccp+1
#define MODEL6_UINT32_RANGE 0x1p32f

TMNF_HD static float x87_mul_double(float value, double multiplier)
{
	return x87_r24(F(value) * multiplier);
}

TMNF_HD static float x87_add_double(float value, double addend)
{
	return x87_r24(F(value) + addend);
}

TMNF_HD static float x87_sub_double(float value, double subtrahend)
{
	return x87_r24(F(value) - subtrahend);
}

TMNF_HD static float model6_abs(float value)
{
	return (float)fabs(F(value));
}

TMNF_HD static float model6_sign_bits(float value)
{
	uint32_t bits;

	memcpy(&bits, &value, sizeof(bits));
	return (bits & UINT32_C(0x80000000)) != 0 ? -1.0f : 1.0f;
}

TMNF_HD static float model6_u32_float(uint32_t value)
{
	int32_t signed_value;
	float result;

	memcpy(&signed_value, &value, sizeof(signed_value));
	result = (float)signed_value;
	if (signed_value < 0)
		result = x87_add(result, MODEL6_UINT32_RANGE);
	return result;
}

TMNF_HD static float model6_sin(float value)
{
	return x87_sin(value);
}

TMNF_HD static float model6_cos(float value)
{
	return x87_cos(value);
}

TMNF_HD static float model6_curve_value(
	const CFuncKeysReal *curve, float position)
{
	return CFuncKeysReal_GetValue(curve, position, NULL);
}

TMNF_HD static float model6_speed_position(float speed)
{
	return x87_mul_double(speed, MODEL6_METERS_PER_SECOND_TO_KMH);
}

TMNF_HD static float model6_curve_from_speed(
	const CFuncKeysReal *curve, float speed)
{
	return model6_curve_value(curve, model6_speed_position(speed));
}

TMNF_HD static float model6_max_side_friction(
	const CSceneVehicleCarTuningCurveSet *curves, float speed)
{
	return model6_curve_from_speed(
		curves->max_side_friction_from_speed, speed);
}

TMNF_HD static float model6_steer_drive_torque(
	const CSceneVehicleCarTuningCurveSet *curves, float speed)
{
	return model6_curve_from_speed(
		curves->steer_drive_torque_from_speed, speed);
}

TMNF_HD static float model6_slipping_accel(
	const CSceneVehicleCarTuningCurveSet *curves, float speed)
{
	return x87_mul(
		curves->m5_slipping_accel_scale,
		model6_curve_from_speed(
			curves->m5_slipping_accel_from_speed, speed));
}

TMNF_HD static float model6_accel(
	const CSceneVehicleCarTuningCurveSet *curves, float speed)
{
	return model6_curve_from_speed(curves->accel_from_speed, speed);
}

TMNF_HD static float model6_steer_slowdown(
	const CSceneVehicleCarTuningCurveSet *curves, float speed)
{
	return model6_curve_from_speed(curves->steer_slowdown_from_speed, speed);
}

TMNF_HD static float model6_rear_gear_accel(
	const CSceneVehicleCarTuningCurveSet *curves, float speed)
{
	return model6_curve_from_speed(
		curves->m6_rear_gear_accel_from_speed, speed);
}

TMNF_HD static float model6_burnout_radius(
	const CSceneVehicleCarTuningCurveSet *curves, float speed)
{
	return model6_curve_from_speed(
		curves->m6_burnout_radius_from_speed, speed);
}

TMNF_HD static float model6_burnout_rollover(
	const CSceneVehicleCarTuningCurveSet *curves, float speed)
{
	return model6_curve_from_speed(
		curves->m6_burnout_rollover_from_speed, speed);
}

TMNF_HD static float model6_donut_rollover(
	const CSceneVehicleCarTuningCurveSet *curves, float speed)
{
	return model6_curve_from_speed(
		curves->m6_donut_rollover_from_speed, speed);
}

TMNF_HD static float model6_rollover_ratio(
	const CSceneVehicleCarTuningCurveSet *curves, float speed_ratio)
{
	return model6_curve_from_speed(
		curves->m6_rollover_lateral_from_speed_ratio, speed_ratio);
}

TMNF_HD static float model6_lateral_speed_from_radius(
	const CSceneVehicleCarTuningCurveSet *curves, float radius)
{
	float value = model6_curve_value(
		curves->m6_lateral_speed_from_burnout_radius, radius);

	return x87_div(value, (float)MODEL6_METERS_PER_SECOND_TO_KMH);
}

TMNF_HD static float model6_exp(float value)
{
	return x87_exp(value);
}

TMNF_HD static float model6_length_squared(const GmVec3 *value)
{
	float xy = x87_add(
		x87_mul(value->x, value->x),
		x87_mul(value->y, value->y));

	return x87_add(xy, x87_mul(value->z, value->z));
}

TMNF_HD static float model6_length(const GmVec3 *value)
{
	return x87_sqrt(model6_length_squared(value));
}

TMNF_HD static int model6_normalize(GmVec3 *value)
{
	float length_squared = model6_length_squared(value);
	float inverse;

	if (!(F(MODEL6_LENGTH_EPSILON) < F(length_squared)))
		return 0;
	inverse = x87_div(1.0f, x87_sqrt(length_squared));
	value->x = x87_mul(inverse, value->x);
	value->y = x87_mul(value->y, inverse);
	value->z = x87_mul(inverse, value->z);
	return 1;
}

TMNF_HD static GmVec3 model6_sub_vec3(const GmVec3 *left, const GmVec3 *right)
{
	GmVec3 result;

	result.x = x87_sub(left->x, right->x);
	result.y = x87_sub(left->y, right->y);
	result.z = x87_sub(left->z, right->z);
	return result;
}

TMNF_HD static GmVec3 model6_cross(const GmVec3 *left, const GmVec3 *right)
{
	GmVec3 result;

	result.x = x87_sub(
		x87_mul(left->y, right->z),
		x87_mul(left->z, right->y));
	result.y = x87_sub(
		x87_mul(left->z, right->x),
		x87_mul(left->x, right->z));
	result.z = x87_sub(
		x87_mul(left->x, right->y),
		x87_mul(left->y, right->x));
	return result;
}

/* Schedule at 0x007C4B7B: x*x + y*y, then z*z. */
TMNF_HD static float model6_dot_xyz(const GmVec3 *left, const GmVec3 *right)
{
	float xy = x87_add(
		x87_mul(left->x, right->x),
		x87_mul(left->y, right->y));

	return x87_add(xy, x87_mul(left->z, right->z));
}

/* Schedule at 0x007C4378: y*y + x*x, then z*z. */
TMNF_HD static float model6_dot_yxz(const GmVec3 *left, const GmVec3 *right)
{
	float yx = x87_add(
		x87_mul(left->y, right->y),
		x87_mul(left->x, right->x));

	return x87_add(yx, x87_mul(left->z, right->z));
}

TMNF_HD static const TMNFVehicleGroundMaterial *model6_wheel_material(
	const CSceneVehicleCarModel6Context *context,
	const CSceneVehicleCarWheel *wheel)
{
	uint16_t material_id =
		(uint16_t)wheel->real_time.contact_material_id;
	uint32_t material_index =
		context->contact->ground_material_indices[material_id];

	return context->contact->ground_materials[material_index];
}

TMNF_HD static float model6_enter_wave(
	const CSceneVehicleCarModel6Context *context, uint32_t tick)
{
	const CSceneVehicleCarModel6Tuning *tuning = context->tuning;
	uint32_t elapsed = tick - context->state->burnout_start_tick;
	float angle = x87_mul_double(model6_u32_float(elapsed), MODEL6_PI);
	float denominator =
		model6_u32_float(tuning->burnout_enter_ticks * 2u);

	angle = x87_div(angle, denominator);
	return x87_add(
		x87_mul(
			x87_sub(tuning->burnout_enter_lateral_scale, 1.0f),
			model6_cos(angle)),
		1.0f);
}

TMNF_HD static float model6_accel_wave(
	const CSceneVehicleCarModel6Context *context, uint32_t tick)
{
	const CSceneVehicleCarModel6Tuning *tuning = context->tuning;
	uint32_t elapsed = tick - context->state->burnout_start_tick;
	float angle = x87_mul_double(model6_u32_float(elapsed), MODEL6_PI);

	angle = x87_div(
		angle, model6_u32_float(tuning->burnout_enter_ticks));
	return x87_add(
		x87_mul(
			x87_sub(tuning->burnout_enter_accel_scale, 1.0f),
			model6_sin(angle)),
		1.0f);
}

TMNF_HD static float model6_exit_wave(
	const CSceneVehicleCarModel6Context *context, uint32_t tick,
	float *extra_acceleration)
{
	const CSceneVehicleCarModel6Tuning *tuning = context->tuning;
	uint32_t elapsed = tick - context->state->burnout_transition_tick;
	float angle = x87_mul_double(model6_u32_float(elapsed), MODEL6_PI);
	uint32_t quotient = elapsed / tuning->burnout_exit_ticks;
	float phase_count = model6_u32_float(quotient);
	float phase_delta;

	angle = x87_div(
		angle, model6_u32_float(tuning->burnout_exit_ticks));
	phase_delta = x87_sub_double(phase_count, 1.0);
	*extra_acceleration = x87_mul(
		x87_mul(phase_delta, phase_delta),
		tuning->burnout_exit_extra_accel);
	return x87_add(
		x87_mul(
			x87_sub(tuning->burnout_exit_accel_scale, 1.0f),
			model6_sin(angle)),
		1.0f);
}

TMNF_HD static void model6_add_material6_forces(
	CSceneVehicleCarModel6Context *context, const GmVec3 *local_speed,
	const GmVec3 *side_force)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	const CSceneVehicleCarModel6Tuning *tuning = context->tuning;
	GmVec3 normalized_speed = *local_speed;
	GmVec3 front_force = { 0.0f, 0.0f, 0.0f };
	GmVec3 rear_force = { 0.0f, 0.0f, 0.0f };
	float length_squared = model6_length_squared(&normalized_speed);

	if (F(MODEL6_LENGTH_EPSILON) < F(length_squared)) {
		float inverse = x87_div(
			1.0f, x87_sqrt(length_squared));

		/* 0x007C50C7 stores only the x component. */
		normalized_speed.x = x87_mul(
			inverse, normalized_speed.x);
	}
	if (MODEL6_POINT_ONE < F(vehicle->input_brake)) {
		GmVec3 drag;

		drag.x = x87_mul_double(local_speed->x, -MODEL6_POINT_ONE);
		drag.y = x87_mul_double(local_speed->y, -MODEL6_POINT_ONE);
		drag.z = x87_mul_double(local_speed->z, -MODEL6_POINT_ONE);
		CSceneVehicleCar_AddVehicleCentralForce(vehicle, &drag);
	}
	if (vehicle->flag_60c == 0
		&& MODEL6_POINT_ONE < F(vehicle->input_gas)
		&& F(vehicle->input_brake) < MODEL6_POINT_ONE) {
		float abs_x = model6_abs(normalized_speed.x);
		float speed_shape = x87_add_double(
			x87_mul_double(abs_x, 20.0), 1.0);
		float absolute_forward =
			x87_add_double(model6_abs(local_speed->z), 1.0);
		float vertical_divisor =
			x87_mul(absolute_forward, absolute_forward);
		float gas_divisor = x87_add_double(
			x87_mul(
				tuning->material6_gas_denominator,
				vehicle->input_gas),
			1.0);
		float front_common = x87_mul_double(
			x87_mul(
				tuning->material6_vertical_scale,
				vehicle->input_gas),
			1.5);
		float rear_common = x87_mul(
			tuning->material6_vertical_scale,
			vehicle->input_gas);

		front_common = x87_mul(front_common, abs_x);
		front_common = x87_mul(front_common, speed_shape);
		front_common = x87_mul(
			front_common, tuning->material6_vertical_shape);
		rear_common = x87_mul(
			x87_mul(
				x87_mul(
					rear_common,
					abs_x),
				speed_shape),
			tuning->material6_vertical_shape);

		front_force.x = x87_div(
			x87_mul(
				tuning->material6_longitudinal_scale,
				normalized_speed.x),
			gas_divisor);
		front_force.z = x87_div(front_common, vertical_divisor);
		rear_force.x = x87_div(
			x87_mul(
				x87_mul(
					normalized_speed.x, -1.0f),
				tuning->material6_longitudinal_scale),
			gas_divisor);
		rear_force.z = x87_div(rear_common, vertical_divisor);
	}

	for (uint32_t index = 0; index < vehicle->wheel_count; ++index) {
		CSceneVehicleCarWheel *wheel = &vehicle->wheels[index];

		if (F(vehicle->input_brake) < MODEL6_POINT_ONE
			&& wheel->real_time.is_sliding != 0) {
			if (index == 0 || index == 1)
				CSceneVehicleCar_AddVehicleForce(
					vehicle, &front_force,
					&wheel->offset_from_vehicle);
			if (index == 2 || index == 3)
				CSceneVehicleCar_AddVehicleForce(
					vehicle, &rear_force,
					&wheel->offset_from_vehicle);
		}
	}
	CSceneVehicleCar_AddVehicleCentralForce(vehicle, side_force);
}

TMNF_HD static void model6_continue_donut(
	CSceneVehicleCarModel6Context *context, const GmVec3 *existing_force,
	const GmVec3 *local_speed, const GmVec3 *local_angular_speed,
	float steering_angle, int grounded, int water_contact,
	float upright_value)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	CSceneVehicleCarModel6State *state = context->state;
	const CSceneVehicleCarModel6Tuning *tuning = context->tuning;
	const CSceneVehicleCarTuningCurveSet *curves = context->curves;
	GmVec3 wheel_normal = { 0.0f, 0.0f, 0.0f };
	GmVec3 world_normal;
	GmVec3 local_radius;
	GmVec3 radial_direction;
	GmVec3 tangent;
	GmVec3 force;
	GmVec3 torque;
	GmIso4 inverse;
	float normal_angle;
	float radius;
	float lateral_speed;
	float signed_angle;
	float steer_torque;
	float countersteer = 0.0f;
	float radial_ratio;

	if (F(vehicle->input_gas) < MODEL6_POINT_ONE
		|| F(vehicle->input_brake) < MODEL6_POINT_ONE
		|| model6_abs(steering_angle) < F(MODEL6_CONTROL_EPSILON)) {
		vehicle->drive_mode = 0;
		goto transition;
	}
	if (state->orbit_sign != model6_sign_bits(steering_angle)
		|| state->side_contact != 0
		|| 0 < state->contact_block_count
		|| grounded == 0 || water_contact != 0
		|| vehicle->flag_60c != 0) {
		vehicle->drive_mode = 0;
		goto transition;
	}

	for (uint32_t index = 0; index < vehicle->wheel_count; ++index) {
		const GmVec3 *normal = &vehicle->wheels[index].real_time.field90;

		wheel_normal.x = x87_add(normal->x, wheel_normal.x);
		wheel_normal.y = x87_add(normal->y, wheel_normal.y);
		wheel_normal.z = x87_add(normal->z, wheel_normal.z);
	}
	if (!model6_normalize(&wheel_normal))
		wheel_normal = (GmVec3){ 0.0f, 1.0f, 0.0f };
	GmVec3_SetMult_Mat3(
		&world_normal, &wheel_normal, (const GmMat3 *)&state->model_iso);
	normal_angle = model6_abs(
		GmVec3_GetAngle(&world_normal, &state->orbit_axis));
	if (F(tuning->donut_normal_angle_limit) < F(normal_angle)) {
		vehicle->drive_mode = 0;
		goto transition;
	}

	GmIso4_SetInverse(&inverse, &state->model_iso);
	GmVec3_SetMult_Iso4(&local_radius, &state->orbit_center, &inverse);
	local_radius = model6_sub_vec3(
		&local_radius, context->body_reference_position);
	radius = model6_length(&local_radius);
	radial_direction = local_radius;
	(void)model6_normalize(&radial_direction);
	tangent = model6_cross(&wheel_normal, &radial_direction);
	lateral_speed = model6_dot_yxz(&tangent, local_speed);

	if (F(tuning->donut_lateral_speed_limit)
			< F(model6_abs(lateral_speed))) {
		vehicle->drive_mode = 0;
	}
	if (F(state->orbit_initial_radius) < F(radius)
		|| F(radius) < F(tuning->donut_radius_min)) {
		float radius_delta = x87_sub(radius, state->orbit_radius);
		float radial_speed = model6_dot_yxz(
			&radial_direction, local_speed);
		float exponent = x87_sub(
			x87_mul(radius_delta, tuning->donut_radius_exponent),
			x87_mul(
				radial_speed,
				tuning->donut_radial_speed_exponent));
		float magnitude = x87_mul(
			x87_div(x87_mul(lateral_speed, lateral_speed), radius),
			model6_exp(exponent));

		force.x = x87_mul(magnitude, radial_direction.x);
		force.y = x87_mul(magnitude, radial_direction.y);
		force.z = x87_mul(magnitude, radial_direction.z);
		CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);
	} else {
		vehicle->drive_mode = 0;
	}

	for (uint32_t index = 0; index < vehicle->wheel_count; ++index) {
		CSceneVehicleCar_WheelAddForceToVehicle(
			vehicle, &vehicle->wheels[index]);
		vehicle->wheels[index].real_time.is_sliding = 0;
	}

	{
		float direction = GmFunc_Sign(steering_angle);
		float target = model6_lateral_speed_from_radius(curves, radius);
		float difference = x87_sub(
			x87_mul(-direction, target), lateral_speed);
		float magnitude = x87_mul(
			tuning->donut_lateral_force_scale, difference);

		force.x = x87_mul(magnitude, tangent.x);
		force.y = x87_mul(magnitude, tangent.y);
		force.z = x87_mul(magnitude, tangent.z);
		CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);

		const GmVec3 forward_axis = { 0.0f, 0.0f, 1.0f };
		signed_angle = x87_div(
			GmVec3_GetAngle(&forward_axis, &radial_direction),
			x87_r24(MODEL6_PI));
		if (!GmFunc_IsANumber(signed_angle)) {
			vehicle->drive_mode = 0;
		} else {
			float angle_radians = x87_mul_double(
				x87_mul(direction, signed_angle), MODEL6_PI);

			if (F(angle_radians)
					< F(-tuning->donut_angle_negative_limit)
				|| F(tuning->donut_angle_positive_limit)
					< F(angle_radians)) {
				vehicle->drive_mode = 0;
			} else {
				if (F(signed_angle) <= 0.0) {
					if (F(local_angular_speed->y) <= 0.0) {
						float shifted =
							x87_add(signed_angle, 1.0f);
						float shifted_squared =
							x87_mul(shifted, shifted);

						steer_torque = x87_mul(
							x87_mul(
								-tuning->donut_steer_quadratic,
								local_angular_speed->y),
							shifted_squared);
					} else {
						steer_torque = x87_mul(
							-tuning->donut_steer_linear,
							local_angular_speed->y);
					}
				} else if (0.0 <= F(local_angular_speed->y)) {
					float shifted = x87_sub(signed_angle, 1.0f);
					float shifted_squared =
						x87_mul(shifted, shifted);

					steer_torque = x87_mul(
						x87_mul(
							-tuning->donut_steer_quadratic,
							local_angular_speed->y),
						shifted_squared);
				} else {
					steer_torque = x87_mul(
						-tuning->donut_steer_linear,
						local_angular_speed->y);
				}
				radial_ratio = x87_div(lateral_speed, radius);
				if ((F(signed_angle) <= 0.0
						&& 0.0 < F(radial_ratio))
					|| (0.0 < F(signed_angle)
						&& F(radial_ratio) < 0.0)) {
					countersteer = x87_mul(
						-tuning->donut_countersteer_scale,
						radial_ratio);
				}
				torque.x = x87_mul(
					x87_add(
						x87_add(
							x87_mul(
								tuning->donut_yaw_angle_scale,
								signed_angle),
							steer_torque),
						countersteer),
					0.0f);
				torque.y = x87_add(
					x87_add(
						x87_mul(
							tuning->donut_yaw_angle_scale,
							signed_angle),
						steer_torque),
					countersteer);
				torque.z = x87_mul(torque.y, 0.0f);
				CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
			}
		}
	}

	torque.z = x87_mul(
		-GmFunc_Sign(local_speed->x),
		model6_donut_rollover(curves, model6_abs(local_speed->x)));
	torque.x = x87_mul(torque.z, 0.0f);
	torque.y = torque.x;
	CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
	torque.x =
		model6_burnout_rollover(curves, local_speed->z);
	torque.y = 0.0f;
	torque.z = 0.0f;
	CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);

transition:
	if (vehicle->drive_mode != 2) {
		uint32_t tick =
			*CMwTimerAdapter_GetTickTime(context->contact->timer);

		vehicle->drive_mode = 1;
		state->burnout_start_tick = tick;
	}
	(void)existing_force;
	(void)upright_value;
}

/* 0x007C3E80 */
TMNF_HD void CSceneVehicleCar_ComputeForcesModel6(
	CSceneVehicleCarModel6Context *context, float model_value,
	const GmVec3 *existing_force, float lateral_force_factor,
	float longitudinal_force_factor, const GmVec3 *local_speed,
	const GmVec3 *local_angular_speed, float steering_angle,
	int grounded, const CSceneVehicleMaterialBlendableVals *material,
	int *sliding, float *brake_force)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	CSceneVehicleCarAuxContext *aux = context->aux;
	CSceneVehicleCarModel6State *state = context->state;
	const CSceneVehicleCarModel6Tuning *tuning = context->tuning;
	const CSceneVehicleCarTuningCurveSet *curves = context->curves;
	uint32_t tick;
	uint32_t wheel_count = vehicle->wheel_count;
	int water_contact;
	int all_material6 = 1;
	int any_sliding = 0;
	int was_sliding = vehicle->turbo_active;
	float upright_value;

	(void)model_value;
	state->model_iso = *context->model_iso_source;
	upright_value = state->model_iso.m[4];
	water_contact = CSceneVehicleCar_ApplyWaterForces(
		aux, existing_force);
	/* 0x007C3ED8 stores car +0x5e4, which is both the air-control lock and
	 * the airborne friction gate read at 0x007C3916 by ApplyFrictionForces. */
	aux->air_control_locked = water_contact;
	context->contact->airborne_friction_gate = water_contact;
	for (uint32_t index = 0; index < wheel_count; ++index) {
		const CSceneVehicleCarWheel *wheel = &vehicle->wheels[index];

		if (wheel->real_time.has_ground_contact == 0
			|| (uint16_t)wheel->real_time.contact_material_id != 6u)
			all_material6 = 0;
	}

	if (vehicle->drive_mode == 2) {
		model6_continue_donut(
			context, existing_force, local_speed,
			local_angular_speed, steering_angle, grounded,
			water_contact, upright_value);
		goto finish;
	}

	tick = *CMwTimerAdapter_GetTickTime(context->contact->timer);
	if (vehicle->drive_mode == 1) {
		uint32_t start = state->burnout_start_tick;

		if (tick < start
			|| tuning->burnout_enter_ticks <= tick - start) {
			state->burnout_transition_tick = tick;
			vehicle->drive_mode = 3;
		} else {
			any_sliding = 1;
		}
	}
	if (vehicle->drive_mode == 3) {
		uint32_t start = state->burnout_transition_tick;

		if (tick < start
			|| tuning->burnout_exit_ticks <= tick - start) {
			vehicle->drive_mode = 0;
			vehicle->force_wheel_speed = 0;
		} else {
			for (uint32_t index = 0; index < wheel_count; ++index)
				vehicle->wheels[index].real_time.is_sliding = 1;
		}
	}

	for (uint32_t index = 0; index < wheel_count; ++index) {
		CSceneVehicleCarWheel *wheel = &vehicle->wheels[index];
		const TMNFVehicleGroundMaterial *wheel_material;
		GmVec3 relative_contact;
		GmVec3 wheel_axis;
		GmVec3 force;
		GmVec3 torque;
		float velocity_on_axis;
		float axis_length_squared;
		float maximum_force;
		float requested_force;
		float damper_modulation;
		float sliding_scale;
		float braking_scale;
		float enter_scale = 1.0f;

		CSceneVehicleCar_WheelAddForceToVehicle(vehicle, wheel);
		wheel_material = model6_wheel_material(context, wheel);
		if (wheel->real_time.has_ground_contact == 0
			|| !(0.0 < F(tuning->lateral_force_scale)))
			continue;

		relative_contact = model6_sub_vec3(
			&wheel->real_time.field54,
			context->body_reference_position);
		wheel_axis.x = x87_sub(
			wheel->real_time.field90.y,
			x87_mul(wheel->real_time.field90.z, 0.0f));
		wheel_axis.y = x87_sub(
			x87_mul(wheel->real_time.field90.z, 0.0f),
			wheel->real_time.field90.x);
		wheel_axis.z = x87_sub(
			x87_mul(wheel->real_time.field90.x, 0.0f),
			x87_mul(wheel->real_time.field90.y, 0.0f));
		axis_length_squared = model6_length_squared(&wheel_axis);
		if (F(axis_length_squared) <= F(MODEL6_LENGTH_EPSILON)) {
			wheel_axis = (GmVec3){ 1.0f, 0.0f, 0.0f };
		} else {
			float inverse = x87_div(
				1.0f, x87_sqrt(axis_length_squared));

			wheel_axis.x = x87_mul(inverse, wheel_axis.x);
			wheel_axis.y = x87_mul(inverse, wheel_axis.y);
			wheel_axis.z = x87_mul(inverse, wheel_axis.z);
		}
		if (wheel->steerable != 0) {
			float cosine = model6_cos(steering_angle);
			float negative_sine = -model6_sin(steering_angle);
			GmVec3 rotated;

			rotated.x = x87_add(
				x87_mul(cosine, wheel_axis.x),
				x87_mul(negative_sine, 0.0f));
			rotated.y = x87_add(
				x87_mul(negative_sine, 0.0f),
				x87_mul(cosine, wheel_axis.y));
			rotated.z = x87_add(
				negative_sine,
				x87_mul(cosine, wheel_axis.z));
			wheel_axis = rotated;
		}
		velocity_on_axis = model6_dot_xyz(local_speed, &wheel_axis);

		{
			GmVec3 rollover_axis;
			float axis_scale = -tuning->burnout_trigger_scale;
			float axis_length;

			rollover_axis.x = x87_mul(
				axis_scale, state->rollover_axis.x);
			rollover_axis.y = x87_mul(
				state->rollover_axis.y, axis_scale);
			rollover_axis.z = x87_mul(
				axis_scale, state->rollover_axis.z);
			axis_length = model6_length(&rollover_axis);
			if (F(axis_length)
					< F(tuning->rollover_axis_min_length))
				rollover_axis =
					(GmVec3){ 0.0f, 0.0f, 0.0f };
			torque = model6_cross(
				&relative_contact, &rollover_axis);
			torque.x = x87_mul(
				x87_mul(torque.x, -1.0f),
				tuning->rollover_torque_x_scale);
			torque.y = 0.0f;
			torque.z = x87_mul(
				x87_mul(torque.z, -1.0f),
				tuning->rollover_torque_z_scale);
			CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
		}
		if (vehicle->drive_mode == 1) {
			torque.x = model6_burnout_rollover(curves, local_speed->z);
			torque.y = 0.0f;
			torque.z = 0.0f;
			CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
			enter_scale = model6_enter_wave(context, tick);
		}

		damper_modulation =
			CSceneVehicleCarTuning_M6GetModulationFromDamperAbsorbVal(
				curves, wheel->real_time.damper_absorb);
		sliding_scale = wheel->real_time.is_sliding == 0
			? 1.0f : tuning->sliding_lateral_limit_scale;
		braking_scale =
			wheel->real_time.is_sliding != 0
				&& MODEL6_POINT_ONE < F(vehicle->input_brake)
			? tuning->braking_lateral_limit_scale : 1.0f;
		maximum_force = x87_mul(
			x87_mul(
				x87_mul(
					x87_mul(
						x87_mul(
							wheel_material->values[3],
							lateral_force_factor),
						model6_max_side_friction(
							curves, local_speed->z)),
					sliding_scale),
				braking_scale),
			damper_modulation);
		requested_force = x87_mul(
			x87_mul(
				x87_mul(
					-tuning->lateral_force_scale, 0.5f),
				velocity_on_axis),
			enter_scale);
		if (model6_abs(requested_force) <= F(maximum_force)) {
			wheel->real_time.is_sliding = 0;
		} else {
			float signed_limit = F(requested_force) <= 0.0
				? -maximum_force : maximum_force;

			wheel->real_time.is_sliding = 1;
			requested_force = x87_add(
				x87_mul(
					x87_sub(1.0f, tuning->lateral_overflow_blend),
					signed_limit),
				x87_mul(
					tuning->lateral_overflow_blend,
					requested_force));
		}
		if (wheel->real_time.is_sliding != 0)
			*sliding = 1;
		force.x = x87_mul(requested_force, wheel_axis.x);
		force.y = x87_mul(requested_force, wheel_axis.y);
		force.z = x87_mul(wheel_axis.z, requested_force);
		if (all_material6 && 6.0 < F(local_speed->z)) {
			model6_add_material6_forces(
				context, local_speed, &force);
			continue;
		}
		CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);
	}

	if (grounded == 0) {
		if (vehicle->drive_mode == 1) {
			state->burnout_transition_tick = tick;
			vehicle->drive_mode = 3;
		}
		goto store_sliding;
	}

	{
		float speed_length = model6_length(local_speed);

		if (vehicle->flag_60c == 0
			&& MODEL6_POINT_ONE < F(vehicle->input_brake)
			&& F(vehicle->input_gas) < MODEL6_POINT_ONE
			&& vehicle->drive_mode == 1) {
			state->burnout_transition_tick = tick;
			vehicle->drive_mode = 3;
		}
		if (vehicle->flag_60c == 0) {
			if (MODEL6_POINT_ONE < F(vehicle->input_gas)
				&& MODEL6_POINT_ONE < F(vehicle->input_brake)
				&& F(local_speed->z) < F(tuning->burnout_speed_min)
				&& MODEL6_POINT_SEVEN_FIVE < F(upright_value)) {
				vehicle->drive_mode = 1;
				vehicle->force_wheel_speed = 1;
				state->burnout_start_tick = tick;
			}
			if (MODEL6_POINT_ONE < F(vehicle->input_gas)
				&& MODEL6_POINT_ONE < F(vehicle->input_brake)
				&& F(local_speed->z) < F(tuning->burnout_speed_max)
				&& F(tuning->burnout_speed_min)
					< F(local_speed->z)
				&& F(MODEL6_CONTROL_EPSILON)
					<= F(model6_abs(steering_angle))) {
				GmVec3 accumulated = { 0.0f, 0.0f, 0.0f };
				GmVec3 tangent;
				GmVec3 pivot_delta;
				GmVec3 pivot_axis;
				GmVec3 local_orbit_direction;
				GmVec3 transformed_reference;
				float angle;
				float direction =
					model6_sign_bits(steering_angle);

				vehicle->drive_mode = 2;
				state->orbit_sign = direction;
				for (uint32_t index = 0;
				     index < wheel_count; ++index) {
					CSceneVehicleCarWheel *wheel =
						&vehicle->wheels[index];

					if (wheel->real_time.has_ground_contact != 0) {
						accumulated.x = x87_add(
							wheel->real_time.field90.x,
							accumulated.x);
						accumulated.y = x87_add(
							wheel->real_time.field90.y,
							accumulated.y);
						accumulated.z = x87_add(
							wheel->real_time.field90.z,
							accumulated.z);
					}
				}
				if (!model6_normalize(&accumulated)) {
					vehicle->drive_mode = 0;
					accumulated =
						(GmVec3){ 0.0f, 1.0f, 0.0f };
				}
				state->orbit_axis = accumulated;
				pivot_delta = model6_sub_vec3(
					&state->pivot_position,
					context->body_reference_position);
				pivot_axis.x = x87_mul(
					state->pivot_axis.y, 0.0f);
				pivot_axis.y = x87_mul(
					state->pivot_axis.x, 0.0f);
				pivot_axis.z = state->pivot_axis.z;
				pivot_delta.x = x87_add(
					pivot_axis.x, pivot_delta.x);
				pivot_delta.y = x87_add(
					pivot_axis.y, pivot_delta.y);
				pivot_delta.z = x87_add(
					pivot_axis.z, pivot_delta.z);
				state->orbit_initial_radius =
					model6_length(&pivot_delta);
				tangent = model6_cross(
					&state->orbit_axis, local_speed);
				tangent.x = x87_mul(direction, tangent.x);
				tangent.y = x87_mul(direction, tangent.y);
				tangent.z = x87_mul(direction, tangent.z);
				(void)model6_normalize(&tangent);
				GmVec3_Mult_Mat3(
					&state->orbit_axis,
					(const GmMat3 *)&state->model_iso);
				if (F(state->orbit_axis.y)
						< MODEL6_POINT_SEVEN_FIVE) {
					vehicle->drive_mode = 0;
					state->orbit_axis =
						(GmVec3){ 0.0f, 1.0f, 0.0f };
				}
				const GmVec3 forward_axis = { 0.0f, 0.0f, 1.0f };
				angle = GmVec3_GetAngle(&forward_axis, &tangent);
				angle = x87_mul(direction, angle);
				if (F(tuning->donut_angle_positive_limit)
						< F(angle)
					|| F(angle)
						< F(-tuning->donut_angle_negative_limit)) {
					vehicle->drive_mode = 0;
				} else {
					float radius = model6_burnout_radius(
						curves, local_speed->z);

					GmVec3_SetMult_Mat3(
						&local_orbit_direction,
						&tangent,
						(const GmMat3 *)&state->model_iso);
					GmVec3_SetMult_Iso4(
						&transformed_reference,
						context->body_reference_position,
						&state->model_iso);
					radius = x87_add(
						radius,
						state->orbit_initial_radius);
					state->orbit_radius = radius;
					state->orbit_center.x = x87_add(
						transformed_reference.x,
						x87_mul(
							radius,
							local_orbit_direction.x));
					state->orbit_center.y = x87_add(
						x87_mul(
							radius,
							local_orbit_direction.y),
						transformed_reference.y);
					state->orbit_center.z = x87_add(
						transformed_reference.z,
						x87_mul(
							radius,
							local_orbit_direction.z));
				}
				vehicle->force_wheel_speed =
					vehicle->drive_mode == 2;
			}
		}

		if (vehicle->drive_mode == 0) {
			if (MODEL6_POINT_ONE < F(vehicle->input_brake)
				&& F(local_speed->z)
					< F(state->reverse_speed_threshold)
				&& model6_abs(local_speed->x) < 2.0)
				state->reverse_mode = 1;
			if (MODEL6_POINT_ONE < F(vehicle->input_gas)
				&& (0.0 < F(local_speed->z)
					|| 2.0 < F(model6_abs(local_speed->x))))
				state->reverse_mode = 0;
			if (F(vehicle->input_gas) < MODEL6_POINT_ONE
				&& F(vehicle->input_brake) < MODEL6_POINT_ONE) {
				if ((0.0 < F(local_speed->z))
					|| model6_abs(local_speed->z) < 2.0)
					state->reverse_mode = 0;
				else
					state->reverse_mode = 1;
			}
			if (0.0 < F(local_speed->z)
				&& aux->turbo_type != TMNF_TURBO_NONE)
				state->reverse_mode = 0;
		} else {
			state->reverse_mode = 0;
		}

		{
			float speed_ratio = x87_div(
				x87_mul(local_speed->x, local_speed->x),
				x87_add_double(
					model6_abs(local_speed->z), 1.0));
			float direction =
				model6_sign_bits(-local_speed->x);
			GmVec3 torque = {
				0.0f,
				0.0f,
				x87_mul(
					model6_rollover_ratio(curves, speed_ratio),
					direction),
			};

			CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
		}

		{
			float total_requested = 0.0f;
			float total_limited = 0.0f;
			for (uint32_t index = 0;
			     index < wheel_count; ++index) {
				CSceneVehicleCarWheel *wheel =
					&vehicle->wheels[index];
				float half_width = x87_mul_double(
					wheel->steerable != 0
						? state->axle_width
						: -state->axle_width,
					0.5);
				GmVec3 offset_speed;
				float steer_modulation;
				float maximum;
				float requested;
				GmVec3 torque;

				offset_speed.x = x87_add(
					x87_mul(
						local_angular_speed->y,
						half_width),
					local_speed->x);
				offset_speed.y = x87_add(
					local_speed->y, 0.0f);
				offset_speed.z = x87_add(
					local_speed->z, 0.0f);
				if (F(speed_length) < F(MODEL6_POINT_SEVEN)) {
					steer_modulation = 0.0f;
				} else if (F(speed_length)
						<= F(tuning->wheel_steer_sine_limit)) {
					float angle = x87_mul_double(
						x87_div(
							speed_length,
							tuning->wheel_steer_sine_limit),
						MODEL6_PI);

					steer_modulation =
						model6_sin(x87_mul_double(
							angle, 0.5));
				} else {
					steer_modulation = 1.0f;
				}
				maximum = x87_mul(
					model6_max_side_friction(
						curves, local_speed->z),
					material->lateral_grip);
				requested = x87_mul(
					x87_mul(
						-tuning->lateral_force_scale,
						0.5f),
					offset_speed.x);
				if (F(maximum) < F(model6_abs(requested))) {
					float absolute_request =
						model6_abs(requested);
					float blended = x87_add(
						x87_mul(
							maximum,
							x87_sub(
								1.0f,
								tuning->wheel_overflow_blend)),
						x87_mul(
							absolute_request,
							tuning->wheel_overflow_blend));

					total_limited = x87_add(
						total_limited, maximum);
					total_requested = x87_add(
						absolute_request,
						total_requested);
					requested = x87_mul(
						model6_sign_bits(requested),
						blended);
					any_sliding = 1;
				}
				requested = x87_mul(
					tuning->wheel_torque_scale, requested);
				if (wheel->steerable != 0) {
					float reverse_direction =
						state->reverse_mode == 0
						? 1.0f : -1.0f;
					float wheel_slide_scale =
						wheel->real_time.is_sliding == 0
						? 1.0f
						: tuning->sliding_steer_torque_scale;
					float drive_torque = model6_steer_drive_torque(
						curves, local_speed->z);
					float correction = x87_mul(
						x87_mul(
							x87_mul(
								x87_mul(
									reverse_direction,
									steer_modulation),
								aux->steering_value),
							drive_torque),
						wheel_slide_scale);

					requested = x87_sub(
						requested, correction);
				}
				torque.x = x87_sub(
					x87_mul(0.0f, 0.0f),
					x87_mul(0.0f, half_width));
				torque.y = x87_sub(
					x87_mul(requested, half_width),
					x87_mul(0.0f, 0.0f));
				torque.z = x87_sub(
					x87_mul(0.0f, 0.0f),
					x87_mul(requested, 0.0f));
				CSceneVehicleCar_AddVehicleTorque(
					vehicle, &torque);
			}

			if (any_sliding) {
				state->last_sliding_tick = tick;
				if (was_sliding == 0)
					state->sliding_start_tick = tick;
				state->sliding_elapsed_ticks =
					tick - state->sliding_start_tick;
			}

			{
				float traction = 1.0f;
				float slipping_acceleration;
				float normal_acceleration;
				float acceleration;
				float steer_slowdown;
				float mode_scale = 1.0f;
				float mode_extra = 0.0f;
				float reverse_bias =
					state->reverse_mode == 0 ? 0.0f : -1.0f;
				float turbo_factor =
					aux->turbo_type == TMNF_TURBO_NONE
					? 0.0f : aux->turbo_factor;
				float direction =
					state->reverse_mode == 0 ? 1.0f : -1.0f;
				float brake = 0.0f;
				float net;
				GmVec3 force;

				if (tick == state->last_sliding_tick
					&& F(MODEL6_CONTROL_EPSILON)
						< F(total_limited)) {
					float loss = x87_div(
						x87_div(
							x87_sub(
								total_requested,
								total_limited),
							total_limited),
						tuning->traction_loss_scale);

					if (F(loss) < 0.0)
						loss = 0.0f;
					else if (1.0 < F(loss))
						loss = 1.0f;
					traction = x87_sub(1.0f, loss);
				}
				slipping_acceleration = model6_slipping_accel(
					curves, local_speed->z);
				normal_acceleration =
					state->reverse_mode == 0
					? model6_accel(curves, local_speed->z)
					: model6_rear_gear_accel(
						curves, local_speed->z);
				if (vehicle->engine_mode == 1) {
					acceleration = 0.0f;
				} else {
					acceleration = x87_add(
						x87_mul(
							x87_sub(1.0f, traction),
							slipping_acceleration),
						x87_mul(
							normal_acceleration,
							traction));
				}
				steer_slowdown = x87_mul(
					x87_mul(
						tuning->steer_slowdown_scale,
						model6_abs(aux->steering_value)),
					model6_steer_slowdown(
						curves, local_speed->z));
				if (vehicle->drive_mode == 1)
					mode_scale =
						model6_accel_wave(context, tick);
				if (vehicle->drive_mode == 3)
					mode_scale = model6_exit_wave(
						context, tick, &mode_extra);
				acceleration = x87_add(
					mode_extra,
					x87_sub(
						x87_mul(
							mode_scale,
							x87_add(
								x87_mul(
									normal_acceleration,
									turbo_factor),
								x87_mul(
									x87_add(
										x87_mul(
											vehicle->input_gas,
											material->braking),
										x87_mul(
											x87_mul(
												reverse_bias,
												material->braking),
											vehicle->input_brake)),
									acceleration))),
						x87_mul(
							steer_slowdown,
							direction)));
				if (water_contact != 0)
					acceleration =
						x87_mul_double(acceleration, 0.5);
				if (vehicle->flag_60c != 0) {
					acceleration =
						aux->turbo_type == TMNF_TURBO_NONE
						? x87_mul(
							normal_acceleration, 0.0f)
						: x87_mul(
							normal_acceleration,
							aux->turbo_factor);
				}

				if (0.0 < F(local_speed->z)) {
					float product = 1.0f;
					float limit;

					for (uint32_t index = 0;
					     index < wheel_count; ++index) {
						if (vehicle->wheels[index]
							    .real_time.is_sliding != 0)
							product = x87_mul(
								tuning->sliding_brake_scale,
								product);
					}
					brake = x87_mul(
						x87_mul(
							x87_add(
								x87_mul(
									tuning->brake_speed_scale,
									local_speed->z),
								tuning->brake_base),
							vehicle->input_brake),
						product);
					limit = x87_mul(
						material->steering,
						*sliding != 0
							? tuning->forward_brake_limit_sliding
							: tuning->forward_brake_limit);
					if (F(limit) < F(brake)) {
						brake = limit;
						any_sliding = 1;
						for (uint32_t index = 0;
						     index < wheel_count; ++index)
							vehicle->wheels[index]
								.real_time.is_sliding = 1;
					}
				}
				if (F(local_speed->z) < 0.0
					&& MODEL6_POINT_ONE
						< F(vehicle->input_gas)) {
					float product = 1.0f;
					float limit;

					if (vehicle->flag_60c == 0) {
						float trigger = x87_mul(
							x87_mul(
								-acceleration,
								tuning->burnout_trigger_scale),
							local_speed->z);

						if (F(tuning->burnout_trigger_limit)
								< F(trigger)
							&& MODEL6_POINT_SEVEN_FIVE
								< F(upright_value)) {
							state->burnout_start_tick = tick;
							vehicle->drive_mode = 1;
							vehicle->force_wheel_speed = 1;
						}
					}
					for (uint32_t index = 0;
					     index < wheel_count; ++index) {
						if (vehicle->wheels[index]
							    .real_time.is_sliding != 0)
							product = x87_mul(
								tuning->sliding_brake_scale,
								product);
					}
					brake = x87_mul(
						x87_mul(
							x87_sub(
								tuning->brake_base,
								x87_mul(
									tuning->brake_speed_scale,
									local_speed->z)),
							vehicle->input_gas),
						product);
					limit = x87_mul(
						material->steering,
						*sliding != 0
							? tuning->reverse_brake_limit_sliding
							: tuning->reverse_brake_limit);
					if (F(limit) < F(brake)) {
						brake = limit;
						any_sliding = 1;
						for (uint32_t index = 0;
						     index < wheel_count; ++index)
							vehicle->wheels[index]
								.real_time.is_sliding = 1;
					}
				}
				*brake_force = brake;
				net = x87_sub(
					acceleration,
					x87_mul(
						model6_sign_bits(local_speed->z),
						brake));
				if (F(x87_mul(
					    tuning->forward_speed_limit_scale,
					    material->acceleration))
						< F(local_speed->z)) {
					net = 0.0 <= F(net)
						? -tuning->speed_limit_force
						: x87_sub(
							net,
							tuning->speed_limit_force);
				}
				if (F(local_speed->z)
						< F(-x87_mul(
							tuning->reverse_speed_limit_scale,
							material->acceleration))) {
					net = F(net) <= 0.0
						? tuning->speed_limit_force
						: x87_add(
							tuning->speed_limit_force,
							net);
				}
				force.x = 0.0f;
				force.y = 0.0f;
				force.z = x87_mul(
					net, longitudinal_force_factor);
				CSceneVehicleCar_AddVehicleCentralForce(
					vehicle, &force);
				force.x = 0.0f;
				force.y = 0.0f;
				force.z = x87_div(
					x87_mul(
						-tuning->vertical_force_scale,
						existing_force->z),
					tuning->vertical_force_divisor);
				CSceneVehicleCar_AddVehicleCentralForce(
					vehicle, &force);
			}
		}
	}

store_sliding:
	vehicle->turbo_active = any_sliding;

finish:
	vehicle->engine.reverse = state->reverse_mode;
	context->contact->friction_input_selector =
		state->reverse_mode;
	vehicle->current_local_speed = *local_speed;
}

/* Adapter consumed by the composed 0x007C69E0 top caller. */
TMNF_HD void VehicleModel6_ComputeForces(
	TMNFVehicleComputeContext *context, float dt,
	const GmVec3 *existing_force, float slope_adherence,
	float slope_secondary, const GmVec3 *linear_speed,
	const GmVec3 *angular_speed, float steering_angle,
	int has_ground_material,
	const CSceneVehicleMaterialBlendableVals *ground_material,
	int *air_control_reset, float *effect_curve_position) {
	CSceneVehicleCarModel6Context *model6 = context->model6;
	if (model6 == NULL || model6->vehicle != context->vehicle) {
		tmnf_abort();
	}
	CSceneVehicleCar_ComputeForcesModel6(
		model6, dt, existing_force, slope_adherence, slope_secondary,
		linear_speed, angular_speed, steering_angle,
		has_ground_material, ground_material,
		air_control_reset, effect_curve_position);
}
