#include <math.h>

#include "tmnf_fp.h"
#include "vehicle_model6_burnout.h"

#define NORMALIZE_LENGTH_SQUARED 9.999999439624929e-11f
#define STEER_EPSILON 0x1.4f8b58p-17f
#define TMNF_PI 0x1.921fb54442d18p+1

TMNF_HD static float axis_length_squared(const GmVec3 *value)
{
	float xy = x87_add(
		x87_mul(value->x, value->x),
		x87_mul(value->y, value->y));

	return x87_add(x87_mul(value->z, value->z), xy);
}

TMNF_HD static float radius_length_squared(const GmVec3 *value)
{
	float xy = x87_add(
		x87_mul(value->y, value->y),
		x87_mul(value->x, value->x));

	return x87_add(xy, x87_mul(value->z, value->z));
}

TMNF_HD static float dot_yxz(const GmVec3 *left, const GmVec3 *right)
{
	float yx = x87_add(
		x87_mul(left->y, right->y),
		x87_mul(left->x, right->x));

	return x87_add(yx, x87_mul(left->z, right->z));
}

/*
 * 0x007C3F47..0x007C4821, UNVALIDATED: maintain an active Model 6 burnout
 * orbit and rearm it when any continuation condition fails.
 */
TMNF_HD void VehicleModel6_ApplyBurnoutDonut(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	VehicleModel6Scratch *scratch)
{
	CSceneVehicleCar *car = context->car;
	VehicleModel6PersistentState *state = context->state;
	const VehicleModel6TuningScalars *scalars = context->scalars;
	const GmVec3 *local_speed = inputs->local_speed;
	const GmVec3 *local_angular_speed = inputs->local_angular_speed;
	GmVec3 contact_axis = { 0.0f, 0.0f, 0.0f };
	GmVec3 world_contact_axis;
	GmIso4 inverse_car_iso;
	GmVec3 center_local;
	GmVec3 radius_vector;
	GmVec3 radius_direction;
	GmVec3 tangent;
	GmVec3 force;
	GmVec3 torque;
	float steer_sign;
	float length_squared;
	float length;
	float inverse_length;
	float axis_angle;
	float radius;
	float tangent_speed;
	float radial_speed;
	float force_scale;
	float normalized_angle;
	float yaw_input;
	float orbit_rate;
	float orbit_correction;
	float yaw_torque;
	float rollover;
	float zero;
	uint32_t wheel_index;

	if (state->burnout_state != 2)
		return;

	if (F(car->input_gas) < 0.1
		|| F(car->input_brake) < 0.1
		|| fabs(F(inputs->steer_angle)) < F(STEER_EPSILON)) {
		state->burnout_state = 0;
		goto commit_transition;
	}

	steer_sign = GmFunc_Sign(inputs->steer_angle);
	if (state->burnout_steer_sign != steer_sign
		|| state->lateral_contact != 0
		|| 0 < state->contact_counter
		|| inputs->valid_ground_material == 0
		|| scratch->water_forces_applied != 0
		|| state->special_physics != 0) {
		state->burnout_state = 0;
		goto commit_transition;
	}

	for (wheel_index = 0; wheel_index < scratch->wheel_count; ++wheel_index) {
		const GmVec3 *wheel_axis =
			&car->wheels[wheel_index].real_time.field90;

		contact_axis.x = x87_add(wheel_axis->x, contact_axis.x);
		contact_axis.y = x87_add(wheel_axis->y, contact_axis.y);
		contact_axis.z = x87_add(wheel_axis->z, contact_axis.z);
	}

	length_squared = axis_length_squared(&contact_axis);
	if (!(F(NORMALIZE_LENGTH_SQUARED) < F(length_squared))) {
		contact_axis.x = 0.0f;
		contact_axis.y = 1.0f;
		contact_axis.z = 0.0f;
	} else {
		length = x87_sqrt(length_squared);
		inverse_length = x87_rcp(length);
		contact_axis.x = x87_mul(inverse_length, contact_axis.x);
		contact_axis.y = x87_mul(contact_axis.y, inverse_length);
		contact_axis.z = x87_mul(inverse_length, contact_axis.z);
	}

	GmVec3_SetMult_Mat3(
		&world_contact_axis, &contact_axis,
		(const GmMat3 *)&state->car_iso);
	axis_angle = (float)fabs(F(GmVec3_GetAngle(
		&world_contact_axis, &state->burnout_axis)));
	if (F(scalars->s28c) < F(axis_angle)) {
		state->burnout_state = 0;
		goto commit_transition;
	}

	GmIso4_SetInverse(&inverse_car_iso, &state->car_iso);
	GmVec3_SetMult_Iso4(
		&center_local, &state->burnout_center, &inverse_car_iso);
	radius_vector.x = x87_sub(
		center_local.x, scratch->body_center_local->x);
	radius_vector.y = x87_sub(
		center_local.y, scratch->body_center_local->y);
	radius_vector.z = x87_sub(
		center_local.z, scratch->body_center_local->z);

	length_squared = radius_length_squared(&radius_vector);
	radius = x87_sqrt(length_squared);
	radius_direction = radius_vector;
	if (F(NORMALIZE_LENGTH_SQUARED) < F(length_squared)) {
		inverse_length = x87_rcp(x87_sqrt(length_squared));
		radius_direction.x = x87_mul(inverse_length, radius_vector.x);
		radius_direction.y = x87_mul(radius_vector.y, inverse_length);
		radius_direction.z = x87_mul(inverse_length, radius_vector.z);
	}

	tangent.x = x87_sub(
		x87_mul(contact_axis.y, radius_direction.z),
		x87_mul(contact_axis.z, radius_direction.y));
	tangent.y = x87_sub(
		x87_mul(radius_direction.x, contact_axis.z),
		x87_mul(contact_axis.x, radius_direction.z));
	tangent.z = x87_sub(
		x87_mul(radius_direction.y, contact_axis.x),
		x87_mul(contact_axis.y, radius_direction.x));
	tangent_speed = dot_yxz(&tangent, local_speed);

	if (F(scalars->s284) < fabs(F(tangent_speed)))
		state->burnout_state = 0;

	if (F(state->burnout_initial_radius) < F(radius)
		|| F(radius) < F(scalars->s280)) {
		float radius_delta =
			x87_sub(radius, state->burnout_target_radius);
		float radius_term = x87_mul(radius_delta, scalars->s278);
		float velocity_term;
		float exponent;
		float exponential;
		float tangent_speed_squared;

		radial_speed = dot_yxz(&radius_direction, local_speed);
		velocity_term = x87_mul(radial_speed, scalars->s27c);
		exponent = x87_sub(radius_term, velocity_term);
		exponential = x87_exp(exponent);
		tangent_speed_squared = x87_mul(tangent_speed, tangent_speed);
		force_scale = x87_mul(
			exponential,
			x87_div(tangent_speed_squared, radius));
		force.x = x87_mul(force_scale, radius_direction.x);
		force.y = x87_mul(force_scale, radius_direction.y);
		force.z = x87_mul(force_scale, radius_direction.z);
		CSceneVehicleCar_AddVehicleCentralForce(car, &force);
	} else {
		state->burnout_state = 0;
	}

	for (wheel_index = 0; wheel_index < scratch->wheel_count; ++wheel_index) {
		CSceneVehicleCarWheel *wheel = &car->wheels[wheel_index];

		CSceneVehicleCar_WheelAddForceToVehicle(car, wheel);
		wheel->real_time.is_sliding = 0;
	}

	steer_sign = GmFunc_Sign(inputs->steer_angle);
	force_scale = x87_mul(
		scalars->s264,
		x87_sub(
			x87_mul(
				-steer_sign,
				CSceneVehicleCarTuning_M6GetLateralSpeedFromBurnoutRadius(
					context->curves, radius)),
			tangent_speed));
	force.x = x87_mul(force_scale, tangent.x);
	force.y = x87_mul(force_scale, tangent.y);
	force.z = x87_mul(force_scale, tangent.z);
	CSceneVehicleCar_AddVehicleCentralForce(car, &force);

	{
		const GmVec3 up = { 0.0f, 0.0f, 1.0f };
		float angle = GmVec3_GetAngle(&up, &radius_direction);

		normalized_angle = x87_r24(F(angle) / TMNF_PI);
	}
	if (GmFunc_IsANumber(normalized_angle) == 0) {
		state->burnout_state = 0;
		goto apply_rollover;
	}

	{
		float signed_angle = x87_r24(
			F(x87_mul(steer_sign, normalized_angle)) * TMNF_PI);

		if (F(signed_angle) < F(-scalars->s294)
			|| F(scalars->s290) < F(signed_angle)) {
			state->burnout_state = 0;
			goto apply_rollover;
		}
	}

	if (F(normalized_angle) <= 0.0) {
		if (F(local_angular_speed->y) <= 0.0) {
			float offset = x87_add(normalized_angle, 1.0f);
			float offset_squared = x87_mul(offset, offset);

			yaw_input = x87_mul(
				x87_mul(-scalars->s270, local_angular_speed->y),
				offset_squared);
		} else {
			yaw_input =
				x87_mul(-scalars->s26c, local_angular_speed->y);
		}
	} else if (0.0 <= F(local_angular_speed->y)) {
		float offset = x87_sub(normalized_angle, 1.0f);
		float offset_squared = x87_mul(offset, offset);

		yaw_input = x87_mul(
			x87_mul(-scalars->s270, local_angular_speed->y),
			offset_squared);
	} else {
		yaw_input = x87_mul(-scalars->s26c, local_angular_speed->y);
	}

	orbit_rate = x87_div(tangent_speed, radius);
	orbit_correction = 0.0f;
	if (F(normalized_angle) <= 0.0) {
		if (0.0 < F(orbit_rate))
			orbit_correction = x87_mul(-scalars->s274, orbit_rate);
	} else if (F(orbit_rate) < 0.0) {
		orbit_correction = x87_mul(-scalars->s274, orbit_rate);
	}

	yaw_torque = x87_add(
		x87_add(
			x87_mul(scalars->s268, normalized_angle),
			yaw_input),
		orbit_correction);
	zero = x87_mul(yaw_torque, 0.0f);
	torque.x = zero;
	torque.y = yaw_torque;
	torque.z = zero;
	CSceneVehicleCar_AddVehicleTorque(car, &torque);

apply_rollover:
	rollover = CSceneVehicleCarTuning_M6GetDonutRolloverFromSpeed(
		context->curves, (float)fabs(F(local_speed->x)));
	rollover = x87_mul(-GmFunc_Sign(local_speed->x), rollover);
	zero = x87_mul(rollover, 0.0f);
	torque.x = zero;
	torque.y = zero;
	torque.z = rollover;
	CSceneVehicleCar_AddVehicleTorque(car, &torque);

	rollover = CSceneVehicleCarTuning_M6GetBurnoutRolloverFromSpeed(
		context->curves, local_speed->z);
	torque.x = rollover;
	torque.y = 0.0f;
	torque.z = 0.0f;
	CSceneVehicleCar_AddVehicleTorque(car, &torque);

commit_transition:
	if (state->burnout_state != 2) {
		state->burnout_state = 1;
		state->burnout_start_tick = scratch->tick;
	}
}
