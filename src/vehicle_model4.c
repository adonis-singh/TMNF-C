/*
 * 0x007FB5F0 CSceneVehicleCar::ComputeForcesModel4 (879 instructions, 437
 * x87). Selected for CSceneVehicleCarTuning+0x354 == 3. Unlike Models 3, 5
 * and 6 it has no per-wheel lateral forces: two whole-body lateral friction
 * evaluations (CSceneVehicleCar::GetLateralFriction, 0x007BF080) act along a
 * yaw-rotated axis and the car x axis, and a drift state at car +0x638..
 * +0x640 rotates the drive direction while the car slides.
 *
 * PORTED, NO ORACLE TRACE YET.
 */
#include "tmnf_hd.h"
#include <math.h>
#include <stdlib.h>

#include "tmnf_fp.h"
#include "vehicle_oldmodels.h"
#include "vehicle_oldmodels_common.h"

TMNF_HD static void add_yaw_torque(CSceneVehicleCar *vehicle, float value)
{
	/* The x and z components are the double 0.0 at 0x00B2C178 times the
	 * value, so they carry its sign. */
	float zero = x87_mul_double(value, 0.0);
	GmVec3 torque = { zero, value, zero };

	CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
}

/* 0x007FB93D..0x007FBBE4: steering torque about y. */
TMNF_HD static void model4_steer_torque(
	CSceneVehicleCarOldModelsContext *context, float speed_length,
	int drifting)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	const CSceneVehicleCarOldModelsTuning *tuning = context->tuning;
	CSceneVehicleCarOldModelsState *state = context->state;
	float direction = context->shared->reverse_mode == 0 ? 1.0f : -1.0f;
	float steering = context->aux->steering_value;
	float absolute_speed = oldmodels_abs(speed_length);

	if (state->m4_drift_state == 2) {
		float absolute_angle = oldmodels_abs(state->m4_drift_angle);

		if (F(OLDMODELS_CONTROL_EPSILON) < F(absolute_angle)
			&& F(OLDMODELS_CONTROL_EPSILON)
				< F(oldmodels_abs(
					tuning->m4_drift_angle_radius_scale))) {
			/* 0x007FB98A..0x007FBAC7. */
			float angle_sign =
				oldmodels_sign_compare(state->m4_drift_angle);
			float radius = oldmodels_div(
				CSceneVehicleCarTuning_M4GetSteerRadiusFromSpeed(
					context->curves, absolute_speed),
				x87_mul(
					tuning->m4_drift_angle_radius_scale,
					absolute_angle));

			if (F(OLDMODELS_CONTROL_EPSILON) < F(radius)) {
				float product = x87_mul(angle_sign, direction);
				float torque = oldmodels_div(
					x87_mul(
						x87_mul(
							x87_mul(-product, speed_length),
							tuning->m4_steer_torque_speed_scale),
						tuning->m4_yaw_damping_linear),
					radius);

				add_yaw_torque(vehicle, torque);
			}
			return;
		}
	}

	/* 0x007FBA9D..0x007FBBE4. */
	if (F(OLDMODELS_CONTROL_EPSILON) < F(oldmodels_abs(steering))) {
		float steer_sign = oldmodels_sign_compare(steering);
		float product = x87_mul(steer_sign, direction);
		float radius_scale = drifting != 0
			? tuning->m4_drift_steer_radius_scale : 1.0f;
		float radius = x87_mul(
			CSceneVehicleCarTuning_M4GetSteerRadiusFromSpeed(
				context->curves, absolute_speed),
			radius_scale);

		if (F(OLDMODELS_CONTROL_EPSILON) < F(radius)) {
			float torque = oldmodels_div(
				x87_mul(
					oldmodels_abs(steering),
					x87_mul(
						x87_mul(
							x87_mul(-product, speed_length),
							tuning->m4_steer_torque_speed_scale),
						tuning->m4_yaw_damping_linear)),
				radius);

			add_yaw_torque(vehicle, torque);
		}
	}
}

/* 0x007FBE3E..0x007FBFEB: drift state transitions. Returns the new
 * sliding flag. */
TMNF_HD static int model4_update_drift(
	CSceneVehicleCarOldModelsContext *context, float dt, int any_sliding,
	int drifting, float lateral_speed)
{
	const CSceneVehicleCarOldModelsTuning *tuning = context->tuning;
	CSceneVehicleCarOldModelsState *state = context->state;
	float steering = context->aux->steering_value;
	float angle;
	int flipped;

	if (any_sliding == 0) {
		if (drifting == 0)
			return any_sliding;
	} else if (drifting == 0) {
		state->m4_drift_angle = 0.0f;
		state->m4_drift_steer_sign = oldmodels_sign_compare(steering);
		state->m4_drift_state = 1;
		return any_sliding;
	}

	/* 0x007FBE97..0x007FBF2D: ((rate * steer) * dt) + angle, clipped. */
	angle = x87_add(
		x87_mul(
			x87_mul(tuning->m4_drift_angle_rate, steering), dt),
		state->m4_drift_angle);
	state->m4_drift_angle = angle;
	if (F(tuning->m4_drift_angle_limit) < F(oldmodels_abs(angle)))
		state->m4_drift_angle = x87_mul(
			tuning->m4_drift_angle_limit,
			oldmodels_sign_compare(angle));

	flipped = (0.0 < F(state->m4_drift_steer_sign)
			&& F(state->m4_drift_angle) < 0.0)
		|| (F(state->m4_drift_steer_sign) < 0.0
			&& 0.0 < F(state->m4_drift_angle));
	if ((F(oldmodels_abs(lateral_speed))
			< F(tuning->m4_drift_exit_lateral_speed)
		&& F(oldmodels_abs(steering)) < F(OLDMODELS_CONTROL_EPSILON))
		|| flipped) {
		state->m4_drift_angle = 0.0f;
		state->m4_drift_state = 0;
		return 0;
	}
	return 1;
}

TMNF_HD void CSceneVehicleCar_ComputeForcesModel4(
	CSceneVehicleCarOldModelsContext *context, float dt,
	const GmVec3 *existing_force, float lateral_force_factor,
	float longitudinal_force_factor, const GmVec3 *local_speed,
	const GmVec3 *local_angular_speed, float steering_angle,
	int grounded, const CSceneVehicleMaterialBlendableVals *material,
	int *sliding, float *brake_force)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	const CSceneVehicleCarOldModelsTuning *tuning = context->tuning;
	CSceneVehicleCarOldModelsState *state = context->state;
	int drifting;
	int any_sliding = 0;
	float speed_length;
	float yaw_angle = 0.0f;
	float sine;
	float cosine;
	float forward_speed;
	float lateral_speed;
	float acceleration;
	float brake;
	float net;
	float longitudinal;
	GmVec3 velocity = *local_speed;
	GmVec3 rotated_axis;
	GmVec3 side_axis = { 1.0f, 0.0f, 0.0f };
	GmVec3 force;
	float rotated_friction;
	float side_friction;
	int rotated_sliding;
	int side_sliding;

	(void)steering_angle;
	(void)existing_force;
	for (uint32_t index = 0; index < vehicle->wheel_count; ++index)
		CSceneVehicleCar_WheelAddForceToVehicle(
			vehicle, &vehicle->wheels[index]);

	drifting = state->m4_drift_state != 0;
	if (grounded == 0)
		goto finish;

	/* 0x007FB645..0x007FB68F. */
	speed_length = x87_sqrt(x87_add(
		x87_add(
			x87_mul(local_speed->y, local_speed->y),
			x87_mul(local_speed->x, local_speed->x)),
		x87_mul(local_speed->z, local_speed->z)));
	VehicleOldModels_UpdateReverseLatch(context, speed_length);

	/* 0x007FB6DA..0x007FB71A: a released wheel while entering the drift
	 * freezes the drift angle at the velocity heading, atan2(vx, vz). */
	if (state->m4_drift_state == 1
		&& F(oldmodels_abs(context->aux->steering_value))
			< F(OLDMODELS_CONTROL_EPSILON)) {
		state->m4_drift_angle =
			oldmodels_atan2(local_speed->x, local_speed->z);
		state->m4_drift_state = 2;
	}
	if (state->m4_drift_state == 1)
		yaw_angle = x87_mul(
			-context->aux->steering_value,
			tuning->m4_drift_entry_angle_scale);
	else if (state->m4_drift_state == 2)
		yaw_angle = state->m4_drift_angle;
	sine = oldmodels_sin(yaw_angle);
	cosine = oldmodels_cos(yaw_angle);
	rotated_axis = (GmVec3){ cosine, 0.0f, -sine };
	/* 0x007FB7DC..0x007FB7FD. */
	forward_speed = x87_add(
		x87_mul(cosine, local_speed->z),
		x87_add(
			x87_mul(sine, local_speed->x),
			x87_mul(local_speed->y, 0.0f)));
	lateral_speed = x87_add(
		x87_add(x87_mul(local_speed->y, 0.0f), local_speed->x),
		x87_mul(0.0f, local_speed->z));

	CSceneVehicleCar_GetLateralFriction(
		context->contact, &velocity, &rotated_axis, material,
		lateral_force_factor, drifting, &rotated_friction,
		&rotated_sliding);
	CSceneVehicleCar_GetLateralFriction(
		context->contact, &velocity, &side_axis, material,
		lateral_force_factor, drifting, &side_friction, &side_sliding);
	/* 0x007FB83A..0x007FB8AB. */
	rotated_friction = x87_mul_double(rotated_friction, 0.5);
	force.x = x87_mul(rotated_friction, rotated_axis.x);
	force.y = x87_mul(rotated_axis.y, rotated_friction);
	force.z = x87_mul(rotated_friction, rotated_axis.z);
	CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);
	side_friction = x87_mul_double(side_friction, 0.5);
	force.x = x87_mul(side_friction, side_axis.x);
	force.y = x87_mul(side_axis.y, side_friction);
	force.z = x87_mul(side_friction, side_axis.z);
	CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);

	/* 0x007FB8B8..0x007FB939: yaw damping -(wy*lin) - ((|wy|*wy)*quad). */
	{
		float yaw_speed = local_angular_speed->y;
		float damping = x87_sub(
			x87_mul(tuning->m4_yaw_damping_linear, -yaw_speed),
			x87_mul(
				x87_mul(oldmodels_abs(yaw_speed), yaw_speed),
				tuning->m4_yaw_damping_quadratic));

		add_yaw_torque(vehicle, damping);
	}
	model4_steer_torque(context, speed_length, drifting);

	/* 0x007FBBE4..0x007FBC9F. */
	acceleration = VehicleOldModels_InputAcceleration(
		context, material,
		CSceneVehicleCarTuning_GetAccelFromSpeed(
			context->curves, forward_speed),
		NAN, 0.0f);
	brake = VehicleOldModels_BrakeForce(
		context, material, forward_speed, *sliding, 0, &any_sliding);
	any_sliding = model4_update_drift(
		context, dt, any_sliding, drifting, lateral_speed);

	*brake_force = brake;
	net = x87_sub(acceleration, brake);
	net = VehicleOldModels_LimitLongitudinal(
		context, material, forward_speed, net);
	/* 0x007FC08C..0x007FC0FF: the force follows the yaw-rotated axis. */
	longitudinal = x87_mul(net, longitudinal_force_factor);
	force.x = x87_mul(longitudinal, sine);
	force.y = x87_mul_double(longitudinal, 0.0);
	force.z = x87_mul(longitudinal, cosine);
	CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);
	{
		GmVec3 torque = { 0.0f, 0.0f, 0.0f };

		torque.x = x87_mul(
			tuning->longitudinal_torque_scale, -longitudinal);
		CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
	}

finish:
	/* 0x007FC108..0x007FC162. */
	for (uint32_t index = 0; index < vehicle->wheel_count; ++index)
		vehicle->wheels[index].real_time.is_sliding = any_sliding;
	vehicle->turbo_active = any_sliding;
}
