/*
 * 0x007FC170 CSceneVehicleCar::ComputeForcesModel5 (1,150 instructions, 583
 * x87). Selected for CSceneVehicleCarTuning+0x354 == 4. Model 6 grew out of
 * this function: the wheel lateral block, drive torque loop, traction blend
 * and brake tail have the same shape, without the burnout and donut states.
 *
 * PORTED, NO ORACLE TRACE YET.
 */
#include "tmnf_hd.h"
#include <math.h>
#include <stdlib.h>

#include "tmnf_fp.h"
#include "vehicle_oldmodels.h"
#include "vehicle_oldmodels_common.h"

/* 0x007FC690..0x007FC988: drive torque for wheel `index` of exactly four
 * (the loop bound is the immediate 4 at 0x007FC981). */
TMNF_HD static void model5_wheel_torque(
	CSceneVehicleCarOldModelsContext *context, CSceneVehicleCarWheel *wheel,
	float speed_length, const GmVec3 *local_speed,
	const GmVec3 *local_angular_speed,
	const CSceneVehicleMaterialBlendableVals *material,
	float *total_limited, float *total_requested, int *any_sliding)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	const CSceneVehicleCarOldModelsTuning *tuning = context->tuning;
	CSceneVehicleCarModel6State *shared = context->shared;
	float half_width = x87_mul_double(
		wheel->steerable != 0 ? shared->axle_width : -shared->axle_width,
		0.5);
	GmVec3 offset_speed;
	GmVec3 torque;
	float steer_modulation;
	float maximum;
	float requested;
	float requested_zero;

	offset_speed.x = x87_add(
		x87_mul(half_width, local_angular_speed->y), local_speed->x);
	offset_speed.y = x87_add(local_speed->y, 0.0f);
	offset_speed.z = x87_add(0.0f, local_speed->z);

	/* 0x007FC711..0x007FC754. */
	if (F(speed_length) <= F(tuning->wheel_steer_sine_limit)) {
		float angle = x87_mul_double(
			x87_mul_double(
				oldmodels_div(
					speed_length, tuning->wheel_steer_sine_limit),
				OLDMODELS_PI),
			0.5);

		steer_modulation = oldmodels_sin(angle);
	} else {
		steer_modulation = 1.0f;
	}

	maximum = x87_mul(
		CSceneVehicleCarTuning_GetMaxSideFrictionFromSpeed(
			context->curves, local_speed->z),
		material->lateral_grip);
	/* 0x007FC78E..0x007FC7BC. */
	requested = x87_mul(
		x87_add(
			x87_add(x87_mul(offset_speed.y, 0.0f), offset_speed.x),
			x87_mul(0.0f, offset_speed.z)),
		x87_mul_double(-tuning->lateral_force_scale, 0.5));
	if (F(maximum) < F(oldmodels_abs(requested))) {
		float blend = tuning->wheel_overflow_blend;
		float absolute = oldmodels_abs(requested);
		/* 0x007FC7FB..0x007FC821: (blend*|req|) + ((1-blend)*max). */
		float blended = x87_add(
			x87_mul(blend, absolute),
			x87_mul(x87_sub(1.0f, blend), maximum));

		*total_limited = x87_add(maximum, *total_limited);
		*total_requested = x87_add(absolute, *total_requested);
		*any_sliding = 1;
		/* 0x007FC83D: sign bit of the unclipped request. */
		requested = x87_mul(oldmodels_sign_bits(requested), blended);
	}
	requested = x87_mul(tuning->wheel_torque_scale, requested);
	if (wheel->steerable != 0) {
		float direction = shared->reverse_mode == 0 ? 1.0f : -1.0f;
		float slide_scale = wheel->real_time.is_sliding == 0
			? 1.0f : tuning->sliding_steer_torque_scale;
		float drive_torque =
			CSceneVehicleCarTuning_GetSteerDriveTorqueFromSpeed(
				context->curves, local_speed->z);

		requested = x87_sub(
			requested,
			x87_mul(
				x87_mul(
					drive_torque,
					x87_mul(
						x87_mul(direction, steer_modulation),
						context->aux->steering_value)),
				slide_scale));
	}
	/* 0x007FC91C..0x007FC971. */
	requested_zero = x87_mul(requested, 0.0f);
	torque.x = x87_sub(
		x87_mul(requested_zero, 0.0f),
		x87_mul(requested_zero, half_width));
	torque.y = x87_sub(
		x87_mul(half_width, requested),
		x87_mul(requested_zero, 0.0f));
	torque.z = x87_sub(
		x87_mul(requested_zero, 0.0f),
		x87_mul(0.0f, requested));
	CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
}

TMNF_HD void CSceneVehicleCar_ComputeForcesModel5(
	CSceneVehicleCarOldModelsContext *context, float dt,
	const GmVec3 *existing_force, float lateral_force_factor,
	float longitudinal_force_factor, const GmVec3 *local_speed,
	const GmVec3 *local_angular_speed, float steering_angle,
	int grounded, const CSceneVehicleMaterialBlendableVals *material,
	int *sliding, float *brake_force)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	CSceneVehicleCarAuxContext *aux = context->aux;
	const CSceneVehicleCarOldModelsTuning *tuning = context->tuning;
	CSceneVehicleCarModel6State *shared = context->shared;
	CSceneVehicleCarOldModelsState *state = context->state;
	int water_contact;
	int was_sliding = vehicle->turbo_active;
	int any_sliding = 0;
	float speed_length;
	float total_limited = 0.0f;
	float total_requested = 0.0f;
	float traction = 1.0f;
	float slipping_acceleration;
	float normal_acceleration;
	float blended_acceleration;
	float steer_gate = 0.0f;
	float steer_slowdown;
	float acceleration;
	float brake;
	float net;
	uint32_t tick;
	GmVec3 force = { 0.0f, 0.0f, 0.0f };

	(void)dt;
	/* 0x007FC184..0x007FC190: the water result is car +0x5e4. */
	water_contact = CSceneVehicleCar_ApplyWaterForces(aux, existing_force);
	aux->air_control_locked = water_contact;
	context->contact->airborne_friction_gate = water_contact;

	for (uint32_t index = 0; index < vehicle->wheel_count; ++index) {
		CSceneVehicleCarWheel *wheel = &vehicle->wheels[index];

		CSceneVehicleCar_WheelAddForceToVehicle(vehicle, wheel);
		if (wheel->real_time.has_ground_contact == 0)
			continue;
		if (!(0.0 < F(tuning->lateral_force_scale)))
			continue;
		VehicleOldModels_WheelLateral(
			context, wheel, lateral_force_factor, local_speed,
			steering_angle, sliding);
	}
	if (grounded == 0) {
		vehicle->turbo_active = 0;
		return;
	}

	/* 0x007FC5FA..0x007FC637. */
	speed_length = x87_sqrt(x87_add(
		x87_add(
			x87_mul(local_speed->y, local_speed->y),
			x87_mul(local_speed->x, local_speed->x)),
		x87_mul(local_speed->z, local_speed->z)));
	VehicleOldModels_UpdateReverseLatch(context, speed_length);

	if (vehicle->wheel_count < 4)
		tmnf_abort();
	for (uint32_t index = 0; index < 4; ++index)
		model5_wheel_torque(
			context, &vehicle->wheels[index], speed_length,
			local_speed, local_angular_speed, material,
			&total_limited, &total_requested, &any_sliding);

	tick = *CMwTimerAdapter_GetTickTime(context->contact->timer);
	if (any_sliding) {
		shared->last_sliding_tick = tick;
		if (was_sliding == 0)
			shared->sliding_start_tick = tick;
		shared->sliding_elapsed_ticks = tick - shared->sliding_start_tick;
	}
	/* 0x007FC9DC..0x007FCA4E. */
	if (tick == shared->last_sliding_tick
		&& F(OLDMODELS_CONTROL_EPSILON) < F(total_limited)) {
		float loss = oldmodels_div(
			oldmodels_div(
				x87_sub(total_requested, total_limited),
				total_limited),
			tuning->traction_loss_scale);
		float clamped = 0.0f;

		if (0.0 < F(loss)) {
			clamped = loss;
			if (1.0 < F(loss))
				clamped = 1.0f;
		}
		traction = x87_sub(1.0f, clamped);
	}
	slipping_acceleration =
		CSceneVehicleCarTuning_M5GetSlippingAccelFromSpeed(
			context->curves, local_speed->z);
	normal_acceleration = CSceneVehicleCarTuning_M5GetAccelFromSpeed(
		context->curves, local_speed->z);
	/* 0x007FCA9C..0x007FCAAE: (t*accel) + ((1-t)*slipping). */
	blended_acceleration = x87_add(
		x87_mul(traction, normal_acceleration),
		x87_mul(x87_sub(1.0f, traction), slipping_acceleration));

	/* 0x007FCAB7..0x007FCB81: steer slow-down gate from the input tick
	 * history at car +0x648/+0x64c. */
	if (F(OLDMODELS_CONTROL_EPSILON)
			< F(oldmodels_abs(vehicle->input_steer))) {
		state->m5_last_steer_tick = tick;
		state->m5_last_steer_sliding = any_sliding;
	}
	if (state->m5_last_steer_tick <= tick
		&& tick - state->m5_last_steer_tick
			< tuning->m5_steer_gate_ticks) {
		steer_gate = state->m5_last_steer_sliding == 0
			|| tuning->m5_steer_gate_needs_sliding == 0
			? 1.0f : 0.0f;
	}
	{
		uint32_t window = shared->sliding_elapsed_ticks;

		if (tuning->m5_steer_gate_sliding_ticks < window)
			window = tuning->m5_steer_gate_sliding_ticks;
		if (tuning->m5_steer_gate_needs_sliding != 0
			&& shared->last_sliding_tick <= tick
			&& tick - shared->last_sliding_tick <= window)
			steer_gate = 0.0f;
	}
	steer_slowdown = x87_mul(
		CSceneVehicleCarTuning_M5GetSteerSlowDownFromSpeed(
			context->curves, local_speed->z),
		x87_mul(tuning->steer_slowdown_scale, steer_gate));
	acceleration = VehicleOldModels_InputAcceleration(
		context, material, normal_acceleration, blended_acceleration,
		steer_slowdown);

	brake = VehicleOldModels_BrakeForce(
		context, material, local_speed->z, *sliding, 1, &any_sliding);
	*brake_force = brake;
	net = x87_sub(acceleration, brake);
	net = VehicleOldModels_LimitLongitudinal(
		context, material, local_speed->z, net);

	force.z = x87_mul(net, longitudinal_force_factor);
	CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);
	if (water_contact == 0) {
		/* 0x007FCF57..0x007FCFE9: clip |force| to +0x1f4 by sign bit. */
		float clipped = force.z;
		GmVec3 torque = { 0.0f, 0.0f, 0.0f };

		if (F(tuning->m5_longitudinal_torque_limit)
				< F(oldmodels_abs(clipped)))
			clipped = x87_mul(
				tuning->m5_longitudinal_torque_limit,
				oldmodels_sign_bits(clipped));
		torque.x = x87_mul(tuning->longitudinal_torque_scale, -clipped);
		CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
	}
	VehicleOldModels_AddVerticalForce(context, existing_force);
	vehicle->turbo_active = any_sliding;
}
