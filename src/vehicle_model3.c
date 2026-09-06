/*
 * 0x007FA770 CSceneVehicleCar::ComputeForcesModel3 (994 instructions, 521
 * x87). Selected for CSceneVehicleCarTuning+0x354 values 0, 1 and 2; the
 * lateral and drive sections run only for value 1 (0x007FA824..0x007FA846,
 * 0x007FAC58).
 *
 * PORTED, NO ORACLE TRACE YET.
 */
#include "tmnf_hd.h"
#include <math.h>
#include <stdlib.h>

#include "tmnf_fp.h"
#include "vehicle_oldmodels.h"
#include "vehicle_oldmodels_common.h"

/* 0x007FAD10..0x007FB021: per-wheel drive torque. */
TMNF_HD static void model3_wheel_torque(
	CSceneVehicleCarOldModelsContext *context, CSceneVehicleCarWheel *wheel,
	float speed_length, const GmVec3 *local_speed,
	const GmVec3 *local_angular_speed,
	const CSceneVehicleMaterialBlendableVals *material)
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

	/* 0x007FAD97..0x007FADE5: sin(((len / limit) * pi) * 0.5) up to the
	 * limit, 1.0 beyond it. */
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
	/* 0x007FAE25..0x007FAE59: ((oy*0 + ox) + 0*oz) * (-scale * 0.5). */
	requested = x87_mul(
		x87_add(
			x87_add(x87_mul(offset_speed.y, 0.0f), offset_speed.x),
			x87_mul(0.0f, offset_speed.z)),
		x87_mul_double(-tuning->lateral_force_scale, 0.5));
	if (F(maximum) < F(oldmodels_abs(requested))) {
		float blend = tuning->wheel_overflow_blend;
		float blended = x87_add(
			x87_mul(x87_sub(1.0f, blend), maximum),
			x87_mul(blend, oldmodels_abs(requested)));

		requested = 0.0 < F(requested) ? blended : -blended;
	}
	requested = x87_mul(tuning->wheel_torque_scale, requested);
	if (wheel->steerable != 0) {
		float direction = shared->reverse_mode == 0 ? 1.0f : -1.0f;
		float slide_scale = wheel->real_time.is_sliding == 0
			? 1.0f : tuning->sliding_steer_torque_scale;
		float drive_torque =
			CSceneVehicleCarTuning_GetSteerDriveTorqueFromSpeed(
				context->curves, local_speed->z);

		/* 0x007FAF7B..0x007FAF96: requested is widened to double
		 * before the subtraction; the product chain is
		 * ((dir*mod)*steer)*curve)*slide. */
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
	/* 0x007FAFA4..0x007FB003: (half, 0, 0) x (req, req*0, req*0). */
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

TMNF_HD void CSceneVehicleCar_ComputeForcesModel3(
	CSceneVehicleCarOldModelsContext *context, float dt,
	const GmVec3 *existing_force, float lateral_force_factor,
	float longitudinal_force_factor, const GmVec3 *local_speed,
	const GmVec3 *local_angular_speed, float steering_angle,
	int grounded, const CSceneVehicleMaterialBlendableVals *material,
	int *sliding, float *brake_force)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	const CSceneVehicleCarOldModelsTuning *tuning = context->tuning;
	int friction_model = context->contact->tuning->friction_model;
	float speed_length;
	float lateral_limit;
	float lateral_request;
	float steer_slowdown;
	float acceleration;
	float brake;
	float net;
	GmVec3 force = { 0.0f, 0.0f, 0.0f };
	GmVec3 torque = { 0.0f, 0.0f, 0.0f };

	(void)dt;
	for (uint32_t index = 0; index < vehicle->wheel_count; ++index) {
		CSceneVehicleCarWheel *wheel = &vehicle->wheels[index];

		CSceneVehicleCar_WheelAddForceToVehicle(vehicle, wheel);
		if (wheel->real_time.has_ground_contact == 0)
			continue;
		/* 0x007FA801: FCOMP of 0.0 against the scale; NaN skips. */
		if (!(0.0 < F(tuning->lateral_force_scale)))
			continue;
		if (friction_model != 1)
			continue;
		VehicleOldModels_WheelLateral(
			context, wheel, lateral_force_factor, local_speed,
			steering_angle, sliding);
	}
	if (grounded == 0)
		return;
	if (friction_model != 1)
		return;

	/* 0x007FAC64..0x007FAC8F: (vy*vy + vx*vx) + vz*vz. */
	speed_length = x87_sqrt(x87_add(
		x87_add(
			x87_mul(local_speed->y, local_speed->y),
			x87_mul(local_speed->x, local_speed->x)),
		x87_mul(local_speed->z, local_speed->z)));
	VehicleOldModels_UpdateReverseLatch(context, speed_length);

	for (uint32_t index = 0; index < vehicle->wheel_count; ++index)
		model3_wheel_torque(
			context, &vehicle->wheels[index], speed_length,
			local_speed, local_angular_speed, material);

	/* 0x007FB027..0x007FB146. */
	acceleration = CSceneVehicleCarTuning_GetAccelFromSpeed(
		context->curves, local_speed->z);
	lateral_limit = x87_mul(
		CSceneVehicleCarTuning_GetMaxSideFrictionFromSpeed(
			context->curves, local_speed->z),
		material->lateral_grip);
	lateral_request = oldmodels_abs(x87_mul(
		x87_mul_double(tuning->lateral_force_scale, 0.5),
		local_speed->x));
	if (F(lateral_limit) < F(lateral_request))
		lateral_request = lateral_limit;
	steer_slowdown = x87_mul(
		CSceneVehicleCarTuning_GetSteerSlowDownFromSpeed(
			context->curves, local_speed->z),
		x87_mul(
			x87_mul(tuning->steer_slowdown_scale, lateral_request),
			oldmodels_abs(context->aux->steering_value)));
	acceleration = VehicleOldModels_InputAcceleration(
		context, material, acceleration, NAN, steer_slowdown);

	brake = VehicleOldModels_BrakeForce(
		context, material, local_speed->z, *sliding, 1, NULL);
	*brake_force = brake;
	net = x87_sub(acceleration, brake);
	net = VehicleOldModels_LimitLongitudinal(
		context, material, local_speed->z, net);

	force.z = x87_mul(net, longitudinal_force_factor);
	CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);
	torque.x = x87_mul(tuning->longitudinal_torque_scale, -force.z);
	CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
	VehicleOldModels_AddVerticalForce(context, existing_force);
}
