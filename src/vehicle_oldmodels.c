/*
 * Regions shared by CSceneVehicleCar::ComputeForcesModel3/4/5. Ported, no
 * oracle trace yet (see vehicle_oldmodels.h).
 */
#include "tmnf_hd.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "tmnf_fp.h"
#include "vehicle_compute.h"
#include "vehicle_oldmodels.h"
#include "vehicle_oldmodels_common.h"

/* Adapter consumed by the composed 0x007C69E0 top caller. */
TMNF_HD void VehicleOldModels_ComputeForces(
	TMNFVehicleComputeContext *context, float dt,
	const GmVec3 *existing_force, float slope_adherence,
	float slope_secondary, const GmVec3 *linear_speed,
	const GmVec3 *angular_speed, float steering_angle,
	int has_ground_material,
	const CSceneVehicleMaterialBlendableVals *ground_material,
	int *air_control_reset, float *effect_curve_position)
{
	CSceneVehicleCarOldModelsContext *oldmodels = context->oldmodels;
	void (*compute)(
		CSceneVehicleCarOldModelsContext *, float, const GmVec3 *, float,
		float, const GmVec3 *, const GmVec3 *, float, int,
		const CSceneVehicleMaterialBlendableVals *, int *, float *);

	if (oldmodels == NULL || oldmodels->vehicle != context->vehicle
		|| oldmodels->contact != context->contact
		|| oldmodels->aux != context->aux)
		tmnf_abort();
	switch (context->contact->tuning->friction_model) {
	case 0:
	case 1:
	case 2:
		compute = CSceneVehicleCar_ComputeForcesModel3;
		break;
	case 3:
		compute = CSceneVehicleCar_ComputeForcesModel4;
		break;
	case 4:
		compute = CSceneVehicleCar_ComputeForcesModel5;
		break;
	default:
		/* Reflection enum Steer01..Steer06 has no other value. */
		tmnf_abort();
	}
	compute(oldmodels, dt, existing_force, slope_adherence, slope_secondary,
		linear_speed, angular_speed, steering_angle, has_ground_material,
		ground_material, air_control_reset, effect_curve_position);
}

TMNF_HD const TMNFVehicleGroundMaterial *VehicleOldModels_WheelMaterial(
	const CSceneVehicleCarOldModelsContext *context,
	const CSceneVehicleCarWheel *wheel)
{
	uint16_t material_id = (uint16_t)wheel->real_time.contact_material_id;
	uint32_t material_index;

	if (material_id >= context->contact->ground_material_index_count)
		tmnf_abort();
	material_index = context->contact->ground_material_indices[material_id];
	if (material_index >= context->contact->ground_material_count)
		tmnf_abort();
	return context->contact->ground_materials[material_index];
}

/*
 * Model 3 0x007FA7F2..0x007FAC19, Model 5 0x007FC1C2..0x007FC5D2. Both
 * schedules are identical (tools/x87trace.py on each slice).
 */
TMNF_HD void VehicleOldModels_WheelLateral(
	CSceneVehicleCarOldModelsContext *context, CSceneVehicleCarWheel *wheel,
	float lateral_force_factor, const GmVec3 *local_speed,
	float steering_angle, int *sliding)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	const CSceneVehicleCarOldModelsTuning *tuning = context->tuning;
	const TMNFVehicleGroundMaterial *wheel_material =
		VehicleOldModels_WheelMaterial(context, wheel);
	const GmVec3 *normal = &wheel->real_time.field90;
	GmVec3 axis;
	GmVec3 force;
	GmVec3 torque;
	float sliding_scale;
	float maximum;
	float length_squared;
	float velocity_on_axis;
	float requested;
	float rollover_coef;
	float rollover;

	/* 0x007FA855: sliding wheels use +0xb0, others 1.0. */
	sliding_scale = wheel->real_time.is_sliding == 0
		? 1.0f : tuning->sliding_lateral_limit_scale;
	/* 0x007FA894..0x007FA8A7: ((mat[3] * factor) * MaxSide) * scale. */
	maximum = x87_mul(
		x87_mul(
			x87_mul(wheel_material->values[3], lateral_force_factor),
			CSceneVehicleCarTuning_GetMaxSideFrictionFromSpeed(
				context->curves, local_speed->z)),
		sliding_scale);

	/* 0x007FA8AB..0x007FA8E1: (0,1,0) x normal with the zero products kept. */
	axis.x = x87_sub(normal->y, x87_mul(normal->z, 0.0f));
	axis.y = x87_sub(x87_mul(normal->z, 0.0f), normal->x);
	axis.z = x87_sub(x87_mul(normal->x, 0.0f), x87_mul(0.0f, normal->y));
	/* 0x007FA8E5..0x007FA901: (y*y + x*x) + z*z. */
	length_squared = x87_add(
		x87_add(x87_mul(axis.y, axis.y), x87_mul(axis.x, axis.x)),
		x87_mul(axis.z, axis.z));
	if (F(length_squared) <= F(OLDMODELS_LENGTH_EPSILON)) {
		axis = (GmVec3){ 1.0f, 0.0f, 0.0f };
	} else {
		float inverse = oldmodels_div(1.0f, x87_sqrt(length_squared));

		axis.x = x87_mul(inverse, axis.x);
		axis.y = x87_mul(inverse, axis.y);
		axis.z = x87_mul(inverse, axis.z);
	}
	if (wheel->steerable != 0) {
		/* 0x007FA979..0x007FAA29: rotate about y by the steering angle.
		 * The sine term is multiplied by the double 0.0 at 0x00B2C178
		 * for x and y and added unscaled to z. */
		float cosine = oldmodels_cos(steering_angle);
		float negative_sine = -oldmodels_sin(steering_angle);
		float sine_zero = x87_mul_double(negative_sine, 0.0);
		GmVec3 rotated;

		rotated.x = x87_mul(cosine, axis.x);
		rotated.y = x87_mul(cosine, axis.y);
		rotated.z = x87_mul(cosine, axis.z);
		axis.x = x87_add(rotated.x, sine_zero);
		axis.y = x87_add(sine_zero, rotated.y);
		axis.z = x87_add(negative_sine, rotated.z);
	}
	/* 0x007FAA50..0x007FAA6E: (vx*ax + vy*ay) + vz*az. */
	velocity_on_axis = x87_add(
		x87_add(
			x87_mul(local_speed->x, axis.x),
			x87_mul(local_speed->y, axis.y)),
		x87_mul(local_speed->z, axis.z));
	/* 0x007FAA7C..0x007FAA8C: dot * (-scale * 0.5). */
	requested = x87_mul(
		velocity_on_axis,
		x87_mul_double(-tuning->lateral_force_scale, 0.5));
	if (F(oldmodels_abs(requested)) <= F(maximum)) {
		wheel->real_time.is_sliding = 0;
	} else {
		float blend = tuning->lateral_overflow_blend;
		/* 0x007FAACD..0x007FAAE7: FCOM against zero, negate unless
		 * requested > 0. */
		float signed_limit = 0.0 < F(requested) ? maximum : -maximum;

		wheel->real_time.is_sliding = 1;
		/* 0x007FAAF8..0x007FAB0F: (req*b) + ((1-b)*limit). */
		requested = x87_add(
			x87_mul(requested, blend),
			x87_mul(x87_sub(1.0f, blend), signed_limit));
	}
	if (wheel->real_time.is_sliding != 0)
		*sliding = 1;

	/* 0x007FAB3F..0x007FAB62. */
	force.x = x87_mul(axis.x, requested);
	force.y = x87_mul(axis.y, requested);
	force.z = x87_mul(requested, axis.z);
	CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);

	/* 0x007FAB84..0x007FABD9: the angle coefficient is stored as a double
	 * (exact widening) before multiplying. */
	rollover_coef = CSceneVehicleCarTuning_GetRolloverLateralCoefFromAngle(
		context->curves, oldmodels_abs(axis.y));
	rollover = x87_mul(
		x87_mul(
			CSceneVehicleCarTuning_GetRolloverLateralFromSpeed(
				context->curves, local_speed->z),
			lateral_force_factor),
		rollover_coef);
	rollover = -rollover;
	/* 0x007FABDD..0x007FAC15: (0, rollover, 0) x force with the zero
	 * products kept. */
	torque.x = x87_sub(x87_mul(rollover, force.z), x87_mul(force.y, 0.0f));
	torque.y = x87_sub(x87_mul(force.x, 0.0f), x87_mul(0.0f, force.z));
	torque.z = x87_sub(x87_mul(force.y, 0.0f), x87_mul(force.x, rollover));
	CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
}

/*
 * Model 3 0x007FAC96..0x007FACEA, Model 4 0x007FB685..0x007FB6D4, Model 5
 * 0x007FC631..0x007FC67C. Skipped while car +0x60c is set. Compares against
 * the float 0.1 promoted to double at 0x00B362C0.
 */
TMNF_HD void VehicleOldModels_UpdateReverseLatch(
	CSceneVehicleCarOldModelsContext *context, float speed_length)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	CSceneVehicleCarModel6State *shared = context->shared;

	if (vehicle->flag_60c != 0)
		return;
	if (F(shared->reverse_speed_threshold) <= F(speed_length)) {
		if (OLDMODELS_POINT_ONE < F(vehicle->input_gas))
			shared->reverse_mode = 0;
	} else if (F(vehicle->input_brake) <= OLDMODELS_POINT_ONE) {
		shared->reverse_mode = 0;
	} else {
		shared->reverse_mode = 1;
	}
	/* Car +0x5c4 is one field in the game; the port keeps a copy in each
	 * component that reads it (same mirroring as vehicle_model6.c). */
	vehicle->engine.reverse = shared->reverse_mode;
	context->contact->friction_input_selector = shared->reverse_mode;
}

/*
 * Model 3 0x007FB14D..0x007FB20E, Model 4 0x007FBC08..0x007FBC9F, Model 5
 * 0x007FCBB5..0x007FCC60.
 *
 *   (((mat.braking * reverse_bias) * brake + mat.braking * gas) + turbo)
 *       * (curve - steer_slowdown)                      Models 3, 4
 *   ((mat.braking * gas + (mat.braking * reverse_bias) * brake) * blend
 *       + curve * turbo) - steer_slowdown               Model 5
 *
 * Model 5 passes its traction blend in `accel_blend` and the others pass
 * NAN, which selects the first form. When car +0x60c is set the result is
 * curve * turbo_factor, or curve * 0 without a turbo.
 */
TMNF_HD float VehicleOldModels_InputAcceleration(
	CSceneVehicleCarOldModelsContext *context,
	const CSceneVehicleMaterialBlendableVals *material,
	float accel_curve_value, float accel_blend, float steer_slowdown)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	CSceneVehicleCarAuxContext *aux = context->aux;
	float reverse_bias = context->shared->reverse_mode == 0 ? 0.0f : -1.0f;
	float turbo = aux->turbo_type == TMNF_TURBO_NONE
		? 0.0f : aux->turbo_factor;
	float result;

	if (isnan(accel_blend)) {
		float inputs = x87_add(
			x87_add(
				x87_mul(
					x87_mul(material->braking, reverse_bias),
					vehicle->input_brake),
				x87_mul(material->braking, vehicle->input_gas)),
			turbo);

		result = x87_mul(
			inputs, x87_sub(accel_curve_value, steer_slowdown));
	} else {
		float inputs = x87_add(
			x87_mul(material->braking, vehicle->input_gas),
			x87_mul(
				x87_mul(material->braking, reverse_bias),
				vehicle->input_brake));

		result = x87_sub(
			x87_add(
				x87_mul(turbo, accel_curve_value),
				x87_mul(inputs, accel_blend)),
			steer_slowdown);
	}
	if (vehicle->flag_60c != 0) {
		result = aux->turbo_type == TMNF_TURBO_NONE
			? x87_mul(accel_curve_value, 0.0f)
			: x87_mul(accel_curve_value, aux->turbo_factor);
	}
	return result;
}

TMNF_HD static float brake_limit(
	CSceneVehicleCarOldModelsContext *context,
	const CSceneVehicleMaterialBlendableVals *material, int sliding)
{
	const CSceneVehicleCarOldModelsTuning *tuning = context->tuning;

	return x87_mul(
		material->steering,
		sliding != 0 ? tuning->brake_limit_sliding : tuning->brake_limit);
}

TMNF_HD static void mark_all_wheels_sliding(
	CSceneVehicleCarOldModelsContext *context, int *any_sliding)
{
	CSceneVehicleCar *vehicle = context->vehicle;

	if (vehicle->wheel_count == 0)
		return;
	if (any_sliding != NULL)
		*any_sliding = 1;
	for (uint32_t index = 0; index < vehicle->wheel_count; ++index)
		vehicle->wheels[index].real_time.is_sliding = 1;
}

/*
 * Model 3 0x007FB20E..0x007FB44A, Model 4 0x007FBCA6..0x007FBE3E, Model 5
 * 0x007FCC60..0x007FCE6B. Forward: ((scale*v + base) * brake) clipped to
 * mat.steering * limit. Reverse (only with car +0x60c): ((base - scale*v)
 * * gas) clipped the same way, then negated. With +0x60c and |v| < 1 the
 * result is scaled by |v|.
 */
TMNF_HD float VehicleOldModels_BrakeForce(
	CSceneVehicleCarOldModelsContext *context,
	const CSceneVehicleMaterialBlendableVals *material,
	float forward_speed, int sliding, int mark_wheels, int *any_sliding)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	const CSceneVehicleCarOldModelsTuning *tuning = context->tuning;
	float brake = 0.0f;

	if (0.0 < F(forward_speed)) {
		float limit = brake_limit(context, material, sliding);

		brake = x87_mul(
			x87_add(
				x87_mul(tuning->brake_speed_scale, forward_speed),
				tuning->brake_base),
			vehicle->input_brake);
		if (F(limit) < F(brake)) {
			brake = limit;
			if (mark_wheels)
				mark_all_wheels_sliding(context, any_sliding);
			else if (any_sliding != NULL)
				*any_sliding = 1;
		}
	}
	if (F(forward_speed) < 0.0) {
		float limit;

		if (vehicle->flag_60c == 0)
			return brake;
		limit = brake_limit(context, material, sliding);
		brake = x87_mul(
			x87_sub(
				tuning->brake_base,
				x87_mul(tuning->brake_speed_scale, forward_speed)),
			vehicle->input_gas);
		if (F(limit) < F(brake)) {
			brake = limit;
			if (mark_wheels)
				mark_all_wheels_sliding(context, any_sliding);
			else if (any_sliding != NULL)
				*any_sliding = 1;
		}
		brake = -brake;
	}
	if (vehicle->flag_60c != 0) {
		float absolute = oldmodels_abs(forward_speed);

		if (F(absolute) < 1.0)
			brake = x87_mul(absolute, brake);
	}
	return brake;
}

/*
 * Model 3 0x007FB45C..0x007FB504, Model 4 0x007FC00A..0x007FC08C, Model 5
 * 0x007FCE7A..0x007FCF08. Both limits are rounded products; the reverse one
 * is negated at the compare.
 */
TMNF_HD float VehicleOldModels_LimitLongitudinal(
	CSceneVehicleCarOldModelsContext *context,
	const CSceneVehicleMaterialBlendableVals *material,
	float forward_speed, float net)
{
	const CSceneVehicleCarOldModelsTuning *tuning = context->tuning;
	float reverse_limit = x87_mul(
		tuning->reverse_speed_limit_scale, material->acceleration);
	float forward_limit = x87_mul(
		tuning->forward_speed_limit_scale, material->acceleration);

	if (F(forward_limit) < F(forward_speed))
		net = -tuning->speed_limit_force;
	if (F(forward_speed) < F(-reverse_limit))
		net = tuning->speed_limit_force;
	return net;
}

/* Model 3 0x007FB57A..0x007FB5D3, Model 5 0x007FCFEE..0x007FD041. */
TMNF_HD void VehicleOldModels_AddVerticalForce(
	CSceneVehicleCarOldModelsContext *context, const GmVec3 *existing_force)
{
	const CSceneVehicleCarOldModelsTuning *tuning = context->tuning;
	GmVec3 force = { 0.0f, 0.0f, 0.0f };

	force.z = oldmodels_div(
		x87_mul(-tuning->vertical_force_scale, existing_force->z),
		tuning->vertical_force_divisor);
	CSceneVehicleCar_AddVehicleCentralForce(context->vehicle, &force);
}
