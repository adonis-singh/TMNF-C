#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tmnf_fp.h"
#include "vehicle_model6_wheels.h"

#define MODEL6_NORMALIZE_EPSILON 0x1.b7cdfcp-34f
#define MODEL6_PI 0x1.921fb60000000p+1
#define MODEL6_BRAKE_THRESHOLD 0.1
#define MODEL6_MATERIAL6_SPEED 6.0

TMNF_HD static float vec_length_squared_yxz(const GmVec3 *value)
{
	float y_squared = x87_mul(value->y, value->y);
	float x_squared = x87_mul(value->x, value->x);
	float z_squared = x87_mul(value->z, value->z);

	return x87_add(x87_add(y_squared, x_squared), z_squared);
}

TMNF_HD static double load_uint32_x87(uint32_t value)
{
	int32_t signed_value;

	memcpy(&signed_value, &value, sizeof(signed_value));
	if (signed_value < 0) {
		return F(x87_r24(
			(double)signed_value + 4294967296.0));
	}
	return (double)signed_value;
}

TMNF_HD static const TMNFVehicleGroundMaterial *wheel_material(
	const VehicleModel6Context *context,
	const CSceneVehicleCarWheel *wheel)
{
	const TMNFVehicleContactContext *contact = context->contact;
	uint16_t material_id = (uint16_t)
		wheel->real_time.contact_material_id;
	uint32_t material_index =
		contact->ground_material_indices[material_id];

	return contact->ground_materials[material_index];
}

TMNF_HD static GmVec3 wheel_lateral_axis(
	const CSceneVehicleCarWheel *wheel, float steer_angle)
{
	const GmVec3 *normal = &wheel->real_time.field90;
	float normal_y_zero = x87_mul(normal->y, 0.0f);
	GmVec3 axis;
	float length_squared;

	axis.x = x87_sub(normal->z, normal_y_zero);
	axis.y = x87_sub(x87_mul(normal->y, 0.0f), normal->x);
	axis.z = x87_sub(
		x87_mul(normal->x, 0.0f),
		x87_mul(normal->z, 0.0f));
	length_squared = vec_length_squared_yxz(&axis);
	if (F(MODEL6_NORMALIZE_EPSILON) < F(length_squared)) {
		float length = x87_sqrt(length_squared);
		float inverse_length = x87_rcp(length);

		axis.x = x87_mul(inverse_length, axis.x);
		axis.y = x87_mul(inverse_length, axis.y);
		axis.z = x87_mul(inverse_length, axis.z);
	} else {
		axis.x = 1.0f;
		axis.y = 0.0f;
		axis.z = 0.0f;
	}

	if (wheel->steerable != 0) {
		float cosine = x87_cos(steer_angle);
		float negative_sine = -x87_sin(steer_angle);
		GmVec3 scaled;
		GmVec3 sine_row;

		scaled.x = x87_mul(cosine, axis.x);
		scaled.y = x87_mul(cosine, axis.y);
		scaled.z = x87_mul(cosine, axis.z);
		sine_row.x = x87_mul(negative_sine, 0.0f);
		sine_row.y = sine_row.x;
		sine_row.z = negative_sine;
		axis.x = x87_add(scaled.x, sine_row.x);
		axis.y = x87_add(sine_row.y, scaled.y);
		axis.z = x87_add(sine_row.z, scaled.z);
	}
	return axis;
}

TMNF_HD static GmVec3 wheel_contact_radius(
	const VehicleModel6Context *context,
	const CSceneVehicleCarWheel *wheel)
{
	const GmVec3 *center = context->body_center_local;
	GmVec3 radius;

	radius.x = x87_sub(wheel->real_time.field54.x, center->x);
	radius.y = x87_sub(wheel->real_time.field54.y, center->y);
	radius.z = x87_sub(wheel->real_time.field54.z, center->z);
	return radius;
}

TMNF_HD static float lateral_speed_on_axis(
	const GmVec3 *speed, const GmVec3 *axis)
{
	float dot_x = x87_mul(speed->x, axis->x);
	float dot_y = x87_mul(speed->y, axis->y);
	float dot_z = x87_mul(speed->z, axis->z);

	return x87_add(x87_add(dot_x, dot_y), dot_z);
}

TMNF_HD static void add_roll_torque(
	VehicleModel6Context *context,
	const GmVec3 *radius)
{
	const VehicleModel6TuningScalars *scalars = context->scalars;
	float force_scale = -scalars->s228;
	GmVec3 force;
	GmVec3 torque;
	float force_length;

	force.x = x87_mul(force_scale, context->state->normalized_force.x);
	force.y = x87_mul(context->state->normalized_force.y, force_scale);
	force.z = x87_mul(force_scale, context->state->normalized_force.z);
	force_length = x87_sqrt(vec_length_squared_yxz(&force));
	if (F(force_length) < F(scalars->s234)) {
		force.x = 0.0f;
		force.y = 0.0f;
		force.z = 0.0f;
	}

	torque.x = x87_mul(
		x87_sub(
			x87_mul(radius->y, force.z),
			x87_mul(radius->z, force.y)),
		-1.0f);
	torque.y = x87_mul(
		x87_sub(
			x87_mul(force.x, radius->z),
			x87_mul(radius->x, force.z)),
		-1.0f);
	torque.z = x87_mul(
		x87_sub(
			x87_mul(force.y, radius->x),
			x87_mul(radius->y, force.x)),
		-1.0f);
	torque.x = x87_mul(scalars->s238, torque.x);
	torque.y = 0.0f;
	torque.z = x87_mul(scalars->s23c, torque.z);
	CSceneVehicleCar_AddVehicleTorque(context->car, &torque);
}

TMNF_HD static float burnout_side_blend(const VehicleModel6Context *context)
{
	const VehicleModel6PersistentState *state = context->state;
	const VehicleModel6TuningScalars *scalars = context->scalars;
	uint32_t elapsed = context->tick - state->burnout_start_tick;
	uint32_t denominator_ticks = scalars->ticks298 * 2u;
	float phase_numerator = x87_r24(
		load_uint32_x87(elapsed) * MODEL6_PI);
	float phase = x87_r24(
		F(phase_numerator) / load_uint32_x87(denominator_ticks));
	float cosine = x87_cos(phase);
	float amplitude = x87_sub(scalars->s2a0, 1.0f);

	return x87_add(1.0f, x87_mul(amplitude, cosine));
}

TMNF_HD static float compute_side_force(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	CSceneVehicleCarWheel *wheel,
	const TMNFVehicleGroundMaterial *material, float lateral_speed)
{
	const VehicleModel6TuningScalars *scalars = context->scalars;
	float burnout_blend = context->state->burnout_state == 1
		? burnout_side_blend(context) : 1.0f;
	float modulation =
		CSceneVehicleCarTuning_M6GetModulationFromDamperAbsorbVal(
			context->curves, wheel->real_time.damper_absorb);
	float slide_multiplier =
		wheel->real_time.is_sliding != 0 ? scalars->s0b0 : 1.0f;
	float brake_multiplier =
		wheel->real_time.is_sliding != 0
			&& MODEL6_BRAKE_THRESHOLD
				< F(context->car->input_brake)
		? scalars->s244 : 1.0f;
	float maximum =
		CSceneVehicleCarTuning_GetMaxSideFrictionFromSpeed(
			context->curves, inputs->local_speed->z);
	float limit = x87_mul(material->values[3], inputs->slope_adherence);
	float requested = -scalars->s0a4;

	limit = x87_mul(limit, maximum);
	limit = x87_mul(limit, slide_multiplier);
	limit = x87_mul(limit, brake_multiplier);
	limit = x87_mul(limit, modulation);
	requested = x87_r24(F(requested) * 0.5);
	requested = x87_mul(requested, lateral_speed);
	requested = x87_mul(requested, burnout_blend);

	if (!(F(limit) < F(fabsf(requested)))) {
		wheel->real_time.is_sliding = 0;
	} else {
		float signed_limit =
			F(requested) <= 0.0 ? -limit : limit;
		float retained = scalars->s0b4;
		float limited = x87_mul(
			x87_sub(1.0f, retained), signed_limit);
		float original = x87_mul(retained, requested);

		wheel->real_time.is_sliding = 1;
		requested = x87_add(limited, original);
	}
	if (wheel->real_time.is_sliding != 0)
		*inputs->sliding_out = 1;
	return requested;
}

TMNF_HD static float material6_longitudinal_force(
	const VehicleModel6TuningScalars *scalars, float gas,
	float absolute_x, float direction_factor, float speed_denominator,
	int front)
{
	float force = x87_mul(scalars->s348, gas);

	if (front)
		force = x87_r24(F(force) * 1.5);
	force = x87_mul(force, absolute_x);
	force = x87_mul(force, direction_factor);
	force = x87_mul(force, scalars->s344);
	return x87_div(force, speed_denominator);
}

TMNF_HD static void add_material6_stabilization(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	const GmVec3 *side_force)
{
	CSceneVehicleCar *car = context->car;
	const VehicleModel6TuningScalars *scalars = context->scalars;
	GmVec3 front = { 0.0f, 0.0f, 0.0f };
	GmVec3 rear = { 0.0f, 0.0f, 0.0f };
	GmVec3 speed = *inputs->local_speed;
	float length_squared = vec_length_squared_yxz(&speed);
	float normalized_x = speed.x;

	if (F(MODEL6_NORMALIZE_EPSILON) < F(length_squared)) {
		float length = x87_sqrt(length_squared);
		float inverse_length = x87_rcp(length);

		normalized_x = x87_mul(inverse_length, normalized_x);
	}
	if (MODEL6_BRAKE_THRESHOLD < F(car->input_brake)) {
		GmVec3 braking_force;

		braking_force.x = x87_r24(F(speed.x) * -0.1);
		braking_force.y = x87_r24(F(speed.y) * -0.1);
		braking_force.z = x87_r24(F(speed.z) * -0.1);
		CSceneVehicleCar_AddVehicleCentralForce(car, &braking_force);
	}

	if (context->state->special_physics == 0
		&& MODEL6_BRAKE_THRESHOLD < F(car->input_gas)
		&& F(car->input_brake) <= MODEL6_BRAKE_THRESHOLD) {
		float absolute_x = fabsf(normalized_x);
		float direction_factor = x87_r24(F(absolute_x) * 20.0);
		float speed_base;
		float speed_denominator;
		float front_numerator;
		float front_denominator;
		float rear_numerator;
		float rear_denominator;

		direction_factor = x87_r24(F(direction_factor) + 1.0);
		speed_base = x87_r24(F(fabsf(speed.z)) + 1.0);
		speed_denominator = x87_mul(speed_base, speed_base);
		front.z = material6_longitudinal_force(
			scalars, car->input_gas, absolute_x,
			direction_factor, speed_denominator, 1);
		front_numerator = x87_mul(scalars->s33c, normalized_x);
		front_denominator = x87_add(
			x87_mul(scalars->s340, car->input_gas), 1.0f);
		front.x = x87_div(front_numerator, front_denominator);
		rear.z = material6_longitudinal_force(
			scalars, car->input_gas, absolute_x,
			direction_factor, speed_denominator, 0);
		rear_numerator = x87_mul(
			scalars->s33c, x87_mul(normalized_x, -1.0f));
		rear_denominator = x87_add(
			x87_mul(scalars->s340, car->input_gas), 1.0f);
		rear.x = x87_div(rear_numerator, rear_denominator);
	}

	for (uint32_t index = 0; index < car->wheel_count; ++index) {
		CSceneVehicleCarWheel *wheel = &car->wheels[index];

		if ((index == 0 || index == 1)
			&& F(car->input_brake) <= MODEL6_BRAKE_THRESHOLD
			&& wheel->real_time.is_sliding != 0) {
			CSceneVehicleCar_AddVehicleForce(
				car, &front, &wheel->offset_from_vehicle);
		}
		if ((index == 2 || index == 3)
			&& F(car->input_brake) <= MODEL6_BRAKE_THRESHOLD
			&& wheel->real_time.is_sliding != 0) {
			CSceneVehicleCar_AddVehicleForce(
				car, &rear, &wheel->offset_from_vehicle);
		}
	}
	CSceneVehicleCar_AddVehicleCentralForce(car, side_force);
}

TMNF_HD static void update_burnout_transition(
	VehicleModel6Context *context, VehicleModel6Scratch *scratch)
{
	VehicleModel6PersistentState *state = context->state;
	const VehicleModel6TuningScalars *scalars = context->scalars;
	uint32_t tick = context->tick;

	if (state->burnout_state == 1) {
		uint32_t start = state->burnout_start_tick;

		if (tick < start || scalars->ticks298 <= tick - start) {
			state->burnout_end_tick = tick;
			state->burnout_state = 3;
			context->car->drive_mode = 3;
		} else {
			scratch->slip_activity = 1;
		}
	}
	if (state->burnout_state == 3) {
		uint32_t end = state->burnout_end_tick;

		if (tick < end || scalars->ticks2a8 <= tick - end) {
			state->burnout_state = 0;
			state->force_wheel_speed = 0;
			context->car->drive_mode = 0;
			context->car->force_wheel_speed = 0;
		} else {
			for (uint32_t index = 0;
			     index < context->car->wheel_count; ++index) {
				context->car->wheels[index]
					.real_time.is_sliding = 1;
			}
		}
	}
}

/* 0x007C4822..0x007C53B9, normal wheel/contact-force preprocessing. */
TMNF_HD void VehicleModel6_ProcessWheelContacts(
	VehicleModel6Context *context, const VehicleModel6Inputs *inputs,
	VehicleModel6Scratch *scratch)
{
	CSceneVehicleCar *car = context->car;

	scratch->tick = context->tick;
	scratch->wheel_count = car->wheel_count;
	scratch->slip_activity = 0;
	if (context->state->burnout_state == 2)
		return;
	update_burnout_transition(context, scratch);

	for (uint32_t index = 0; index < car->wheel_count; ++index) {
		CSceneVehicleCarWheel *wheel = &car->wheels[index];
		const TMNFVehicleGroundMaterial *material;
		GmVec3 axis;
		GmVec3 radius;
		GmVec3 side_force;
		float lateral_speed;
		float requested;

		CSceneVehicleCar_WheelAddForceToVehicle(car, wheel);
		material = wheel_material(context, wheel);
		if (wheel->real_time.has_ground_contact == 0)
			continue;
		if (!(0.0 <= F(context->scalars->s0a4)))
			continue;

		radius = wheel_contact_radius(context, wheel);
		axis = wheel_lateral_axis(wheel, inputs->steer_angle);
		lateral_speed = lateral_speed_on_axis(
			inputs->local_speed, &axis);
		add_roll_torque(context, &radius);
		if (context->state->burnout_state == 1) {
			GmVec3 burnout_torque = {
				CSceneVehicleCarTuning_M6GetBurnoutRolloverFromSpeed(
					context->curves,
					inputs->local_speed->z),
				0.0f,
				0.0f,
			};

			CSceneVehicleCar_AddVehicleTorque(
				car, &burnout_torque);
		}

		requested = compute_side_force(
			context, inputs, wheel, material, lateral_speed);
		side_force.x = x87_mul(requested, axis.x);
		side_force.y = x87_mul(requested, axis.y);
		side_force.z = x87_mul(axis.z, requested);
		if (scratch->all_wheels_material6 != 0
			&& MODEL6_MATERIAL6_SPEED
				< F(inputs->local_speed->z)) {
			add_material6_stabilization(
				context, inputs, &side_force);
		} else {
			CSceneVehicleCar_AddVehicleCentralForce(
				car, &side_force);
		}
	}
}
