#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "tmnf_fp.h"
#include "vehicle_contact.h"

#define CONTACT_LENGTH_EPSILON 0x1.b7cdfcp-34f
#define FRICTION_SPEED_EPSILON 0x1.4f8b58p-17f
#define CONTACT_ANGLE_RADIANS 0x1.921fb60000000p-1
#define TMNF_PI 0x1.921fb60000000p+1

TMNF_HD static float vec_dot_yxz(const GmVec3 *left, const GmVec3 *right)
{
	return x87_add(
		x87_add(
			x87_mul(left->y, right->y),
			x87_mul(left->x, right->x)),
		x87_mul(left->z, right->z));
}

TMNF_HD static float vec_length_squared_yxz(const GmVec3 *value)
{
	return vec_dot_yxz(value, value);
}

TMNF_HD static float vec_length_squared_xyz(const GmVec3 *value)
{
	return x87_add(
		x87_mul(value->z, value->z),
		x87_add(
			x87_mul(value->x, value->x),
			x87_mul(value->y, value->y)));
}

TMNF_HD static GmIso4 dyna_iso(const CHmsStateDyna *state)
{
	GmIso4 iso;

	memcpy(iso.m, state->rot.m, sizeof(iso.m));
	iso.t[0] = state->pos.x;
	iso.t[1] = state->pos.y;
	iso.t[2] = state->pos.z;
	return iso;
}

TMNF_HD static void set_contact_material_id(
	CSceneVehicleCarWheelRealTimeState *state, uint16_t material_id)
{
	memcpy(&state->contact_material_id, &material_id, sizeof(material_id));
}

TMNF_HD static float slope_curve(float ratio, float lower, float upper)
{
	float numerator = x87_sub(ratio, lower);
	float denominator = x87_sub(upper, lower);
	float normalized = x87_div(numerator, denominator);
	float radians = x87_r24(F(normalized) * TMNF_PI);
	float half_radians = x87_r24(F(radians) * 0.5);
	float cosine = x87_cos(half_radians);

	return x87_sub(1.0f, cosine);
}

/* 0x0093A4A0, UNVALIDATED: returns the timer's current simulation tick. */
TMNF_HD const uint32_t *CMwTimerAdapter_GetTickTime(
	const TMNFVehicleContactTimer *self)
{
	return &self->tick_time;
}

/*
 * 0x007BE390, UNVALIDATED: applies a local impulse at a local point and clamps
 * the resulting rigid-body speeds with the selected vehicle tuning.
 */
TMNF_HD void CSceneVehicleCar_AddVehicleImpulse(
	TMNFVehicleContactContext *context, const GmVec3 *impulse,
	const GmVec3 *point)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	CHmsStateDyna *state = vehicle->dyna_state;
	const CHmsDynaParams *params;
	const TMNFVehicleContactTuning *tuning = context->tuning;
	GmIso4 iso;
	GmVec3 world_impulse;
	GmVec3 world_point;
	GmVec3 world_center;
	GmVec3 lever;
	GmVec3 angular_delta;
	GmVec3 linear_delta;
	GmVec3 new_linear_speed;
	float inverse_mass;
	float old_speed_squared;
	float new_speed_squared;
	float angular_speed_squared;
	float max_angular_squared;

	if (state == NULL)
		return;
	params = vehicle->dyna_params;
	if (params == NULL)
		tmnf_abort();

	iso = dyna_iso(state);
	GmVec3_SetMult_Mat3(&world_impulse, impulse, &state->rot);
	GmVec3_SetMult_Iso4(&world_point, point, &iso);
	inverse_mass = x87_rcp(params->mass);
	linear_delta.x = x87_mul(world_impulse.x, inverse_mass);
	linear_delta.y = x87_mul(world_impulse.y, inverse_mass);
	linear_delta.z = x87_mul(inverse_mass, world_impulse.z);
	new_linear_speed.x = x87_add(state->linVel.x, linear_delta.x);
	new_linear_speed.y = x87_add(state->linVel.y, linear_delta.y);
	new_linear_speed.z = x87_add(state->linVel.z, linear_delta.z);
	old_speed_squared = vec_length_squared_yxz(&state->linVel);
	new_speed_squared = vec_length_squared_yxz(&new_linear_speed);
	if (F(old_speed_squared) < F(new_speed_squared)
		&& F(tuning->max_linear_speed_delta)
			< F(x87_sub(new_speed_squared, old_speed_squared))) {
		new_linear_speed.x = 0.0f;
		new_linear_speed.y = 0.0f;
		new_linear_speed.z = 0.0f;
	}
	state->linVel = new_linear_speed;

	GmVec3_SetMult_Iso4(&world_center, &params->comOffset, &iso);
	lever.x = x87_sub(world_point.x, world_center.x);
	lever.y = x87_sub(world_point.y, world_center.y);
	lever.z = x87_sub(world_point.z, world_center.z);
	angular_delta.x = x87_sub(
		x87_mul(world_impulse.z, lever.y),
		x87_mul(world_impulse.y, lever.z));
	angular_delta.y = x87_sub(
		x87_mul(world_impulse.x, lever.z),
		x87_mul(lever.x, world_impulse.z));
	angular_delta.z = x87_sub(
		x87_mul(lever.x, world_impulse.y),
		x87_mul(world_impulse.x, lever.y));
	GmVec3_Mult_Mat3(&angular_delta, &state->invInertiaWorld);
	angular_delta.x = x87_mul(
		tuning->angular_xz_scale, angular_delta.x);
	angular_delta.y = x87_mul(
		tuning->angular_xz_scale, angular_delta.y);
	angular_delta.z = x87_mul(
		tuning->angular_xz_scale, angular_delta.z);
	angular_delta.y = x87_mul(
		tuning->angular_y_scale, angular_delta.y);
	state->angVel.x = x87_add(state->angVel.x, angular_delta.x);
	state->angVel.y = x87_add(state->angVel.y, angular_delta.y);
	state->angVel.z = x87_add(state->angVel.z, angular_delta.z);

	angular_speed_squared = vec_length_squared_yxz(&state->angVel);
	max_angular_squared = x87_mul(
		tuning->max_angular_speed, tuning->max_angular_speed);
	if (F(max_angular_squared) < F(angular_speed_squared)) {
		float angular_speed = x87_sqrt(angular_speed_squared);
		float ratio = x87_div(tuning->max_angular_speed, angular_speed);

		state->angVel.x = x87_mul(state->angVel.x, ratio);
		state->angVel.y = x87_mul(state->angVel.y, ratio);
		state->angVel.z = x87_mul(ratio, state->angVel.z);
	}

	vehicle->total_impulse_added.x = x87_add(
		impulse->x, vehicle->total_impulse_added.x);
	vehicle->total_impulse_added.y = x87_add(
		impulse->y, vehicle->total_impulse_added.y);
	vehicle->total_impulse_added.z = x87_add(
		impulse->z, vehicle->total_impulse_added.z);
}

TMNF_HD static void apply_wheel_impulse(
	TMNFVehicleContactContext *context,
	const GmVec3 *relative_speed, const GmVec3 *normal,
	const GmVec3 *point, float restitution)
{
	const CHmsDynaParams *params = context->vehicle->dyna_params;
	GmVec3 lever;
	GmVec3 impulse;

	if (params == NULL)
		tmnf_abort();
	lever.x = x87_sub(point->x, params->comOffset.x);
	lever.y = x87_sub(point->y, params->comOffset.y);
	lever.z = x87_sub(point->z, params->comOffset.z);
	SDynaMath_ComputeImpulse(
		params->mass, &params->invInertiaBody, -restitution,
		relative_speed, normal, &lever, &impulse);
	CSceneVehicleCar_AddVehicleImpulse(context, &impulse, point);
}

/*
 * 0x007C11D0, UNVALIDATED: classifies a wheel contact, accumulates its contact
 * state, removes accepted replacement, and applies the collision impulse.
 */
TMNF_HD void CSceneVehicleCar_WheelAbsorbContact(
	TMNFVehicleContactContext *context,
	TMNFVehicleContactWheelState *wheel_state,
	CHmsPhysicalContact *contact)
{
	CSceneVehicleCarWheel *wheel = wheel_state->wheel;
	CSceneVehicleCarWheelRealTimeState *real_time = &wheel->real_time;
	const TMNFVehicleContactTuning *tuning = context->tuning;
	float ground_threshold = x87_sin((float)CONTACT_ANGLE_RADIANS);
	int ground_contact =
		F(fabs(F(contact->normal.x))) < F(ground_threshold);

	real_time->has_ground_contact = ground_contact;
	if (!ground_contact) {
		context->side_contact = 1;
		wheel->contact_relative_local_distance = contact->position;
		wheel->field15c = 1;
	}
	if (ground_contact) {
		real_time->ground_contact_count++;
		real_time->field90.x = x87_add(
			real_time->field90.x, contact->normal.x);
		real_time->field90.y = x87_add(
			contact->normal.y, real_time->field90.y);
		real_time->field90.z = x87_add(
			contact->normal.z, real_time->field90.z);
		set_contact_material_id(
			real_time, contact->other_surface_material);
	}
	contact->accepted = 0;

	if (contact->other_body != NULL) {
		const GmIso4 *other_iso;
		GmVec3 other_up;

		if (context->resolve_body_iso == NULL
			|| context->vehicle_contact_rotation == NULL) {
			tmnf_abort();
		}
		other_iso = context->resolve_body_iso(
			context->resolve_body_iso_user, contact->other_body);
		if (other_iso == NULL)
			tmnf_abort();
		other_up.x = other_iso->m[2];
		other_up.y = other_iso->m[5];
		other_up.z = other_iso->m[8];
		real_time->relative_rotz_axis = other_up;
		wheel_state->contact_body = contact->other_body;
		GmVec3_MultTranspose(
			&real_time->relative_rotz_axis,
			context->vehicle_contact_rotation);
	}
	real_time->field54 = contact->position;

	if (tuning->wheel_contact_model == 2) {
		float replacement_y = x87_add(
			x87_add(
				x87_mul(contact->replacement.x, 0.0f),
				contact->replacement.y),
			x87_mul(0.0f, contact->replacement.z));
		int replacement_accepted;

		if (!(0.0 < F(replacement_y))) {
			replacement_accepted = 1;
		} else {
			float contact_replacement;

			replacement_accepted = 0;
			if (!(F(tuning->damper_max)
			    < -F(FRICTION_SPEED_EPSILON))) {
				float damper_replacement = x87_sub(
					real_time->damper_absorb,
					tuning->damper_max);

				if (!(F(replacement_y)
				    < F(damper_replacement))) {
					replacement_accepted = 1;
					replacement_y = damper_replacement;
				}
			}
			contact_replacement = replacement_y;
			if (F(replacement_y) <= F(real_time->field08))
				replacement_y = real_time->field08;
			real_time->field08 = replacement_y;
			{
				float zero_replacement = x87_mul(
					0.0f, contact_replacement);

				contact->replacement.x = x87_sub(
					contact->replacement.x, zero_replacement);
				contact->replacement.y = x87_sub(
					contact->replacement.y, contact_replacement);
				contact->replacement.z = x87_sub(
					contact->replacement.z, zero_replacement);
			}
		}

		if (!ground_contact) {
			float restitution =
				contact->other_surface_material == 4
					? tuning->restitution_air_material4
					: tuning->restitution_air;
			float normal_speed = vec_dot_yxz(
				&contact->normal, &contact->relative_speed);

			if (F(normal_speed) < 0.0) {
				GmVec3 impulse_point = contact->position;

				impulse_point.y =
					context->vehicle->dyna_params->comOffset.y;
				apply_wheel_impulse(
					context, &contact->relative_speed,
					&contact->normal, &impulse_point,
					restitution);
			}
		} else {
			float restitution =
				contact->other_surface_material == 4
					? tuning->restitution_ground_material4
					: tuning->restitution_ground;
			float normal_speed = vec_dot_yxz(
				&contact->normal, &contact->relative_speed);

			if (F(normal_speed) < 0.0) {
				GmVec3 projected;
				float vertical_projection;

				projected.x = x87_mul(
					contact->normal.x, normal_speed);
				projected.y = x87_mul(
					contact->normal.y, normal_speed);
				projected.z = x87_mul(
					normal_speed, contact->normal.z);
				vertical_projection = x87_add(
					x87_add(
						x87_mul(projected.x, 0.0f),
						projected.y),
					x87_mul(projected.z, 0.0f));
				if (replacement_accepted
					|| !(F(vertical_projection) < 0.0)) {
					apply_wheel_impulse(
						context, &contact->relative_speed,
						&contact->normal, &contact->position,
						restitution);
				} else {
					GmVec3 adjusted_speed;
					float horizontal_projection = x87_mul(
						vertical_projection, 0.0f);

					adjusted_speed.x = x87_sub(
						contact->relative_speed.x,
						horizontal_projection);
					adjusted_speed.y = x87_sub(
						contact->relative_speed.y,
						vertical_projection);
					adjusted_speed.z = x87_sub(
						contact->relative_speed.z,
						horizontal_projection);
					if (F(vec_dot_yxz(
						    &contact->normal,
						    &adjusted_speed)) < 0.0) {
						apply_wheel_impulse(
							context, &adjusted_speed,
							&contact->normal,
							&wheel_state->impulse_point,
							restitution);
					}
				}
			}
		}
	}
}

TMNF_HD static int32_t wheel_index_from_tree(
	const TMNFVehicleContactContext *context, uint32_t tree_ref)
{
	for (uint32_t i = 0; i < context->wheel_count; ++i) {
		if (context->wheel_tree_refs[i] == tree_ref)
			return (int32_t)i;
	}
	for (uint32_t i = 0; i < context->body_tree_count; ++i) {
		if (context->body_tree_refs[i] == tree_ref)
			return -1;
	}
	tmnf_abort();
}

TMNF_HD static void absorb_body_contact(
	TMNFVehicleContactContext *context, CHmsPhysicalContact *contact)
{
	const TMNFVehicleContactTuning *tuning = context->tuning;
	CSceneVehicleCar *vehicle = context->vehicle;
	const CHmsDynaParams *params = vehicle->dyna_params;
	float normal_speed;
	float tangent_ratio;
	float restitution;
	GmVec3 normal_projection;
	GmVec3 tangent;
	GmVec3 impulse_direction;
	float normal_length;
	float tangent_length;
	float impulse_length;
	GmVec3 lever;
	GmVec3 impulse;

	if (F(contact->normal.y) < -0.75
		&& tuning->friction_model == 5) {
		float replacement_projection =
			vec_dot_yxz(&contact->normal, &contact->replacement);

		contact->replacement.x = x87_mul(
			contact->normal.x, replacement_projection);
		contact->replacement.y = x87_mul(
			contact->normal.y, replacement_projection);
		contact->replacement.z = x87_mul(
			replacement_projection, contact->normal.z);
	}

	context->body_contact_position_sum.x = x87_add(
		context->body_contact_position_sum.x, contact->position.x);
	context->body_contact_position_sum.y = x87_add(
		contact->position.y, context->body_contact_position_sum.y);
	context->body_contact_position_sum.z = x87_add(
		contact->position.z, context->body_contact_position_sum.z);
	context->body_contact_normal_sum.x = x87_add(
		contact->normal.x, context->body_contact_normal_sum.x);
	context->body_contact_normal_sum.y = x87_add(
		contact->normal.y, context->body_contact_normal_sum.y);
	context->body_contact_normal_sum.z = x87_add(
		contact->normal.z, context->body_contact_normal_sum.z);
	context->body_contact_count++;
	*context->contact_block_count = 1;
	*context->event_source_c = (uint8_t)contact->other_surface_material;

	if (tuning->wheel_contact_model != 2)
		goto reject;
	normal_speed = vec_dot_yxz(
		&contact->normal, &contact->relative_speed);
	if (!(F(normal_speed) < 0.0))
		goto reject;

	if (contact->other_surface_material == 4) {
		tangent_ratio = tuning->body_tangent_ratio_material4;
		restitution = tuning->restitution_air_material4;
	} else {
		tangent_ratio = tuning->body_tangent_ratio;
		restitution = tuning->restitution_air;
	}
	normal_projection.x = x87_mul(
		contact->normal.x, normal_speed);
	normal_projection.y = x87_mul(
		contact->normal.y, normal_speed);
	normal_projection.z = x87_mul(
		normal_speed, contact->normal.z);
	tangent.x = x87_sub(
		contact->relative_speed.x, normal_projection.x);
	tangent.y = x87_sub(
		contact->relative_speed.y, normal_projection.y);
	tangent.z = x87_sub(
		contact->relative_speed.z, normal_projection.z);
	normal_length = x87_sqrt(
		vec_length_squared_xyz(&normal_projection));
	tangent_length = x87_sqrt(vec_length_squared_xyz(&tangent));
	{
		float tangent_limit = x87_mul(
			normal_length, tangent_ratio);

		if (F(tangent_limit) < F(tangent_length)) {
			float scale = x87_div(tangent_limit, tangent_length);

			tangent.x = x87_mul(scale, tangent.x);
			tangent.y = x87_mul(tangent.y, scale);
			tangent.z = x87_mul(scale, tangent.z);
		}
	}
	impulse_direction.x = -x87_add(
		tangent.x, normal_projection.x);
	impulse_direction.y = -x87_add(
		tangent.y, normal_projection.y);
	impulse_direction.z = -x87_add(
		tangent.z, normal_projection.z);
	impulse_length = x87_sqrt(
		vec_length_squared_xyz(&impulse_direction));
	if (!(F(FRICTION_SPEED_EPSILON) < F(impulse_length)))
		goto reject;
	{
		float inverse_length = x87_rcp(impulse_length);

		impulse_direction.x = x87_mul(
			inverse_length, impulse_direction.x);
		impulse_direction.y = x87_mul(
			impulse_direction.y, inverse_length);
		impulse_direction.z = x87_mul(
			inverse_length, impulse_direction.z);
	}
	if (params == NULL)
		tmnf_abort();
	lever.x = x87_sub(contact->position.x, params->comOffset.x);
	lever.y = x87_sub(contact->position.y, params->comOffset.y);
	lever.z = x87_sub(contact->position.z, params->comOffset.z);
	SDynaMath_ComputeImpulse(
		params->mass, &params->invInertiaBody, -restitution,
		&contact->relative_speed, &impulse_direction, &lever, &impulse);
	CSceneVehicleCar_AddVehicleImpulse(
		context, &impulse, &contact->position);

reject:
	contact->accepted = 0;
}

/*
 * 0x007C3410: filters scene contacts, records impact metrics, maps the
 * collision tree to a wheel, and dispatches wheel or body response.
 */
TMNF_HD void CSceneVehicleCar_AbsorbContact(
	TMNFVehicleContactContext *context, CHmsPhysicalContact *contact)
{
	int32_t wheel_index;
	float impact;

	if (contact->other_surface_material == 13u
		|| contact->other_surface_material == 23u) {
		contact->replacement = (GmVec3){ 0.0f, 0.0f, 0.0f };
		contact->accepted = 0;
		return;
	}
	if (context->wheel_tree_refs == NULL ||
		context->body_tree_refs == NULL ||
		context->air_control_immediate == NULL ||
		context->contact_block_count == NULL ||
		context->event_source_c == NULL ||
		context->event_source_ab == NULL ||
		context->event_metric_a == NULL ||
		context->event_metric_b == NULL ||
		context->event_metric_c == NULL) {
		tmnf_abort();
	}
	*context->air_control_immediate = 1;
	wheel_index = wheel_index_from_tree(context, contact->tree_ref);
	impact = (float)fabs(F(vec_dot_yxz(
		&contact->normal, &contact->relative_speed)));
	if (wheel_index >= 0 && !(F(contact->normal.y) < F(0.2f))) {
		if (context->wheels[wheel_index].wheel->steerable != 0) {
			*context->event_metric_a = x87_add(
				impact, *context->event_metric_a);
		} else {
			*context->event_metric_b = x87_add(
				*context->event_metric_b, impact);
		}
	} else {
		*context->event_metric_c = x87_add(
			*context->event_metric_c, impact);
	}

	if (wheel_index < 0) {
		absorb_body_contact(context, contact);
		return;
	}
	*context->event_source_ab =
		(uint8_t)contact->other_surface_material;
	CSceneVehicleCar_WheelAbsorbContact(
		context, &context->wheels[wheel_index], contact);
	context->wheel_contact_absorb_count++;
}

/*
 * 0x007BEB40, UNVALIDATED: maps the contact-normal slope to two configured
 * adherence ramps.
 */
TMNF_HD void CSceneVehicleCar_GetSlopeAdherence(
	const TMNFVehicleContactContext *context, const GmVec3 *normal,
	float *adherence, float *secondary)
{
	const TMNFVehicleContactTuning *tuning = context->tuning;
	float length_squared = vec_length_squared_yxz(normal);

	if (F(CONTACT_LENGTH_EPSILON) < F(length_squared)) {
		float length = x87_sqrt(length_squared);
		float ratio = (float)fabs(F(x87_div(normal->y, length)));

		*adherence = ratio;
		if (F(ratio) < F(tuning->slope_adherence_min))
			*adherence = 0.0f;
		else if (F(tuning->slope_adherence_max) < F(ratio))
			*adherence = 1.0f;
		else
			*adherence = slope_curve(
				ratio, tuning->slope_adherence_min,
				tuning->slope_adherence_max);

		ratio = (float)fabs(F(x87_div(normal->y, length)));
		*secondary = ratio;
		if (F(ratio) < F(tuning->slope_secondary_min)) {
			*secondary = 0.0f;
			return;
		}
		if (F(tuning->slope_secondary_max) < F(ratio)) {
			*secondary = 1.0f;
			return;
		}
		*secondary = slope_curve(
			ratio, tuning->slope_secondary_min,
			tuning->slope_secondary_max);
	}
}

/* 0x007BD1E0, UNVALIDATED: reports whether any wheel has ground contact. */
TMNF_HD int CSceneVehicleCar_IsGroundContact(
	const TMNFVehicleContactContext *context)
{
	uint32_t index;

	for (index = 0; index < context->wheel_count; ++index) {
		if (context->wheels[index].wheel->real_time.has_ground_contact != 0)
			return 1;
	}
	return 0;
}

/*
 * 0x007BF5C0, UNVALIDATED: reports whether every contacting wheel has the
 * requested material ID and at least one wheel is contacting.
 */
TMNF_HD int CSceneVehicleCar_IsAllWheelGroundContactId(
	const TMNFVehicleContactContext *context, uint8_t material_id)
{
	uint32_t inactive_count = 0;
	uint32_t index;

	for (index = 0; index < context->wheel_count; ++index) {
		const CSceneVehicleCarWheelRealTimeState *state =
			&context->wheels[index].wheel->real_time;

		if (state->has_ground_contact == 0)
			inactive_count++;
		else if ((uint16_t)state->contact_material_id
			!= (uint16_t)material_id)
			return 0;
	}
	return inactive_count < context->wheel_count;
}

/*
 * 0x007BF620, UNVALIDATED: returns the first contacting wheel with the
 * requested material ID.
 */
TMNF_HD int CSceneVehicleCar_IsGroundContactId(
	const TMNFVehicleContactContext *context, uint8_t material_id,
	GmVec3 *relative_axis, CHmsResponseBody **contact_body)
{
	uint32_t index;

	for (index = 0; index < context->wheel_count; ++index) {
		const TMNFVehicleContactWheelState *wheel = &context->wheels[index];
		const CSceneVehicleCarWheelRealTimeState *state =
			&wheel->wheel->real_time;

		if (state->has_ground_contact != 0
			&& (uint16_t)state->contact_material_id
				== (uint16_t)material_id) {
			*relative_axis = state->relative_rotz_axis;
			*contact_body = wheel->contact_body;
			return 1;
		}
	}
	return 0;
}

/*
 * 0x007BED10, UNVALIDATED: applies central drag and timed lateral contact
 * slowdown forces.
 */
TMNF_HD void CSceneVehicleCar_ApplyFrictionForces(
	TMNFVehicleContactContext *context, const GmVec3 *velocity)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	TMNFVehicleContactTuning *tuning = context->tuning;
	float selected_input;

	if ((tuning->friction_model == 4 || tuning->friction_model == 5)
		&& context->airborne_friction_gate != 0
		&& CSceneVehicleCar_IsGroundContact(context) == 0) {
		return;
	}

	selected_input = context->friction_input_selector == 0
		? vehicle->input_gas : vehicle->input_brake;
	if (F(selected_input) < F(FRICTION_SPEED_EPSILON)
		|| vehicle->flag_60c != 0) {
		GmVec3 force = *velocity;
		float length_squared = vec_length_squared_yxz(&force);

		if (F(CONTACT_LENGTH_EPSILON) < F(length_squared)) {
			float length = x87_sqrt(length_squared);
			float inverse_length = x87_rcp(length);
			float scale;

			force.x = x87_mul(inverse_length, force.x);
			force.y = x87_mul(force.y, inverse_length);
			force.z = x87_mul(inverse_length, force.z);
			scale = -tuning->friction_force;
			force.x = x87_mul(scale, force.x);
			force.y = x87_mul(force.y, scale);
			force.z = x87_mul(scale, force.z);
			if (vehicle->flag_60c == 0) {
				GmVec3 extra;

				scale = -tuning->extra_friction_force;
				extra.x = x87_mul(velocity->x, scale);
				extra.y = x87_mul(velocity->y, scale);
				extra.z = x87_mul(scale, velocity->z);
				force.x = x87_add(extra.x, force.x);
				force.y = x87_add(extra.y, force.y);
				force.z = x87_add(extra.z, force.z);
			}
			CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);
		}
	}

	if (tuning->friction_model < 4) {
		if (context->side_contact != 0) {
			GmVec3 force;
			float scale;

			/* The game stores interpolation mode 1 into the tuning
			 * curve here; the step accessor below evaluates with mode
			 * 1 regardless and nothing else on a model < 4 car reads
			 * the field, so the tuning stays read-only (cold). */
			scale = -CSceneVehicleCarTuning_GetLateralContactSlowDownFromSpeed(
				tuning->curves, velocity->z);
			force.x = x87_mul(scale, velocity->x);
			force.y = x87_mul(velocity->y, scale);
			force.z = x87_mul(scale, velocity->z);
			CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);
		}
		return;
	}

	if (context->timer == NULL)
		tmnf_abort();
	{
		uint32_t tick = *CMwTimerAdapter_GetTickTime(context->timer);

		if (context->side_contact != 0)
			context->last_side_contact_tick = tick;
		if (context->last_side_contact_tick <= tick
			&& tick - context->last_side_contact_tick
				< tuning->lateral_contact_duration_ticks) {
			float length_squared = vec_length_squared_xyz(velocity);
			float length = x87_sqrt(length_squared);

			if (F(FRICTION_SPEED_EPSILON) < F(length)) {
				float inverse_length = x87_rcp(length);
				GmVec3 normalized;
				GmVec3 force;
				float scale;

				normalized.x = x87_mul(
					velocity->x, inverse_length);
				normalized.y = x87_mul(
					velocity->y, inverse_length);
				normalized.z = x87_mul(
					inverse_length, velocity->z);
				scale =
					-CSceneVehicleCarTuning_M5GetLateralContactSlowDownFromSpeed(
						tuning->curves, length);
				force.x = x87_mul(scale, normalized.x);
				force.y = x87_mul(normalized.y, scale);
				force.z = x87_mul(scale, normalized.z);
				CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);
			}
		}
	}
}

/*
 * 0x007BF080, UNVALIDATED: computes and clamps the Model 4 lateral friction
 * force for one material blend.
 */
TMNF_HD void CSceneVehicleCar_GetLateralFriction(
	const TMNFVehicleContactContext *context, const GmVec3 *velocity,
	const GmVec3 *lateral_axis,
	const CSceneVehicleMaterialBlendableVals *material, float load,
	int grounded, float *friction, int *sliding)
{
	const TMNFVehicleContactTuning *tuning = context->tuning;
	float lateral_speed = vec_dot_yxz(velocity, lateral_axis);
	float absolute_speed = (float)fabs(F(lateral_speed));
	float linear = x87_mul(tuning->lateral_linear, -lateral_speed);
	float quadratic = x87_mul(
		x87_mul(lateral_speed, absolute_speed),
		tuning->lateral_quadratic);
	float requested = x87_sub(linear, quadratic);
	float ground_scale = grounded == 0
		? 1.0f : tuning->lateral_ground_scale;
	float maximum = CSceneVehicleCarTuning_M4GetMaxFrictionForceFromSpeed(
		tuning->curves, absolute_speed);
	float limit = x87_mul(material->lateral_grip, load);
	uint32_t requested_bits;

	limit = x87_mul(limit, maximum);
	limit = x87_mul(limit, ground_scale);
	if (F(limit) < F(fabs(F(requested)))) {
		float sign;

		memcpy(&requested_bits, &requested, sizeof(requested_bits));
		sign = (requested_bits & UINT32_C(0x80000000)) != 0
			? -1.0f : 1.0f;
		*sliding = 1;
		*friction = x87_mul(limit, sign);
		return;
	}
	*sliding = 0;
	*friction = requested;
}

/*
 * 0x007C2800, UNVALIDATED: averages the material blend values for contacting
 * wheels. The original routine indexes material selection with wheel zero.
 */
TMNF_HD void CSceneVehicleCar_ComputeVehicleGroundMaterialVals(
	const TMNFVehicleContactContext *context,
	CSceneVehicleMaterialBlendableVals *values, int *has_material)
{
	float *output = &values->acceleration;
	uint32_t contact_count = 0;
	uint32_t index;

	*has_material = 0;
	output[0] = 0.0f;
	output[1] = 0.0f;
	output[2] = 0.0f;
	output[3] = 0.0f;
	for (index = 0; index < context->wheel_count; ++index) {
		if (context->wheels[index].wheel->real_time.has_ground_contact != 0) {
			uint16_t material_id = (uint16_t)
				context->wheels[0].wheel->real_time.contact_material_id;
			uint32_t material_index =
				context->ground_material_indices[material_id];
			const TMNFVehicleGroundMaterial *material =
				context->ground_materials[material_index];

			contact_count++;
			output[0] = x87_add(material->values[0], output[0]);
			output[1] = x87_add(material->values[1], output[1]);
			output[2] = x87_add(material->values[2], output[2]);
			output[3] = x87_add(material->values[3], output[3]);
			*has_material = 1;
		}
	}
	if (contact_count != 0) {
		int32_t signed_count;
		float float_count;
		float inverse_count;

		memcpy(&signed_count, &contact_count, sizeof(signed_count));
		float_count = (float)signed_count;
		if (signed_count < 0)
			float_count = x87_add(float_count, 4294967296.0f);
		inverse_count = x87_rcp(float_count);
		output[0] = x87_mul(inverse_count, output[0]);
		output[1] = x87_mul(output[1], inverse_count);
		output[2] = x87_mul(inverse_count, output[2]);
		output[3] = x87_mul(inverse_count, output[3]);
	}
}
