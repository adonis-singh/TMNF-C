#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "tmnf_fp.h"
#include "vehicle_compute.h"

#define COMPUTE_LENGTH_EPSILON 0x1.b7cdfcp-34f
#define COMPUTE_VALUE_EPSILON 0x1.4f8b58p-17f
/* 0x00B3D2A8: float 3.6 promoted to double (0x007C777E, 0x007C7803). */
#define KMH_PER_MPS 0x1.ccccccp+1

TMNF_HD static float vec_length_squared_yxz(const GmVec3 *value)
{
	return x87_add(
		x87_add(
			x87_mul(value->y, value->y),
			x87_mul(value->x, value->x)),
		x87_mul(value->z, value->z));
}

TMNF_HD static float clamp01(float value)
{
	if (isnan(F(value)) || F(value) < 0.0)
		return 0.0f;
	if (1.0 < F(value))
		return 1.0f;
	return value;
}

TMNF_HD static float clamp_symmetric(float value, float limit)
{
	float lower = -limit;

	if (F(value) < F(lower))
		return lower;
	if (F(limit) < F(value))
		return limit;
	return value;
}

TMNF_HD static void set_vehicle_linear_speed(
	TMNFVehicleComputeContext *context, const GmVec3 *speed)
{
	CHmsItem_SetLinearSpeed(context->item, speed);
}

TMNF_HD static void add_vehicle_impulse(
	TMNFVehicleComputeContext *context, const GmVec3 *impulse)
{
	CSceneVehicleCar *vehicle = context->vehicle;

	CHmsItem_AddImpulse(context->item, impulse);
	vehicle->total_impulse_added.x = x87_add(
		vehicle->total_impulse_added.x, impulse->x);
	vehicle->total_impulse_added.y = x87_add(
		impulse->y, vehicle->total_impulse_added.y);
	vehicle->total_impulse_added.z = x87_add(
		impulse->z, vehicle->total_impulse_added.z);
}

TMNF_HD static void clamp_vehicle_linear_speed(
	TMNFVehicleComputeContext *context, GmVec3 *speed)
{
	float maximum = context->state->local_speed_limit;
	float maximum_squared = x87_mul(maximum, maximum);
	float length_squared = vec_length_squared_yxz(speed);

	if (F(maximum_squared) < F(length_squared)
		&& F(COMPUTE_LENGTH_EPSILON) < F(maximum_squared)) {
		float length = x87_sqrt(length_squared);
		float scale = x87_div(maximum, length);

		speed->x = x87_mul(scale, speed->x);
		speed->y = x87_mul(speed->y, scale);
		speed->z = x87_mul(scale, speed->z);
		set_vehicle_linear_speed(context, speed);
	}
}

TMNF_HD static float compute_steering_angle(
	const TMNFVehicleComputeContext *context, const GmVec3 *linear_speed)
{
	const CSceneVehicleCarTuningAux *tuning = context->aux->tuning;
	float denominator = x87_add(
		x87_mul(
			(float)fabs(F(linear_speed->z)),
			tuning->steering_speed_scale),
		tuning->steering_speed_base);
	float angle;

	if (F(COMPUTE_VALUE_EPSILON) <= F(denominator)) {
		float reciprocal = x87_rcp(denominator);

		angle = GmFunc_AsinSafe(reciprocal);
	} else {
		angle = 0.0f;
	}
	return x87_mul(-context->aux->steering_value, angle);
}

TMNF_HD static void scan_wheel_contacts(
	const CSceneVehicleCar *vehicle, int *any_contact,
	int *any_active_contact)
{
	*any_contact = 0;
	*any_active_contact = 0;
	for (uint32_t i = 0; i < vehicle->wheel_count; ++i) {
		const CSceneVehicleCarWheel *wheel = &vehicle->wheels[i];

		if (wheel->real_time.has_ground_contact != 0) {
			*any_contact = 1;
			if (wheel->active != 0)
				*any_active_contact = 1;
		}
	}
}

TMNF_HD static void apply_air_effect(
	TMNFVehicleComputeContext *context, const GmVec3 *existing_force,
	uint32_t tick, int grounded)
{
	TMNFVehicleComputeState *state = context->state;
	const TMNFVehicleComputeTuning *tuning = context->tuning;

	if (!(F(COMPUTE_VALUE_EPSILON) < F(state->air_effect_threshold)))
		return;

	switch (state->air_effect_mode) {
	case 1:
		if (grounded != 0 && state->air_impulse_cooldown_tick < tick) {
			GmVec3 impulse = {
				-existing_force->x,
				-existing_force->y,
				-existing_force->z,
			};
			float length_squared = vec_length_squared_yxz(&impulse);

			if (F(COMPUTE_LENGTH_EPSILON) < F(length_squared)) {
				float length = x87_sqrt(length_squared);
				float inverse_length = x87_rcp(length);

				impulse.x = x87_mul(inverse_length, impulse.x);
				impulse.y = x87_mul(impulse.y, inverse_length);
				impulse.z = x87_mul(inverse_length, impulse.z);
			}
			impulse.x = x87_mul(tuning->air_impulse_scale, impulse.x);
			impulse.y = x87_mul(impulse.y, tuning->air_impulse_scale);
			impulse.z = x87_mul(tuning->air_impulse_scale, impulse.z);
			add_vehicle_impulse(context, &impulse);
			state->air_impulse_cooldown_tick = tick + 100;
		}
		break;
	case 2:
		context->vehicle->dyna_params->forceFieldScale =
			tuning->special_force_field_scale;
		break;
	case 3:
		context->aux->turbo_type = TMNF_TURBO_NORMAL;
		context->aux->turbo_factor = tuning->normal_turbo_factor;
		break;
	default:
		break;
	}
}

TMNF_HD static uint32_t contact_token(
	TMNFVehicleComputeContext *context, const CHmsResponseBody *body)
{
	if (body == NULL)
		return 0;
	if (context->contact_token == NULL)
		tmnf_abort();
	return context->contact_token(context->runtime, body);
}

TMNF_HD static void update_surface_effects(
	TMNFVehicleComputeContext *context, uint32_t tick)
{
	const TMNFVehicleComputeTuning *tuning = context->tuning;
	GmVec3 relative_axis;
	CHmsResponseBody *body;

	if (CSceneVehicleCar_IsGroundContactId(
			context->contact, 7, &relative_axis, &body)) {
		float factor = x87_mul(
			tuning->normal_turbo_factor, relative_axis.z);

		CSceneVehicleCar_EnableTurbo(
			context->aux, tick, tuning->normal_turbo_duration,
			factor, TMNF_TURBO_NORMAL, 0);
	}
	if (CSceneVehicleCar_IsGroundContactId(
			context->contact, 26, &relative_axis, &body)) {
		float factor = x87_mul(
			tuning->roulette_turbo_factor, relative_axis.z);

		CSceneVehicleCar_EnableTurbo(
			context->aux, tick, tuning->roulette_turbo_duration,
			factor, TMNF_TURBO_NORMAL, 0);
	}
	if (CSceneVehicleCar_IsGroundContactId(
			context->contact, 30, &relative_axis, &body)) {
		float factor = x87_mul(
			tuning->normal_turbo_factor, relative_axis.z);
		uint32_t token = contact_token(context, body);

		CSceneVehicleCar_EnableTurbo(
			context->aux, tick, tuning->normal_turbo_duration,
			factor, TMNF_TURBO_ROULETTE, token);
	}
	if (CSceneVehicleCar_IsGroundContactId(
			context->contact, 29, &relative_axis, &body)) {
		context->vehicle->flag_60c = 1;
	}
	CSceneVehicleCar_UpdateTurbo(context->aux, tick);
}

TMNF_HD static void update_event_levels(
	TMNFVehicleComputeContext *context, uint32_t tick)
{
	TMNFVehicleComputeState *state = context->state;
	const TMNFVehicleComputeTuning *tuning = context->tuning;

	if (F(tuning->event_ab_trigger) < F(state->event_metric_a)) {
		if (F(state->event_metric_a) <= F(tuning->event_ab_level2_min)) {
			if (state->event_level_a == 0)
				state->event_level_a = 1;
		} else if (state->event_level_a < 2) {
			state->event_level_a = 2;
		}
	}
	if (F(tuning->event_ab_trigger) < F(state->event_metric_b)) {
		if (F(state->event_metric_b) <= F(tuning->event_ab_level2_min)) {
			if (state->event_level_b == 0)
				state->event_level_b = 1;
		} else if (state->event_level_b < 2) {
			state->event_level_b = 2;
		}
	}
	if (F(tuning->event_c_trigger) < F(state->event_metric_c)) {
		if (F(state->event_metric_c) <= F(tuning->event_c_level1_max)) {
			if (state->event_level_c == 0)
				state->event_level_c = 1;
		} else if (state->event_level_c < 2) {
			state->event_level_c = 2;
		}
	}

	if (state->peak_event_level_b < state->event_level_b) {
		state->peak_event_level_b = state->event_level_b;
		state->peak_event_source_ab = state->event_source_ab;
	}
	if (state->peak_event_level_a < state->event_level_a) {
		state->peak_event_level_a = state->event_level_a;
		state->peak_event_source_ab = state->event_source_ab;
	}
	if (state->peak_event_level_c < state->event_level_c) {
		state->peak_event_level_c = state->event_level_c;
		state->peak_event_source_c = state->event_source_c;
	}
	state->last_force_tick = tick;
}

TMNF_HD static void update_force_history(
	TMNFVehicleComputeContext *context, float dt,
	const GmVec3 *old_force, const GmVec3 *old_impulse)
{
	TMNFVehicleComputeState *state = context->state;
	float history;
	float adjustment;

	GmSpringFloat_Integrate(&state->spring_a, dt);
	history = x87_add(x87_mul(old_force->x, dt), old_impulse->x);
	history = clamp_symmetric(history, state->history_force_limit);
	adjustment = x87_mul(
		x87_div(history, state->history_force_limit),
		state->history_force_scale);
	state->spring_a.value = clamp_symmetric(
		state->spring_a.value, state->spring_value_limit);
	state->spring_a.velocity = clamp_symmetric(
		x87_add(state->spring_a.velocity, adjustment),
		state->history_force_scale);

	GmSpringFloat_Integrate(&state->spring_c, dt);
	history = x87_sub(
		x87_mul(-old_force->z, dt), old_impulse->z);
	history = clamp_symmetric(history, state->history_force_limit);
	adjustment = x87_mul(
		x87_div(history, state->history_force_limit),
		state->history_force_scale);
	state->spring_c.value = clamp_symmetric(
		state->spring_c.value, state->spring_value_limit);
	state->spring_c.velocity = clamp_symmetric(
		x87_add(state->spring_c.velocity, adjustment),
		state->history_force_scale);
}

TMNF_HD static void update_contact_accumulators(
	TMNFVehicleComputeContext *context, float dt,
	const GmVec3 *linear_speed)
{
	TMNFVehicleComputeState *state = context->state;
	const TMNFVehicleComputeTuning *tuning = context->tuning;
	float direction =
		CSceneVehicleCar_IsAllWheelGroundContactId(context->contact, 6)
		? 1.0f : -1.0f;
	float speed_kmh = (float)fabs(
		F(x87_r24(F(linear_speed->z) * KMH_PER_MPS)));
	float curve_value;
	float delta;

	if (tuning->contact_rise_curve == NULL
		|| tuning->contact_decay_curve == NULL) {
		tmnf_abort();
	}
	curve_value = CFuncKeysReal_GetValue(
		tuning->contact_rise_curve, speed_kmh, NULL);
	delta = x87_mul(x87_mul(curve_value, dt), direction);
	state->contact_rise = clamp01(x87_add(delta, state->contact_rise));

	curve_value = CFuncKeysReal_GetValue(
		tuning->contact_decay_curve, speed_kmh, NULL);
	delta = x87_mul(x87_mul(curve_value, dt), direction);
	state->contact_decay = clamp01(
		x87_add(delta, state->contact_decay));
}

TMNF_HD static void reset_contact_frame(TMNFVehicleComputeContext *context)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	uint16_t zero_material = 0;

	for (uint32_t i = 0; i < vehicle->wheel_count; ++i) {
		CSceneVehicleCarWheel *wheel = &vehicle->wheels[i];
		CSceneVehicleCarWheelRealTimeState *real_time = &wheel->real_time;

		real_time->has_ground_contact = 0;
		real_time->ground_contact_count = 0;
		real_time->field54 = (GmVec3){ 0.0f, 0.0f, 0.0f };
		real_time->field90 = (GmVec3){ 0.0f, 0.0f, 0.0f };
		memcpy(
			&real_time->contact_material_id,
			&zero_material, sizeof(zero_material));
		wheel->field15c = 0;
	}
	context->state->event_metric_a = 0.0f;
	context->state->state_5d8 = 0;
	context->state->event_metric_b = 0.0f;
	context->aux->air_control_immediate = 0;
	context->state->event_metric_c = 0.0f;
	context->contact->side_contact = 0;
	context->contact->wheel_contact_absorb_count = 0;
	context->contact->body_contact_count = 0;
	context->contact->body_contact_position_sum =
		(GmVec3){ 0.0f, 0.0f, 0.0f };
	context->contact->body_contact_normal_sum =
		(GmVec3){ 0.0f, 0.0f, 0.0f };
}

TMNF_HD static void finish_active_frame(
	TMNFVehicleComputeContext *context, float dt,
	const GmVec3 *linear_speed, const GmVec3 *old_force,
	const GmVec3 *old_impulse, float effect_curve_position)
{
	TMNFVehicleComputeState *state = context->state;
	const TMNFVehicleComputeTuning *tuning = context->tuning;
	GmVec3 force;
	float inverse_divisor;
	float curve_value;
	float delta;

	CHmsItem_GetForce(context->item, &force);
	inverse_divisor = x87_rcp(tuning->normalized_force_divisor);
	state->normalized_force.x = x87_mul(inverse_divisor, force.x);
	state->normalized_force.y = x87_mul(force.y, inverse_divisor);
	state->normalized_force.z = x87_mul(inverse_divisor, force.z);

	if (tuning->effect_curve == NULL)
		tmnf_abort();
	curve_value = CFuncKeysReal_GetValue(
		tuning->effect_curve, effect_curve_position, NULL);
	delta = x87_mul(dt, x87_add(curve_value, tuning->effect_curve_bias));
	state->effect_accumulator = clamp01(
		x87_add(state->effect_accumulator, delta));

	update_force_history(context, dt, old_force, old_impulse);
	update_contact_accumulators(context, dt, linear_speed);
	reset_contact_frame(context);
}

/* 0x007C69E0 */
TMNF_HD void CSceneVehicleCar_ComputeForces(
	TMNFVehicleComputeContext *context, float dt)
{
	CSceneVehicleCar *vehicle = context->vehicle;
	TMNFVehicleComputeState *state = context->state;
	const TMNFVehicleComputeTuning *tuning = context->tuning;
	GmVec3 old_impulse = vehicle->total_impulse_added;
	GmVec3 old_force = vehicle->total_force_added;
	uint32_t integration_flags = context->aux->integration_flags;
	GmVec3 linear_speed;
	GmVec3 angular_speed = { 0.0f, 0.0f, 0.0f };
	GmVec3 existing_force = { 0.0f, 0.0f, 0.0f };
	CSceneVehicleMaterialBlendableVals ground_material = {
		1.0f, 1.0f, 1.0f, 1.0f
	};
	int has_ground_material = 0;
	float slope_adherence = 1.0f;
	float slope_secondary = 1.0f;
	float steering_angle;
	float effect_curve_position = 0.0f;
	int model6_sliding = 0;
	int grounded;
	uint32_t tick;

	if (context->aux->vehicle != vehicle
		|| context->contact->vehicle != vehicle
		|| context->contact->wheel_count != vehicle->wheel_count) {
		tmnf_abort();
	}

	vehicle->total_force_added = (GmVec3){ 0.0f, 0.0f, 0.0f };
	vehicle->total_impulse_added = (GmVec3){ 0.0f, 0.0f, 0.0f };

	if ((integration_flags & 0x30u) != 0
		|| F(state->simulation_gate) < 0.0) {
		const GmVec3 zero = { 0.0f, 0.0f, 0.0f };

		CHmsItem_SetLinearSpeed(context->item, &zero);
		CHmsItem_SetAngularSpeed(context->item, &zero);
		CHmsItem_SetForce(context->item, &zero);
		CHmsItem_SetTorque(context->item, &zero);
		return;
	}

	if (context->fake_contacts_active != 0)
		tmnf_abort();
	CSceneVehicleCar_CreateFakeContacts(context->aux, context->contact);
	CSceneVehicleCar_IntegrateVehicle(context->aux, dt);

	if (context->contact->timer == NULL)
		tmnf_abort();
	tick = *CMwTimerAdapter_GetTickTime(context->contact->timer);
	grounded = CSceneVehicleCar_IsGroundContact(context->contact);
	vehicle->dyna_params->forceFieldScale = grounded != 0
		? tuning->grounded_force_field_scale
		: tuning->airborne_force_field_scale;
	vehicle->dyna_params->dragLinear = grounded != 0
		? 0.0f : tuning->airborne_linear_drag;

	if ((integration_flags & 2u) == 0)
		return;

	CHmsItem_GetLinearSpeed(context->item, &linear_speed);
	if ((integration_flags & 8u) != 0) {
		linear_speed.x = 0.0f;
		linear_speed.z = 0.0f;
		set_vehicle_linear_speed(context, &linear_speed);
		finish_active_frame(
			context, dt, &linear_speed, &old_force,
			&old_impulse, effect_curve_position);
		return;
	}

	CHmsItem_GetAngularSpeed(context->item, &angular_speed);
	CHmsItem_GetForce(context->item, &existing_force);
	state->computed_brake_force = 0.0f;
	CSceneVehicleCar_ApplyFrictionForces(context->contact, &linear_speed);
	clamp_vehicle_linear_speed(context, &linear_speed);
	CSceneVehicleCar_ComputeVehicleGroundMaterialVals(
		context->contact, &ground_material, &has_ground_material);
	CSceneVehicleCar_GetSlopeAdherence(
		context->contact, &existing_force,
		&slope_adherence, &slope_secondary);
	steering_angle = compute_steering_angle(context, &linear_speed);
	vehicle->current_local_speed = linear_speed;

	if (context->water_forces_active != 0)
		tmnf_abort();
	/* 0x007C69E0 tests +0x354 for 3, 4, 5 in that order; the else branch
	 * is Model 3 (values 0, 1, 2). */
	if (context->contact->tuning->friction_model == 5) {
		VehicleModel6_ComputeForces(
			context, dt, &existing_force, slope_adherence,
			slope_secondary, &linear_speed, &angular_speed,
			steering_angle, has_ground_material, &ground_material,
			&model6_sliding, &effect_curve_position);
	} else {
		VehicleOldModels_ComputeForces(
			context, dt, &existing_force, slope_adherence,
			slope_secondary, &linear_speed, &angular_speed,
			steering_angle, has_ground_material, &ground_material,
			&model6_sliding, &effect_curve_position);
	}

	{
		int any_contact;
		int any_active_contact;

		scan_wheel_contacts(
			vehicle, &any_contact, &any_active_contact);
		if (any_contact != 0) {
			float braking = x87_mul(
				-vehicle->input_brake, state->brake_input_scale);
			float grounded_drag = x87_mul(
				tuning->grounded_drag_term,
				state->grounded_drag_scale);

			state->computed_brake_force =
				x87_sub(braking, grounded_drag);
		}
		if (any_active_contact != 0
			&& F(tuning->active_contact_stop_threshold) < 0.0) {
			linear_speed.x = 0.0f;
			set_vehicle_linear_speed(context, &linear_speed);
		}
		/* 0x007C6ED6..0x007C6F9F: the reset argument is [esp+0x2c], set at
		 * 0x007C6F0F when a wheel has contact (+0x124) and is active (+0x0);
		 * the Model6 sliding out-parameter is the separate [esp+0x28]. */
		CSceneVehicleCar_ComputeAirControl(
			context->aux, &angular_speed, tick,
			grounded, any_active_contact);
	}

	apply_air_effect(context, &existing_force, tick, grounded);
	update_event_levels(context, tick);
	update_surface_effects(context, tick);
	if (context->post_force == NULL)
		tmnf_abort();
	context->post_force(context->runtime, vehicle);

	finish_active_frame(
		context, dt, &linear_speed, &old_force,
		&old_impulse, effect_curve_position);
}

/* 0x007C7D40 */
TMNF_HD void CCallbackSceneVehicleCarComputeForces_ComputeForces(
	TMNFVehicleComputeContext *context, CHmsItem *item, float dt)
{
	if (context->item != item)
		tmnf_abort();
	CSceneVehicleCar_ComputeForces(context, dt);
}

TMNF_HD void TMNFVehicleComputeForces_PhysicsStep2Adapter(
	void *user, CSceneVehicleCar *vehicle, float dt)
{
	TMNFVehicleComputeContext *context = (TMNFVehicleComputeContext *)user;

	if (context->vehicle != vehicle)
		tmnf_abort();
	CCallbackSceneVehicleCarComputeForces_ComputeForces(
		context, context->item, dt);
}
