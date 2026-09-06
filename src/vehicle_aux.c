#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hms_item.h"
#include "tmnf_fp.h"
#include "vehicle_aux.h"
#include "vehicle_contact.h"
#include "vehicle_fake_contact_mask.h"
#include "track.h"
#include "vehicle_curve.h"

#define AUX_EPSILON 0x1.4f8b58p-17f
#define ROULETTE_LOW 0x1.24924ap-1f
/* 0x00B9EF58: float 6/7 promoted to double. */
#define ROULETTE_HIGH 0x1.b6db6ep-1
#define UINT32_RANGE_FLOAT 0x1p32f

TMNF_HD static float u32_to_x87_float(uint32_t value)
{
	int32_t signed_value = (int32_t)value;
	float result = (float)signed_value;

	if (signed_value < 0)
		result = x87_add(result, UINT32_RANGE_FLOAT);
	return result;
}

/*
 * UNVALIDATED: native view of 0x00586200 for the two auxiliary tuning curves.
 * All callers begin the forward search at key zero, as the original functions
 * do at these call sites.
 */
TMNF_HD static float aux_curve_get_value(
	const TMNFVehicleAuxCurve *curve, float position)
{
	uint32_t lower;
	uint32_t upper;
	float difference;
	float blend;
	float lower_weight;

	if (curve->count == 0)
		return 0.0f;
	if (curve->count == 1)
		return curve->values[0];
	if (position < curve->lower_bounds[0])
		return curve->values[0];
	lower = curve->count - 1;
	if (curve->upper_bounds[lower] < position)
		return curve->values[lower];

	lower = 0;
	upper = 1;
	for (uint32_t attempts = 1; attempts <= curve->count; ++attempts) {
		if (curve->lower_bounds[lower] <= position
			&& position <= curve->upper_bounds[upper])
			break;
		lower = upper;
		upper++;
		if (upper == curve->count)
			upper = 0;
	}
	if (curve->interpolation == 1)
		return curve->values[lower];

	difference = x87_sub(
		curve->positions[upper], curve->positions[lower]);
	if (fabs(F(difference)) < F(AUX_EPSILON))
		blend = 0.0f;
	else
		blend = x87_div(x87_sub(position, curve->positions[lower]), difference);
	lower_weight = x87_r24(1.0 - F(blend));
	return x87_add(
		x87_mul(lower_weight, curve->values[lower]),
		x87_mul(blend, curve->values[upper]));
}

/* 0x00534400, UNVALIDATED: transforms local angular speed into world space. */
TMNF_HD static void set_local_angular_speed(
	CSceneVehicleCar *vehicle, const GmVec3 *local_speed)
{
	GmVec3 world_speed;

	GmVec3_SetMult_Mat3(
		&world_speed, local_speed, &vehicle->dyna_state->rot);
	vehicle->dyna_state->angVel = world_speed;
}

/* UNVALIDATED: local 0x008E0D70 schedule used by IntegrateVehicle. */
TMNF_HD static void mat3_rotate_y(GmMat3 *matrix, float angle)
{
	float sine = x87_sin(angle);
	float cosine = x87_cos(angle);
	float negative_sine = -sine;
	float old_value;

	old_value = matrix->m[0];
	matrix->m[0] = x87_add(
		x87_mul(matrix->m[6], sine),
		x87_mul(cosine, old_value));
	matrix->m[6] = x87_add(
		x87_mul(matrix->m[6], cosine),
		x87_mul(negative_sine, old_value));
	old_value = matrix->m[1];
	matrix->m[1] = x87_add(
		x87_mul(old_value, cosine),
		x87_mul(matrix->m[7], sine));
	matrix->m[7] = x87_add(
		x87_mul(matrix->m[7], cosine),
		x87_mul(negative_sine, old_value));
	old_value = matrix->m[2];
	matrix->m[2] = x87_add(
		x87_mul(old_value, cosine),
		x87_mul(matrix->m[8], sine));
	matrix->m[8] = x87_add(
		x87_mul(matrix->m[8], cosine),
		x87_mul(negative_sine, old_value));
}

/* 0x007C8AA0, UNVALIDATED: binds the surface transform to its runtime tree. */
TMNF_HD static void update_surface(
	CSceneVehicleCarAuxContext *self, uint32_t wheel_index)
{
	CSceneVehicleCarWheel *wheel = &self->vehicle->wheels[wheel_index];

	if (wheel->surface_handler == NULL)
		return;
	if (self->set_surface_location == NULL)
		tmnf_abort();
	self->set_surface_location(
		self->runtime, wheel->surface_handler,
		&self->wheels[wheel_index].surface_location);
}

/* 0x007BC820, UNVALIDATED: maps an unsigned roulette remainder to {0,.5,1}. */
TMNF_HD float CSceneVehicleCar_GetRouletteValue01(
	uint32_t value, uint32_t divisor)
{
	uint32_t remainder = value % divisor;
	float numerator = u32_to_x87_float(remainder);
	float denominator = u32_to_x87_float(divisor);
	float ratio = x87_div(numerator, denominator);

	if (F(ratio) < F(ROULETTE_LOW))
		return 0.0f;
	if (F(ratio) < ROULETTE_HIGH)
		return 0.5f;
	return 1.0f;
}

/* 0x007BC890, UNVALIDATED: converts roulette value to its boost multiplier. */
TMNF_HD float CSceneVehicleCar_GetRouletteBoostFactorFromValue01(float value)
{
	return x87_add(value, 1.0f);
}

/* 0x007BC8B0: advances the active turbo interval. */
TMNF_HD void CSceneVehicleCar_UpdateTurbo(
	CSceneVehicleCarAuxContext *self, uint32_t tick)
{
	if (self->turbo_type != TMNF_TURBO_NONE) {
		if (self->turbo_end_tick < tick)
			self->turbo_type = TMNF_TURBO_NONE;
		if (self->turbo_type != TMNF_TURBO_NONE) {
			float elapsed = u32_to_x87_float(
				tick - self->turbo_start_tick);
			float duration = u32_to_x87_float(
				self->turbo_end_tick - self->turbo_start_tick);

			self->turbo_progress =
				x87_div(elapsed, duration);
			return;
		}
	}
	self->turbo_progress = 0.0f;
}

/* 0x007BCF90, UNVALIDATED: starts or refreshes a normal/roulette turbo. */
TMNF_HD void CSceneVehicleCar_EnableTurbo(
	CSceneVehicleCarAuxContext *self, uint32_t tick, uint32_t duration,
	float factor, TMNFVehicleTurboType type, uint32_t roulette_token)
{
	if (self->turbo_type != type) {
		self->turbo_start_tick = tick;
		if (self->turbo_sound_attached != 0) {
			if (self->play_turbo_sound == NULL)
				tmnf_abort();
			self->play_turbo_sound(self->runtime);
		}
		self->roulette_token = 0;
	}
	if (type == TMNF_TURBO_NORMAL) {
		self->turbo_factor = factor;
	} else if (type == TMNF_TURBO_ROULETTE
		&& self->roulette_token != roulette_token) {
		float value = CSceneVehicleCar_GetRouletteValue01(
			tick - self->turbo_epoch_tick, self->roulette_modulus);

		self->roulette_value = value;
		self->turbo_factor = x87_mul(
			CSceneVehicleCar_GetRouletteBoostFactorFromValue01(value),
			factor);
		self->roulette_token = roulette_token;
	}
	self->turbo_end_tick = tick + duration;
	self->turbo_type = type;
}

/* 0x007BD3F0, UNVALIDATED: integrates suspension and its surface transform. */
TMNF_HD void CSceneVehicleCar_WheelIntegrate(
	CSceneVehicleCarAuxContext *self, uint32_t wheel_index, float dt)
{
	CSceneVehicleCar *vehicle = self->vehicle;
	CSceneVehicleCarWheel *wheel = &vehicle->wheels[wheel_index];
	CSceneVehicleCarWheelAux *aux_wheel = &self->wheels[wheel_index];
	const CSceneVehicleCarTuning *tuning = vehicle->tuning;
	float absorb;
	float vertical_offset;
	float horizontal_zero;

	switch (tuning->suspension_model) {
	case 0: {
		float spring;
		float damping;
		float acceleration;
		float velocity;

		absorb = x87_sub(
			wheel->real_time.damper_absorb,
			wheel->real_time.field08);
		wheel->real_time.damper_absorb = absorb;
		wheel->real_time.field08 = 0.0f;
		aux_wheel->surface_location = aux_wheel->surface_source;
		spring = x87_mul(
			x87_sub(tuning->suspension_rest_length, absorb),
			tuning->suspension_stiffness);
		damping = x87_mul(
			tuning->suspension_damping,
			wheel->real_time.field04);
		acceleration = x87_sub(spring, damping);
		velocity = x87_add(
			x87_mul(acceleration, dt),
			wheel->real_time.field04);
		wheel->real_time.field04 = velocity;
		absorb = x87_add(x87_mul(velocity, dt), absorb);
		wheel->real_time.damper_absorb = absorb;
		vertical_offset = -absorb;
		horizontal_zero = x87_mul(vertical_offset, 0.0f);
		aux_wheel->surface_location.t[0] = x87_add(
			aux_wheel->surface_location.t[0], horizontal_zero);
		aux_wheel->surface_location.t[1] = x87_add(
			vertical_offset, aux_wheel->surface_location.t[1]);
		aux_wheel->surface_location.t[2] = x87_add(
			aux_wheel->surface_location.t[2], horizontal_zero);
		update_surface(self, wheel_index);
		return;
	}
	case 1:
	case 2: {
		float old_absorb = wheel->real_time.damper_absorb;
		float effective = x87_sub(
			old_absorb, wheel->real_time.field08);
		float delta = x87_sub(
			tuning->suspension_rest_length, effective);
		float correction = x87_mul(
			x87_mul(delta, dt),
			self->tuning->suspension_follow_rate);

		aux_wheel->surface_location = aux_wheel->surface_source;
		absorb = x87_add(correction, effective);
		wheel->real_time.field04 = x87_div(x87_sub(absorb, old_absorb), dt);
		wheel->real_time.damper_absorb = absorb;
		wheel->real_time.field08 = 0.0f;
		vertical_offset = -absorb;
		horizontal_zero = x87_mul(vertical_offset, 0.0f);
		aux_wheel->surface_location.t[0] = x87_add(
			horizontal_zero, aux_wheel->surface_location.t[0]);
		aux_wheel->surface_location.t[1] = x87_add(
			aux_wheel->surface_location.t[1], vertical_offset);
		aux_wheel->surface_location.t[2] = x87_add(
			aux_wheel->surface_location.t[2], horizontal_zero);
		update_surface(self, wheel_index);
		return;
	}
	default:
		update_surface(self, wheel_index);
		return;
	}
}

/* 0x007BF1D0, UNVALIDATED: applies local angular control while airborne. */
TMNF_HD void CSceneVehicleCar_ComputeAirControl(
	CSceneVehicleCarAuxContext *self, const GmVec3 *angular_speed,
	uint32_t tick, int suppress_torque, int reset)
{
	CSceneVehicleCar *vehicle = self->vehicle;
	const CSceneVehicleCarTuningAux *tuning = self->tuning;
	GmVec3 torque_direction = {
		-angular_speed->x,
		-angular_speed->y,
		-angular_speed->z,
	};
	GmVec3 controlled_speed;
	int reversed = 0;

	if ((vehicle->tuning->engine_model == 4
			|| vehicle->tuning->engine_model == 5)
		&& self->air_control_locked != 0)
		return;
	if (reset != 0) {
		self->air_control_tick = tick;
		self->air_control_speed = *angular_speed;
		goto apply_torque;
	}
	if (self->air_control_immediate != 0) {
		self->air_control_speed = *angular_speed;
		goto apply_torque;
	}
	if (tuning->air_control_window_ticks
			<= tick - self->air_control_tick)
		goto apply_torque;

	controlled_speed = *angular_speed;
	if ((F(AUX_EPSILON) < F(vehicle->input_steer)
			&& F(self->air_control_speed.y) < 0.0)
		|| (F(vehicle->input_steer) < F(-AUX_EPSILON)
			&& 0.0 < F(self->air_control_speed.y))) {
		if (F(tuning->air_reversal_threshold)
				< fabs(F(angular_speed->y))) {
			reversed = 1;
			self->air_control_speed.y = angular_speed->y;
		}
	} else {
		if ((F(AUX_EPSILON) < F(vehicle->input_steer)
				&& 0.0 < F(self->air_control_speed.y))
			|| (F(vehicle->input_steer) < F(-AUX_EPSILON)
				&& F(self->air_control_speed.y) < 0.0))
			reversed = 1;
		self->air_control_speed.y = angular_speed->y;
	}
	controlled_speed.y = self->air_control_speed.y;

	if (vehicle->tuning->engine_model == 4
		|| vehicle->tuning->engine_model == 5) {
		if (F(AUX_EPSILON) < F(vehicle->input_brake)
			&& 0.0 < F(self->air_control_speed.x))
			self->air_control_speed.x = 0.0f;
		else
			self->air_control_speed.x = angular_speed->x;
		controlled_speed.x = self->air_control_speed.x;
	}
	if (reversed != 0) {
		torque_direction.x = x87_mul(torque_direction.x, 3.0f);
		torque_direction.y = x87_mul(torque_direction.y, 3.0f);
		torque_direction.z = x87_mul(torque_direction.z, 3.0f);
	}
	if (suppress_torque == 0) {
		float curve_value;

		if (tuning->air_vertical_curve == NULL)
			tmnf_abort();
		curve_value = aux_curve_get_value(
			tuning->air_vertical_curve,
			(float)fabs(F(angular_speed->z)));
		torque_direction.z = x87_mul(
			torque_direction.z, curve_value);
	}
	set_local_angular_speed(vehicle, &controlled_speed);

apply_torque:
	if (suppress_torque == 0) {
		float xy_squared = x87_add(
			x87_mul(torque_direction.y, torque_direction.y),
			x87_mul(torque_direction.x, torque_direction.x));
		float length_squared = x87_add(
			xy_squared,
			x87_mul(torque_direction.z, torque_direction.z));
		float length = x87_sqrt(length_squared);

		if (F(AUX_EPSILON) <= F(length)) {
			float inverse_length = x87_rcp(length);
			float quadratic_term;
			float linear_term;
			float magnitude;
			GmVec3 torque;

			torque.x = x87_mul(
				inverse_length, torque_direction.x);
			torque.y = x87_mul(
				inverse_length, torque_direction.y);
			torque.z = x87_mul(
				inverse_length, torque_direction.z);
			quadratic_term = x87_mul(
				x87_mul(tuning->air_torque_quadratic, length),
				length);
			linear_term = x87_mul(
				length, tuning->air_torque_linear);
			magnitude = x87_add(quadratic_term, linear_term);
			torque.x = x87_mul(magnitude, torque.x);
			torque.y = x87_mul(torque.y, magnitude);
			torque.z = x87_mul(magnitude, torque.z);
			CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
		}
	}
}

/* 0x004FF950 GmMap2<unsigned char>::IsInside and 0x004FFAC0 GetValue share
 * one coordinate schedule: (p - origin) / cell, truncating fistp to int64,
 * then an unsigned compare of the low 32 bits against the extent. */
TMNF_HD static uint32_t water_map_coordinate(float value, float origin, float cell)
{
	float offset = x87_sub(value, origin);
	float scaled = x87_div(offset, cell);
	return (uint32_t)(int64_t)F(scaled);
}

TMNF_HD static int water_map_is_inside(
	const TmnfTrackWater *map, float x, float z)
{
	uint32_t ix = water_map_coordinate(x, map->origin_x, map->cell_x);
	uint32_t iz = water_map_coordinate(z, map->origin_z, map->cell_z);
	return ix < map->width && iz < map->height;
}

TMNF_HD static uint8_t water_map_value(const TmnfTrackWater *map, float x, float z)
{
	uint32_t ix = water_map_coordinate(x, map->origin_x, map->cell_x);
	uint32_t iz = water_map_coordinate(z, map->origin_z, map->cell_z);
	if (ix >= map->width || iz >= map->height)
		return map->default_cell;
	return map->cells[iz * map->width + ix];
}

/* (y*y + x*x) + z*z with every operation rounded, as at 0x007C2BEF and
 * 0x007C2CE3. */
TMNF_HD static float water_length_squared(const GmVec3 *v)
{
	return x87_add(
		x87_add(x87_mul(v->y, v->y), x87_mul(v->x, v->x)),
		x87_mul(v->z, v->z));
}

/* 0x007F3F40 CSceneVehicleCarTuning::GetWaterFrictionFromSpeed: speed times
 * float-widened 3.6 (0x00B3D2A8), then curve +0x218 from key zero. */
TMNF_HD static float water_friction_from_speed(
	const CSceneVehicleCarTuningAux *tuning, float speed)
{
	float kmh = x87_r24(F(speed) * 0x1.ccccccp+1);
	uint32_t lower_index = 0;
	float value;

	CFuncKeysReal_GetValueOut(
		tuning->water_friction_curve, kmh, &value, &lower_index);
	return value;
}

/* 0x007BE690 CSceneVehicleCar::AddVehicleImpulse(GmVec3 const&). */
TMNF_HD static void add_vehicle_central_impulse(
	CSceneVehicleCar *vehicle, const GmVec3 *impulse)
{
	CHmsItem_AddImpulse((CHmsItem *)vehicle->hms_item, impulse);
	vehicle->total_impulse_added.x = x87_add(
		vehicle->total_impulse_added.x, impulse->x);
	vehicle->total_impulse_added.y = x87_add(
		impulse->y, vehicle->total_impulse_added.y);
	vehicle->total_impulse_added.z = x87_add(
		impulse->z, vehicle->total_impulse_added.z);
}

/*
 * 0x007C2910 CSceneVehicleCar::ApplyWaterForces. Returns 1 when the water
 * drag/buoyancy force and torque were applied, 0 otherwise (including the
 * tick on which the entry impulse is applied).
 *
 * 0x007C97A0 CSceneVehicle::WaterSplash (car +0x1f8, +0x204..+0x20c,
 * +0xb8) only records the splash for audio/visual playback and is not
 * represented; none of those words are physics inputs or oracle fields.
 */
TMNF_HD int CSceneVehicleCar_ApplyWaterForces(
	CSceneVehicleCarAuxContext *self, const GmVec3 *existing_force)
{
	CSceneVehicleCar *vehicle = self->vehicle;
	const CSceneVehicleCarTuningAux *tuning = self->tuning;
	const TmnfTrackWater *map = tuning->water_map;
	const GmMat3 *rotation = &vehicle->dyna_state->rot;
	GmBoxAligned box;
	float half_y;
	float top;
	float bottom;
	float depth;
	GmVec3 local_speed;
	GmVec3 world_speed;
	GmVec3 local_angular;
	GmVec3 drag = { 0.0f, 0.0f, 0.0f };
	GmVec3 torque;
	GmVec3 buoyancy;
	GmVec3 force;
	float horizontal_squared;
	float threshold;
	float speed;
	float angular_speed;
	float scale;

	if (map == NULL)
		tmnf_abort();
	/* 0x007C2939: GmBoxAligned::SetMult(car+0x1dc, corpus location). */
	GmBoxAligned_SetMult(
		&box, &self->body_box, (const GmIso4 *)rotation);
	half_y = fabsf(box.half_extent.y);
	top = x87_add(box.center.y, half_y);
	bottom = x87_sub(box.center.y, half_y);

	/* 0x007C29B3..0x007C2A2A: map gate. */
	if (water_map_is_inside(map, box.center.x, box.center.z) != 0
		|| map->default_cell != 1
		|| F(map->level) <= F(bottom)) {
		if (F(top) <= F(map->floor))
			return 0;
		if (F(map->level) <= F(bottom))
			return 0;
		if (water_map_value(map, box.center.x, box.center.z) != 1)
			return 0;
	}
	depth = x87_sub(map->level, bottom);
	if (!(0.5f < F(depth)))
		return 0;

	CHmsItem_GetLinearSpeed((CHmsItem *)vehicle->hms_item, &local_speed);
	GmVec3_SetMult_Mat3(&world_speed, &local_speed, rotation);
	horizontal_squared = x87_add(
		x87_mul(world_speed.x, world_speed.x),
		x87_mul(world_speed.z, world_speed.z));
	threshold = tuning->water_entry_speed_threshold;

	/* 0x007C2A8C..0x007C2AEC: entry-impulse eligibility. 0x00B41EA8 is
	 * float-widened 0.9; 0x00B574FC is -1e-05f. */
	if (self->air_control_immediate == 0
		&& F(depth) < 0x1.ccccccp-1
		&& F(x87_sub(map->level, top)) < 0.0
		&& F(world_speed.y) < -0x1.4f8b58p-17f) {
		float ratio;
		uint32_t lower_index;
		int apply_impulse;

		if (F(horizontal_squared) <= F(x87_mul(threshold, threshold))) {
			/* 0x007C2E0D: slow horizontal entry needs total speed above
			 * the +0x20c minimum. */
			float minimum = tuning->water_entry_speed_minimum;
			float speed_squared = water_length_squared(&local_speed);
			float minimum_squared = x87_mul(minimum, minimum);

			apply_impulse = F(minimum_squared) < F(speed_squared);
			ratio = 0.0f;
			memcpy(&lower_index, &minimum_squared, 4);
		} else {
			float horizontal = x87_sqrt(horizontal_squared);

			ratio = x87_div(-horizontal, world_speed.y);
			apply_impulse = !(F(ratio) < 0.0);
			memcpy(&lower_index, &horizontal, 4);
		}
		if (apply_impulse) {
			/* 0x007C2B35..0x007C2BD9. Both curve lookups start from the
			 * stale float bits left in the index slot, exactly as the
			 * game does. */
			float vertical;
			float horizontal;
			GmVec3 impulse;

			CFuncKeysReal_GetValueOut(
				tuning->water_impulse_vertical_curve, ratio,
				&vertical, &lower_index);
			CFuncKeysReal_GetValueOut(
				tuning->water_impulse_horizontal_curve, ratio,
				&horizontal, &lower_index);
			impulse.x = x87_mul(-horizontal, world_speed.x);
			impulse.y = x87_mul(-vertical, world_speed.y);
			impulse.z = x87_mul(-horizontal, world_speed.z);
			GmVec3_MultTranspose(&impulse, rotation);
			add_vehicle_central_impulse(vehicle, &impulse);
			/* 0x007C2BDE: an applied impulse ends the call with 0. */
			return 0;
		}
	}

	/* 0x007C2BEF..0x007C2E0A: drag, angular drag, buoyancy. */
	speed = x87_sqrt(water_length_squared(&local_speed));
	if (0x1.4f8b58p-17f < F(speed)) {
		scale = -water_friction_from_speed(tuning, speed);
		drag.x = x87_mul(scale, local_speed.x);
		drag.y = x87_mul(local_speed.y, scale);
		drag.z = x87_mul(scale, local_speed.z);
	}
	CHmsItem_GetAngularSpeed((CHmsItem *)vehicle->hms_item, &local_angular);
	scale = -tuning->water_angular_drag_linear;
	torque.x = x87_mul(scale, local_angular.x);
	torque.y = x87_mul(local_angular.y, scale);
	torque.z = x87_mul(scale, local_angular.z);
	angular_speed = x87_sqrt(water_length_squared(&local_angular));
	scale = x87_mul(-angular_speed, tuning->water_angular_drag_quadratic);
	torque.x = x87_add(x87_mul(scale, local_angular.x), torque.x);
	torque.y = x87_add(x87_mul(local_angular.y, scale), torque.y);
	torque.z = x87_add(x87_mul(scale, local_angular.z), torque.z);
	buoyancy.x = 0.0f;
	buoyancy.y = -tuning->water_buoyancy;
	buoyancy.z = 0.0f;
	GmVec3_MultTranspose(&buoyancy, rotation);
	force.x = x87_sub(x87_add(buoyancy.x, drag.x), existing_force->x);
	force.y = x87_sub(x87_add(buoyancy.y, drag.y), existing_force->y);
	force.z = x87_sub(x87_add(buoyancy.z, drag.z), existing_force->z);
	CSceneVehicleCar_AddVehicleCentralForce(vehicle, &force);
	CSceneVehicleCar_AddVehicleTorque(vehicle, &torque);
	return 1;
}

/* 0x007C3900, UNVALIDATED: integrates dry A01 wheel, engine, and steer state. */
TMNF_HD void CSceneVehicleCar_IntegrateVehicle(
	CSceneVehicleCarAuxContext *self, float dt)
{
	CSceneVehicleCar *vehicle = self->vehicle;
	const CSceneVehicleCarTuningAux *tuning = self->tuning;
	GmVec3 local_speed = vehicle->dyna_state->linVel;

	GmVec3_MultTranspose(&local_speed, &vehicle->dyna_state->rot);
	if ((self->integration_flags & 1u) != 0) {
		float absolute_speed = (float)fabs(F(local_speed.z));
		float steering_scale = x87_add(
			x87_mul(absolute_speed, tuning->steering_speed_scale),
			tuning->steering_speed_base);

		for (uint32_t i = 0; i < vehicle->wheel_count; ++i) {
			CSceneVehicleCarWheel *wheel = &vehicle->wheels[i];
			GmMat3 source_matrix;

			memcpy(
				source_matrix.m, self->wheels[i].surface_source.m,
				sizeof(source_matrix.m));
			GmMat3_Set(&wheel->real_time.basis0, &source_matrix);
			if (wheel->steerable == 0) {
				wheel->real_time.blend_target = 0.0f;
			} else {
				float rotation = 0.0f;
				float degrees = 30.0f;
				float radians;

				if (F(AUX_EPSILON) <= F(steering_scale))
					rotation = x87_div(-self->steering_value, steering_scale);
				mat3_rotate_y(
					&wheel->real_time.basis0, rotation);
				if (tuning->steering_angle_curve != NULL) {
					/* 0x007C3A2E: 0x00B3D2A8, float 3.6
					 * promoted to double. */
					float speed_kmh = x87_r24(
						F(absolute_speed)
						* 0x1.ccccccp+1);

					degrees = aux_curve_get_value(
						tuning->steering_angle_curve,
						speed_kmh);
				}
				/* 0x007C3A54: 0x00B36110, float pi promoted to
				 * double; 0x007C3A5A divides by 0x00B36AB8 (180). */
				radians = x87_r24(
					F(degrees)
					* 0x1.921fb6p+1);
				radians = x87_r24(F(radians) / 180.0);
				wheel->real_time.blend_target = x87_mul(
					-self->steering_value, radians);
			}
			CSceneVehicleCar_WheelUpdateSpeedFromVehicleSpeed(
				vehicle, wheel, local_speed.z, dt);
			CSceneVehicleCarWheelRealTimeState_Integrate(
				&wheel->real_time, dt);
		}
	}
	if ((self->integration_flags & 2u) != 0) {
		for (uint32_t i = 0; i < vehicle->wheel_count; ++i)
			CSceneVehicleCar_WheelIntegrate(self, i, dt);
	}
	if ((self->integration_flags & 4u) != 0) {
		if (vehicle->flag_60c != 0) {
			vehicle->engine.rpm = 0.0f;
		} else {
			float throttle = vehicle->engine.reverse != 0
				? vehicle->input_brake
				: vehicle->input_gas;

			CSceneVehicleCar_EngineIntegrate(vehicle, throttle, dt);
		}
	}

	if (0.0 < F(tuning->steering_slew_rate)) {
		float direction =
			0.0 <= F(x87_sub(
				self->steering_value, vehicle->input_steer))
			? -1.0f
			: 1.0f;
		float next = x87_add(
			x87_mul(
				x87_mul(tuning->steering_slew_rate, direction),
				dt),
			self->steering_value);

		if (F(vehicle->input_steer) <= F(self->steering_value)) {
			if (F(next) < F(vehicle->input_steer))
				next = vehicle->input_steer;
		} else if (F(vehicle->input_steer) < F(next)) {
			next = vehicle->input_steer;
		}
		self->steering_value = next;
	} else {
		self->steering_value = vehicle->input_steer;
	}

	if (self->finish_integration == NULL)
		tmnf_abort();
	self->finish_integration(self->runtime, vehicle);
}

/* 0x007C3CC8..0x007C3D1B (X) and 0x007C3D27..0x007C3D78 (Z): fmod by the
 * material period, float store, fabs, divide by the period, multiply by the
 * integer image extent, then truncating fistp. */
TMNF_HD static uint32_t fake_contact_mask_coordinate(
	float value, float period, uint32_t extent)
{
	float wrapped = x87_r24(fmod(F(value), F(period)));
	float fraction = x87_div(fabsf(wrapped), period);
	float scaled = x87_mul(fraction, (float)extent);
	uint32_t coordinate = (uint32_t)ftol(F(scaled));

	if (coordinate >= extent)
		tmnf_abort();
	return coordinate;
}

/* 0x007C3C00 */
TMNF_HD void CSceneVehicleCar_CreateFakeContacts(
	CSceneVehicleCarAuxContext *self,
	TMNFVehicleContactContext *contact)
{
	CSceneVehicleCar *vehicle = self->vehicle;
	GmVec3 local_speed;

	if (contact == NULL || contact->vehicle != vehicle
		|| self->wheel_count != vehicle->wheel_count
		|| contact->wheel_count != vehicle->wheel_count) {
		tmnf_abort();
	}
	CHmsItem_GetLinearSpeed((CHmsItem *)vehicle->hms_item, &local_speed);
	for (uint32_t i = 0; i < vehicle->wheel_count; ++i) {
		CSceneVehicleCarWheel *wheel = &vehicle->wheels[i];
		uint16_t material_id =
			(uint16_t)wheel->real_time.contact_material_id;
		const TMNFVehicleGroundMaterial *material;
		GmVec3 local_position;
		GmVec3 world_position;
		uint32_t x;
		uint32_t y;
		uint8_t mask_value;
		float impact_speed;
		CHmsPhysicalContact fake_contact;

		if (wheel->real_time.has_ground_contact == 0)
			continue;
		if (material_id >= contact->ground_material_index_count)
			tmnf_abort();
		material = contact->ground_materials[
			contact->ground_material_indices[material_id]];
		/* 0x007C3C7E..0x007C3C83: a material without a bump mask ends
		 * the whole pass; later wheels are not visited. */
		if (material->fake_contact_mask == NULL)
			return;
		local_position.x = self->wheels[i].surface_source.t[0];
		local_position.y = self->wheels[i].surface_source.t[1];
		local_position.z = self->wheels[i].surface_source.t[2];
		GmVec3_SetMult_Iso4(
			&world_position, &local_position,
			(const GmIso4 *)&vehicle->dyna_state->rot);
		x = fake_contact_mask_coordinate(
			world_position.x, material->fake_contact_period_x,
			TMNF_FAKE_CONTACT_MASK_WIDTH);
		y = fake_contact_mask_coordinate(
			world_position.z, material->fake_contact_period_z,
			TMNF_FAKE_CONTACT_MASK_HEIGHT);
		mask_value = material->fake_contact_mask[
			y * TMNF_FAKE_CONTACT_MASK_WIDTH + x];
		if (mask_value == 0)
			continue;

		/* 0x007C3D9A..0x007C3DC5 */
		impact_speed = x87_r24((double)mask_value / 255.0);
		impact_speed = x87_mul(impact_speed, local_speed.z);
		impact_speed = x87_mul(
			impact_speed, material->fake_contact_impulse_scale);
		if (F(material->fake_contact_impulse_limit) < F(impact_speed))
			impact_speed = material->fake_contact_impulse_limit;

		memset(&fake_contact, 0, sizeof(fake_contact));
		fake_contact.normal.y = 1.0f;
		fake_contact.position = local_position;
		fake_contact.relative_speed.y = -impact_speed;
		fake_contact.other_surface_material = material_id;
		CSceneVehicleCar_WheelAbsorbContact(
			contact, &contact->wheels[i], &fake_contact);
	}
}
