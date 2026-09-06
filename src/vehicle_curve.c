#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "tmnf_fp.h"
#include "vehicle_curve.h"

/* DAT_00b36288 and DAT_00b5f068 in the curve functions. */
#define CURVE_BOUNDARY_EPSILON 0x1.4f8b588e368f1p-17
#define CURVE_BLEND_EPSILON 0x1.4f8b6p-17f

/* DAT_00b3d2a8 in the tuning getters. */
#define METERS_PER_SECOND_TO_KMH 0x1.ccccccp+1

TMNF_HD static float curve_lower_boundary(float position)
{
	return x87_r24(F(position) - CURVE_BOUNDARY_EPSILON);
}

TMNF_HD static float curve_upper_boundary(float position)
{
	return x87_r24(F(position) + CURVE_BOUNDARY_EPSILON);
}

void CFuncKeys_Compile(CFuncKeys *keys)
{
	float *lower = (float *)malloc(2 * (size_t)keys->count * sizeof(float));
	if (lower == NULL && keys->count != 0)
		abort();
	float *upper = lower + keys->count;
	for (uint32_t i = 0; i < keys->count; ++i) {
		lower[i] = curve_lower_boundary(keys->positions[i]);
		upper[i] = curve_upper_boundary(keys->positions[i]);
	}
	keys->lower_bounds = lower;
	keys->upper_bounds = upper;
}

/* Same values into storage the caller has already linked through
 * lower_bounds (2 * count floats: lower then upper). */
void CFuncKeys_CompileInto(CFuncKeys *keys)
{
	float *lower = (float *)(uintptr_t)keys->lower_bounds;
	float *upper = lower + keys->count;
	for (uint32_t i = 0; i < keys->count; ++i) {
		lower[i] = curve_lower_boundary(keys->positions[i]);
		upper[i] = curve_upper_boundary(keys->positions[i]);
	}
	keys->upper_bounds = upper;
}

void CFuncKeys_Release(CFuncKeys *keys)
{
	free((float *)keys->lower_bounds);
	keys->lower_bounds = NULL;
	keys->upper_bounds = NULL;
}

/*
 * 0x005914C0, UNVALIDATED: locate the adjacent position keys, searching in
 * the direction selected by the caller.
 */
TMNF_HD void CFuncKeys_GetBoundingIndices(
	const CFuncKeys *self, float position, uint32_t *lower_index,
	uint32_t *upper_index, int forward)
{
	uint32_t count = self->count;
	uint32_t search_index;
	uint32_t lower;
	uint32_t upper;
	uint32_t attempts;

	if (count == 0) {
		*lower_index = UINT32_MAX;
		*upper_index = UINT32_MAX;
		return;
	}
	if (count == 1) {
		*lower_index = 0;
		*upper_index = 0;
		return;
	}
	const float *lower_bounds = self->lower_bounds;
	const float *upper_bounds = self->upper_bounds;
	if (position < lower_bounds[0]) {
		*lower_index = 0;
		*upper_index = 0;
		return;
	}

	lower = count - 1;
	if (upper_bounds[lower] < position) {
		*lower_index = lower;
		*upper_index = lower;
		return;
	}

	search_index = *lower_index;
	attempts = 0;
	if (forward) {
		if (search_index >= count)
			search_index = 0;
		for (;;) {
			lower = search_index;
			search_index++;
			if (search_index >= count)
				search_index = 0;
			upper = search_index;
			attempts++;
			if (attempts > count)
				break;
			if (!(lower_bounds[lower] <= position) ||
			    !(position <= upper_bounds[upper]))
				continue;
			break;
		}
	} else {
		search_index++;
		if ((search_index & UINT32_C(0x80000000)) != 0)
			search_index = 0;
		for (;;) {
			upper = search_index;
			search_index--;
			if ((search_index & UINT32_C(0x80000000)) != 0)
				search_index = count;
			lower = search_index;
			attempts++;
			if (attempts > count)
				break;
			if (!(lower_bounds[lower] <= position) ||
			    !(position <= upper_bounds[upper]))
				continue;
			break;
		}
	}

	*lower_index = lower;
	*upper_index = upper;
}

/*
 * 0x00591670, UNVALIDATED: locate the adjacent keys and compute the normalized
 * position between them.
 */
TMNF_HD int CFuncKeys_ComputeBlendCoef(
	const CFuncKeys *self, float position, uint32_t *lower_index,
	uint32_t *upper_index, float *blend, int forward)
{
	float lower_position;
	float difference;
	float magnitude;
	float numerator;

	CFuncKeys_GetBoundingIndices(
		self, position, lower_index, upper_index, forward);
	if (*lower_index == UINT32_MAX)
		return 0;
	if (*lower_index == *upper_index) {
		*blend = 0.0f;
		return 1;
	}

	lower_position = self->positions[*lower_index];
	difference = x87_sub(self->positions[*upper_index], lower_position);
	magnitude = fabsf(difference);
	if (F(magnitude) < F(CURVE_BLEND_EPSILON)) {
		*blend = 0.0f;
		return 1;
	}

	numerator = x87_sub(position, lower_position);
	*blend = x87_div(numerator, difference);
	return 1;
}

/*
 * 0x00585E70, UNVALIDATED: evaluate a real-valued curve using the selected
 * interpolation mode.
 */
TMNF_HD void CFuncKeysReal_GetRealAt(
	const CFuncKeysReal *self, float position, float *value,
	uint32_t *lower_index, uint32_t *upper_index, float *blend,
	int32_t interpolation, int forward)
{
	float lower_weight;
	float lower_value;
	float upper_value;

	if (!CFuncKeys_ComputeBlendCoef(
		    &self->keys, position, lower_index, upper_index, blend,
		    forward)) {
		*value = 0.0f;
		return;
	}
	if (interpolation == 1) {
		*value = self->values[*lower_index];
		return;
	}

	lower_weight = x87_r24(1.0 - F(*blend));
	lower_value = x87_mul(lower_weight, self->values[*lower_index]);
	upper_value = x87_mul(*blend, self->values[*upper_index]);
	*value = x87_add(lower_value, upper_value);
}

TMNF_HD static void cfunc_keys_real_get_value_out_interp(
	const CFuncKeysReal *self, float position, float *value,
	uint32_t *lower_index, int32_t interpolation)
{
	uint32_t upper_index = *lower_index + 1;
	float blend;

	CFuncKeysReal_GetRealAt(
		self, position, value, lower_index, &upper_index, &blend,
		interpolation, 1);
}

/*
 * 0x00586200, UNVALIDATED: evaluate a curve and update the caller's cached
 * lower index.
 */
TMNF_HD void CFuncKeysReal_GetValueOut(
	const CFuncKeysReal *self, float position, float *value,
	uint32_t *lower_index)
{
	cfunc_keys_real_get_value_out_interp(
		self, position, value, lower_index, self->interpolation);
}

/*
 * 0x00586240, UNVALIDATED: return a curve value with an optional cached lower
 * index.
 */
TMNF_HD float CFuncKeysReal_GetValue(
	const CFuncKeysReal *self, float position, uint32_t *lower_index)
{
	uint32_t local_lower_index = lower_index == NULL ? 0 : *lower_index;
	uint32_t upper_index = local_lower_index + 1;
	uint32_t *selected_lower_index =
		lower_index == NULL ? &local_lower_index : lower_index;
	float blend;
	float value;

	CFuncKeysReal_GetRealAt(
		self, position, &value, selected_lower_index, &upper_index,
		&blend, self->interpolation, 1);
	return value;
}

TMNF_HD static float tuning_speed_position(float speed)
{
	return x87_r24(F(speed) * METERS_PER_SECOND_TO_KMH);
}

TMNF_HD static float tuning_get_value(const CFuncKeysReal *curve, float position)
{
	uint32_t lower_index = 0;
	float value;

	CFuncKeysReal_GetValueOut(curve, position, &value, &lower_index);
	return value;
}

TMNF_HD static float tuning_get_step_value(const CFuncKeysReal *curve, float position)
{
	uint32_t lower_index = 0;
	float value;

	cfunc_keys_real_get_value_out_interp(
		curve, position, &value, &lower_index, 1);
	return value;
}

/*
 * 0x007F3BE0, UNVALIDATED: evaluate the acceleration curve after converting
 * speed to km/h.
 */
TMNF_HD float CSceneVehicleCarTuning_GetAccelFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_step_value(
		self->accel_from_speed, tuning_speed_position(speed));
}

/*
 * 0x007F3C30, UNVALIDATED: evaluate lateral rollover against speed in km/h.
 */
TMNF_HD float CSceneVehicleCarTuning_GetRolloverLateralFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_value(
		self->rollover_lateral_from_speed, tuning_speed_position(speed));
}

/*
 * 0x007F3C70, UNVALIDATED: evaluate maximum side friction against speed in
 * km/h.
 */
TMNF_HD float CSceneVehicleCarTuning_GetMaxSideFrictionFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_value(
		self->max_side_friction_from_speed, tuning_speed_position(speed));
}

/*
 * 0x007F3CB0, UNVALIDATED: evaluate lateral contact slowdown against speed in
 * km/h.
 */
TMNF_HD float CSceneVehicleCarTuning_GetLateralContactSlowDownFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_step_value(
		self->lateral_contact_slowdown_from_speed,
		tuning_speed_position(speed));
}

/*
 * 0x007F3D00, UNVALIDATED: evaluate steering slowdown against speed in km/h.
 */
TMNF_HD float CSceneVehicleCarTuning_GetSteerSlowDownFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_step_value(
		self->steer_slowdown_from_speed, tuning_speed_position(speed));
}

/*
 * 0x007F3D50, UNVALIDATED: evaluate lateral rollover directly against angle.
 */
TMNF_HD float CSceneVehicleCarTuning_GetRolloverLateralCoefFromAngle(
	const CSceneVehicleCarTuningCurveSet *self, float angle)
{
	return tuning_get_value(self->rollover_lateral_coef_from_angle, angle);
}

/*
 * 0x007F3D80, UNVALIDATED: evaluate steering drive torque against speed in
 * km/h.
 */
TMNF_HD float CSceneVehicleCarTuning_GetSteerDriveTorqueFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_value(
		self->steer_drive_torque_from_speed,
		tuning_speed_position(speed));
}

/*
 * 0x007F3DC0, UNVALIDATED: evaluate the Model 4 steering radius curve.
 */
TMNF_HD float CSceneVehicleCarTuning_M4GetSteerRadiusFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_value(
		self->m4_steer_radius_from_speed, tuning_speed_position(speed));
}

/*
 * 0x007F3E00, UNVALIDATED: evaluate the Model 4 maximum friction curve.
 */
TMNF_HD float CSceneVehicleCarTuning_M4GetMaxFrictionForceFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_value(
		self->m4_max_friction_force_from_speed,
		tuning_speed_position(speed));
}

/*
 * 0x007F3E40, UNVALIDATED: evaluate the Model 5 acceleration curve.
 */
TMNF_HD float CSceneVehicleCarTuning_M5GetAccelFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_value(
		self->accel_from_speed, tuning_speed_position(speed));
}

/*
 * 0x007F3E80, UNVALIDATED: evaluate and scale Model 5 slipping acceleration.
 */
TMNF_HD float CSceneVehicleCarTuning_M5GetSlippingAccelFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	float value = tuning_get_value(
		self->m5_slipping_accel_from_speed,
		tuning_speed_position(speed));

	return x87_mul(self->m5_slipping_accel_scale, value);
}

/*
 * 0x007F3ED0, UNVALIDATED: evaluate Model 5 steering slowdown.
 */
TMNF_HD float CSceneVehicleCarTuning_M5GetSteerSlowDownFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_value(
		self->steer_slowdown_from_speed, tuning_speed_position(speed));
}

/*
 * 0x007F3F10, UNVALIDATED: evaluate Model 5 lateral contact slowdown.
 */
TMNF_HD float CSceneVehicleCarTuning_M5GetLateralContactSlowDownFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return CFuncKeysReal_GetValue(
		self->lateral_contact_slowdown_from_speed,
		tuning_speed_position(speed), NULL);
}

/*
 * 0x007F3F40, UNVALIDATED: evaluate water friction against speed in km/h.
 */
TMNF_HD float CSceneVehicleCarTuning_GetWaterFrictionFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_value(
		self->water_friction_from_speed, tuning_speed_position(speed));
}

/*
 * 0x007F3F80, UNVALIDATED: normalize damper absorption and evaluate its
 * modulation curve.
 */
TMNF_HD float CSceneVehicleCarTuning_M6GetModulationFromDamperAbsorbVal(
	const CSceneVehicleCarTuningCurveSet *self, float damper_absorb)
{
	float position;

	if (F(self->damper_min) == F(self->damper_max)) {
		position = 0.0f;
	} else {
		float numerator = x87_sub(damper_absorb, self->damper_min);
		float denominator = x87_sub(self->damper_max, self->damper_min);

		position = x87_div(numerator, denominator);
	}
	return tuning_get_value(self->m6_damper_modulation, position);
}

/*
 * 0x007F3FF0, UNVALIDATED: evaluate rear-gear acceleration against speed.
 */
TMNF_HD float CSceneVehicleCarTuning_M6GetRearGearAccelFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_value(
		self->m6_rear_gear_accel_from_speed,
		tuning_speed_position(speed));
}

/*
 * 0x007F4030, UNVALIDATED: evaluate burnout radius against speed.
 */
TMNF_HD float CSceneVehicleCarTuning_M6GetBurnoutRadiusFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_value(
		self->m6_burnout_radius_from_speed, tuning_speed_position(speed));
}

/*
 * 0x007F4070, UNVALIDATED: evaluate burnout lateral speed and convert it from
 * km/h to m/s.
 */
TMNF_HD float CSceneVehicleCarTuning_M6GetLateralSpeedFromBurnoutRadius(
	const CSceneVehicleCarTuningCurveSet *self, float radius)
{
	float value = tuning_get_value(
		self->m6_lateral_speed_from_burnout_radius, radius);

	return x87_r24(F(value) / METERS_PER_SECOND_TO_KMH);
}

/*
 * 0x007F40B0, UNVALIDATED: evaluate burnout rollover against speed.
 */
TMNF_HD float CSceneVehicleCarTuning_M6GetBurnoutRolloverFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_value(
		self->m6_burnout_rollover_from_speed,
		tuning_speed_position(speed));
}

/*
 * 0x007F40F0, UNVALIDATED: evaluate donut rollover against speed.
 */
TMNF_HD float CSceneVehicleCarTuning_M6GetDonutRolloverFromSpeed(
	const CSceneVehicleCarTuningCurveSet *self, float speed)
{
	return tuning_get_value(
		self->m6_donut_rollover_from_speed,
		tuning_speed_position(speed));
}

/*
 * 0x007F4130, UNVALIDATED: evaluate lateral rollover against speed ratio.
 */
TMNF_HD float CSceneVehicleCarTuning_M6GetRolloverLateralFromSpeedRatio(
	const CSceneVehicleCarTuningCurveSet *self, float speed_ratio)
{
	return tuning_get_value(
		self->m6_rollover_lateral_from_speed_ratio,
		tuning_speed_position(speed_ratio));
}
