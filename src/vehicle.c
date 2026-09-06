#include <math.h>
#include <stdlib.h>

#include "tmnf_fp.h"
#include "vehicle.h"

#define WHEEL_ANGLE_PERIOD 0x1.921fb6p+10f
#define NORMALIZE_EPSILON 0x1.b7cdfcp-34f
#define WHEEL_SPEED_EPSILON 0x1.4f8b58p-17f
#define WHEEL_SPEED_DAMPING 0x1.fd70a4p-1

TMNF_HD static float gm_mod(float value, float lower, float upper)
{
	float period;
	float relative;
	float result;

	if (F(lower) < F(value) && F(value) < F(upper))
		return value;
	period = x87_sub(upper, lower);
	relative = x87_sub(value, lower);
	result = x87_r24(fmod(F(relative), F(period)));
	if (F(result) < 0.0)
		result = x87_add(result, period);
	return x87_add(result, lower);
}

TMNF_HD static float vec_length_squared(const GmVec3 *value)
{
	return x87_add(
		x87_add(x87_mul(value->x, value->x), x87_mul(value->y, value->y)),
		x87_mul(value->z, value->z));
}

TMNF_HD static void normalize_if_nonzero(GmVec3 *value)
{
	float length_squared = vec_length_squared(value);

	if (F(NORMALIZE_EPSILON) < F(length_squared)) {
		float length = x87_sqrt(length_squared);
		float inverse_length = x87_rcp(length);
		value->x = x87_mul(inverse_length, value->x);
		value->y = x87_mul(value->y, inverse_length);
		value->z = x87_mul(inverse_length, value->z);
	}
}

TMNF_HD static void mat3_set_up_v_and_dov(
	GmMat3 *matrix, const GmVec3 *up_value, const GmVec3 *dov)
{
	GmVec3 line0;
	GmVec3 line1 = *up_value;
	GmVec3 line2;

	line0.x = x87_sub(
		x87_mul(up_value->y, dov->z), x87_mul(up_value->z, dov->y));
	line0.y = x87_sub(
		x87_mul(up_value->z, dov->x), x87_mul(up_value->x, dov->z));
	line0.z = x87_sub(
		x87_mul(up_value->x, dov->y), x87_mul(dov->x, up_value->y));
	normalize_if_nonzero(&line0);
	normalize_if_nonzero(&line1);
	line2.x = x87_sub(
		x87_mul(line0.y, line1.z), x87_mul(line0.z, line1.y));
	line2.y = x87_sub(
		x87_mul(line1.x, line0.z), x87_mul(line0.x, line1.z));
	line2.z = x87_sub(
		x87_mul(line0.x, line1.y), x87_mul(line0.y, line1.x));
	matrix->m[0] = line0.x;
	matrix->m[3] = line0.y;
	matrix->m[6] = line0.z;
	matrix->m[1] = line1.x;
	matrix->m[4] = line1.y;
	matrix->m[7] = line1.z;
	matrix->m[2] = line2.x;
	matrix->m[5] = line2.y;
	matrix->m[8] = line2.z;
}

TMNF_HD static void add_vec3(GmVec3 *sum, const GmVec3 *value)
{
	sum->x = x87_add(sum->x, value->x);
	sum->y = x87_add(value->y, sum->y);
	sum->z = x87_add(value->z, sum->z);
}

/* 0x007BFD80 */
TMNF_HD void CSceneVehicle_VehicleInputSteerSet(
	CSceneVehicleCar *self, float value)
{
	self->input_steer = value;
}

/* 0x007BFD90 */
TMNF_HD void CSceneVehicle_VehicleInputGasSet(
	CSceneVehicleCar *self, float value)
{
	self->input_gas = value;
}

/* 0x007BFDA0 */
TMNF_HD void CSceneVehicle_VehicleInputBrakeSet(
	CSceneVehicleCar *self, float value)
{
	self->input_brake = value;
}

/* 0x004FE500  Maps timestamped digital/analog race inputs to car controls. */
TMNF_HD void CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
	const TMNFRaceInputs *inputs, CSceneVehicleCar *vehicle)
{
	uint32_t latest;
	uint32_t event_time;
	float gas;
	float brake;
	float steer;

	latest = inputs->accelerate_time;
	if (latest < inputs->brake_time)
		latest = inputs->brake_time;
	event_time = inputs->gas_analog_time;
	if (event_time == latest
		&& inputs->accelerate == 0
		&& inputs->brake == 0
		&& 0.01 < fabs(F(inputs->gas_analog))) {
		event_time = latest + 1;
	}

	if (event_time <= latest) {
		gas = inputs->accelerate != 0 ? 1.0f : 0.0f;
		brake = inputs->brake != 0 ? 1.0f : 0.0f;
	}
	else if (!isnan(F(inputs->gas_analog))
		&& 0.3 <= F(inputs->gas_analog)) {
		gas = 1.0f;
		brake = 0.0f;
	}
	else if (!isnan(F(inputs->gas_analog))
		&& F(inputs->gas_analog) <= -0.3) {
		gas = 0.0f;
		brake = 1.0f;
	}
	else {
		gas = 0.0f;
		brake = 0.0f;
	}
	CSceneVehicle_VehicleInputGasSet(vehicle, gas);
	CSceneVehicle_VehicleInputBrakeSet(vehicle, brake);

	latest = inputs->steer_left_time;
	if (latest < inputs->steer_right_time)
		latest = inputs->steer_right_time;
	event_time = inputs->steer_analog_time;
	if (event_time == latest
		&& inputs->steer_left == 0
		&& inputs->steer_right == 0
		&& 0.01 < fabs(F(inputs->steer_analog))) {
		event_time = latest + 1;
	}

	if (event_time <= latest) {
		if (inputs->steer_left != 0)
			steer = -1.0f;
		else if (inputs->steer_right != 0)
			steer = 1.0f;
		else
			steer = 0.0f;
	}
	else {
		steer = (float)-F(inputs->steer_analog);
	}
	CSceneVehicle_VehicleInputSteerSet(vehicle, steer);
}

/* 0x007C1060  Integrates one wheel's angular and contact blend state. */
TMNF_HD void CSceneVehicleCarWheelRealTimeState_Integrate(
	CSceneVehicleCarWheelRealTimeState *self, float dt)
{
	float phase_input = x87_add(
		x87_mul(self->field6c, dt), self->rotation_phase);
	float axis_length_squared;

	self->rotation_phase = gm_mod(phase_input, 0.0f, WHEEL_ANGLE_PERIOD);
	axis_length_squared = vec_length_squared(&self->field90);
	if (F(NORMALIZE_EPSILON) < F(axis_length_squared)) {
		GmVec3 dov;

		normalize_if_nonzero(&self->field90);
		dov.x = x87_sub(
			x87_mul(self->field90.z, 0.0f),
			x87_mul(self->field90.y, 0.0f));
		dov.y = x87_sub(
			x87_mul(self->field90.x, 0.0f), self->field90.z);
		dov.z = x87_sub(
			self->field90.y, x87_mul(self->field90.x, 0.0f));
		mat3_set_up_v_and_dov(&self->basis1, &self->field90, &dov);
	}
	if (F(self->blend_target) <= F(self->blend_value)) {
		self->blend_value = x87_sub(self->blend_value, dt);
		if (F(self->blend_value) < F(self->blend_target))
			self->blend_value = self->blend_target;
	}
	else {
		self->blend_value = x87_add(self->blend_value, dt);
		if (F(self->blend_target) < F(self->blend_value))
			self->blend_value = self->blend_target;
	}
}

/* 0x007C0EC0  Updates angular wheel speed from contact and driver inputs. */
TMNF_HD void CSceneVehicleCar_WheelUpdateSpeedFromVehicleSpeed(
	CSceneVehicleCar *self, CSceneVehicleCarWheel *wheel,
	float vehicle_speed, float dt)
{
	float acceleration;
	float target;
	float speed;

	if (wheel->real_time.has_ground_contact != 0) {
		if (self->force_wheel_speed != 0 && self->block_wheel_speed == 0)
			wheel->real_time.field6c = self->forced_wheel_speed;
		else
			wheel->real_time.field6c =
				x87_div(vehicle_speed, wheel->radius);
		return;
	}

	acceleration = 0.0f;
	target = 0.0f;
	if (F(self->input_brake) <= F(WHEEL_SPEED_EPSILON)) {
		if (F(self->input_gas) <= F(WHEEL_SPEED_EPSILON)
			|| self->block_wheel_speed != 0 || self->flag_60c != 0) {
			wheel->real_time.field6c = x87_r24(
				F(wheel->real_time.field6c) * WHEEL_SPEED_DAMPING);
		}
		else {
			target = x87_r24(F(self->input_gas) * 200.0);
			acceleration = 100.0f;
		}
	}
	else {
		target = x87_sub(1.0f, self->input_brake);
		if (!isnan(F(target))) {
			if (F(target) <= 0.0)
				target = 0.0f;
			else if (1.0 <= F(target))
				target = 1.0f;
		}
		acceleration = -100.0f;
	}

	if (F(WHEEL_SPEED_EPSILON) <= fabs(F(acceleration))) {
		speed = x87_add(
			x87_mul(acceleration, dt), wheel->real_time.field6c);
		wheel->real_time.field6c = speed;
		if ((0.0 < F(acceleration) && F(target) < F(speed))
			|| (F(acceleration) < 0.0 && F(speed) < F(target))) {
			wheel->real_time.field6c = target;
		}
	}
}

/* 0x008F46C0  Integrates a scalar damped spring by one simulation step. */
TMNF_HD void GmSpringFloat_Integrate(GmSpringFloat *self, float dt)
{
	float displacement;
	float spring_force;
	float damping_force;
	float acceleration;
	float velocity;

	displacement = x87_sub(self->target, self->value);
	spring_force = x87_mul(displacement, self->stiffness);
	damping_force = x87_mul(self->damping, self->velocity);
	acceleration = x87_sub(spring_force, damping_force);
	velocity = x87_add(x87_mul(acceleration, dt), self->velocity);
	self->velocity = velocity;
	self->value = x87_add(x87_mul(dt, velocity), self->value);
}

/* 0x007BD090  Computes one contact impulse from effective point mass. */
TMNF_HD void SDynaMath_ComputeImpulse(
	float mass, const GmMat3 *inverse_inertia, float restitution,
	const GmVec3 *relative_speed, const GmVec3 *normal,
	const GmVec3 *lever_arm, GmVec3 *impulse)
{
	GmVec3 angular;
	GmVec3 projected;
	float relative_normal_speed;
	float rotational_mass;
	float denominator;
	float numerator;
	float scale;

	angular.x = x87_sub(
		x87_mul(normal->z, lever_arm->y),
		x87_mul(normal->y, lever_arm->z));
	angular.y = x87_sub(
		x87_mul(normal->x, lever_arm->z),
		x87_mul(lever_arm->x, normal->z));
	angular.z = x87_sub(
		x87_mul(normal->y, lever_arm->x),
		x87_mul(normal->x, lever_arm->y));
	GmVec3_Mult_Mat3(&angular, inverse_inertia);

	projected.x = x87_sub(
		x87_mul(lever_arm->z, angular.y),
		x87_mul(angular.z, lever_arm->y));
	projected.y = x87_sub(
		x87_mul(lever_arm->x, angular.z),
		x87_mul(lever_arm->z, angular.x));
	projected.z = x87_sub(
		x87_mul(angular.x, lever_arm->y),
		x87_mul(lever_arm->x, angular.y));
	relative_normal_speed = x87_add(
		x87_add(
			x87_mul(relative_speed->y, normal->y),
			x87_mul(relative_speed->x, normal->x)),
		x87_mul(relative_speed->z, normal->z));
	rotational_mass = x87_add(
		x87_add(
			x87_mul(normal->x, projected.x),
			x87_mul(normal->y, projected.y)),
		x87_mul(projected.z, normal->z));
	denominator = x87_add(x87_rcp(mass), rotational_mass);
	numerator = x87_mul(
		x87_sub(restitution, 1.0f), relative_normal_speed);
	scale = x87_div(numerator, denominator);
	impulse->x = x87_mul(normal->x, scale);
	impulse->y = x87_mul(normal->y, scale);
	impulse->z = x87_mul(scale, normal->z);
}

/* 0x007BE310  Adds a local central force to the vehicle rigid body. */
TMNF_HD void CSceneVehicleCar_AddVehicleCentralForce(
	CSceneVehicleCar *self, const GmVec3 *force)
{
	GmVec3 world_force;

	GmVec3_SetMult_Mat3(&world_force, force, &self->dyna_state->rot);
	add_vec3(&self->dyna_state->force, &world_force);
	add_vec3(&self->total_force_added, force);
}

/* 0x007BE360  Adds a local torque to the vehicle rigid body. */
TMNF_HD void CSceneVehicleCar_AddVehicleTorque(
	CSceneVehicleCar *self, const GmVec3 *torque)
{
	GmVec3 world_torque;

	GmVec3_SetMult_Mat3(&world_torque, torque, &self->dyna_state->rot);
	add_vec3(&self->dyna_state->torque, &world_torque);
}

/* 0x007BE2C0  Adds a local force at a local point on the vehicle. */
TMNF_HD void CSceneVehicleCar_AddVehicleForce(
	CSceneVehicleCar *self, const GmVec3 *force, const GmVec3 *point)
{
	GmVec3 world_force;
	GmVec3 world_point;
	GmVec3 center_of_mass;
	GmVec3 lever;
	GmVec3 torque;

	GmVec3_SetMult_Mat3(&world_force, force, &self->dyna_state->rot);
	GmVec3_SetMult_Mat3(&world_point, point, &self->dyna_state->rot);
	world_point.x = x87_add(world_point.x, self->dyna_state->pos.x);
	world_point.y = x87_add(world_point.y, self->dyna_state->pos.y);
	world_point.z = x87_add(world_point.z, self->dyna_state->pos.z);
	GmVec3_SetMult_Mat3(
		&center_of_mass, &self->dyna_params->comOffset,
		&self->dyna_state->rot);
	center_of_mass.x = x87_add(
		center_of_mass.x, self->dyna_state->pos.x);
	center_of_mass.y = x87_add(
		center_of_mass.y, self->dyna_state->pos.y);
	center_of_mass.z = x87_add(
		center_of_mass.z, self->dyna_state->pos.z);
	lever.x = x87_sub(world_point.x, center_of_mass.x);
	lever.y = x87_sub(world_point.y, center_of_mass.y);
	lever.z = x87_sub(world_point.z, center_of_mass.z);
	torque.x = x87_sub(
		x87_mul(lever.y, world_force.z),
		x87_mul(lever.z, world_force.y));
	torque.y = x87_sub(
		x87_mul(lever.z, world_force.x),
		x87_mul(lever.x, world_force.z));
	torque.z = x87_sub(
		x87_mul(lever.x, world_force.y),
		x87_mul(lever.y, world_force.x));
	add_vec3(&self->dyna_state->force, &world_force);
	add_vec3(&self->dyna_state->torque, &torque);
	add_vec3(&self->total_force_added, force);
}

/* 0x007BE00B  The Steer01..Steer05 engine (tuning +0x354 != 5): a speed
 * ratio drives the rpm towards max_rpm with a fixed gain and shifts through
 * the upshift/downshift tables; clutch, target_rpm and engine_mode are not
 * touched. */
TMNF_HD static void engine_integrate_oldmodels(
	CSceneVehicleCar *self, float throttle, float dt, int shift_ready)
{
	CSceneVehicleCarEngine *engine = &self->engine;
	const CSceneVehicleCarTuning *tuning = self->tuning;
	const GmVec3 *speed = &self->current_local_speed;
	/* 0x00B43310 0.2, 0x00B5B8E0 0.3, 0x00B362C0 0.1, 0x00B9EFB8 1.9:
	 * floats widened to double; 0x00B80D18 0.04f, 0x00B3D274 12.0f,
	 * 0x00B36AE8 3.5f. */
	float limit = x87_mul(tuning->forward_speed_limit_scale, 0.2f);
	float magnitude = (float)fabs(F(throttle));
	int can_shift = !shift_ready && !(0.0 < F(engine->shift_timer));
	uint32_t index = engine->gear > 1 ? (uint32_t)engine->gear - 1 : 0;
	float upshift = tuning->gear_upshift[index];
	float downshift = tuning->gear_downshift[index];
	float ratio = tuning->gear_ratios[index];
	float sum = x87_add(
		x87_add(
			x87_mul(x87_mul(speed->x, speed->x), 0.3f),
			x87_mul(speed->z, speed->z)),
		x87_mul(x87_mul(speed->y, speed->y), 0.1f));
	float ratio_speed = x87_mul(x87_div(x87_sqrt(sum), limit), ratio);
	float drive;
	float target;

	if (can_shift) {
		drive = ratio_speed;
		if (engine->reverse != 0) {
			if (engine->gear != 0) {
				engine->gear = 0;
				engine->shift_timer = 0.04f;
			}
		}
		else if (engine->gear == 0) {
			engine->gear = 1;
			engine->shift_timer = 0.04f;
		}
		else if (F(upshift) < F(ratio_speed) && engine->gear < 5) {
			engine->gear += 1;
			engine->shift_timer = 0.04f;
		}
		else if (F(ratio_speed) < F(downshift) && engine->gear > 1) {
			engine->gear -= 1;
			engine->shift_timer = 0.04f;
		}
	}
	else {
		drive = magnitude;
		if (0.0 <= F(engine->shift_timer)
			&& !(F(x87_add(dt, dt)) < F(engine->shift_timer))) {
			engine->rpm = x87_sub(
				engine->rpm,
				x87_mul(x87_mul(engine->max_rpm, dt), 1.9f));
		}
	}
	target = x87_sub(x87_mul(engine->max_rpm, drive), engine->rpm);
	engine->rpm = x87_add(
		x87_mul(x87_mul(target, dt), can_shift ? 12.0f : 3.5f),
		engine->rpm);
}

/* 0x007BD700  Integrates the engine and gearbox state; the model-5 body is
 * 0x007BD7C5.., every other tuning takes 0x007BE00B. */
TMNF_HD void CSceneVehicleCar_EngineIntegrate(
	CSceneVehicleCar *self, float throttle, float dt)
{
	CSceneVehicleCarEngine *engine = &self->engine;
	const CSceneVehicleCarTuning *tuning = self->tuning;
	int throttle_on = 0.1 < F(throttle);
	int all_airborne = 1;
	int shift_ready;
	float value;
	float target;
	uint32_t gear_index;

	for (uint32_t i = 0; i < self->wheel_count; ++i) {
		if (self->wheels[i].real_time.has_ground_contact != 0) {
			all_airborne = 0;
			break;
		}
	}
	if (0.0 < F(engine->shift_timer))
		engine->shift_timer = x87_sub(engine->shift_timer, dt);
	shift_ready = all_airborne || 0.0 < F(engine->shift_timer);

	if (tuning->engine_model != 5) {
		engine_integrate_oldmodels(self, throttle, dt, shift_ready);
		goto clamp_rpm;
	}

	if (shift_ready) {
		if (throttle_on)
			engine->rpm = x87_add(
				x87_mul(tuning->engine_rpm_accel, dt), engine->rpm);
		else
			engine->rpm = x87_sub(
				engine->rpm,
				x87_mul(tuning->engine_rpm_decel, dt));
		goto clamp_rpm;
	}

	if (self->drive_mode == 1 || self->drive_mode == 2) {
		if (engine->gear != 0)
			self->engine_mode = 4;
	}
	else if (self->engine_mode == 4) {
		self->engine_mode = 0;
	}

	if (self->engine_mode == 2) {
		self->engine_limit_flag =
			F(tuning->speed_32c) < F(self->current_local_speed.z)
			|| F(self->current_local_speed.z) < F(tuning->speed_330);
		engine->clutch = 1.0f;
		engine->target_rpm = x87_add(
			x87_mul(tuning->gear_aux[1], 0.0f),
			x87_mul(
				(float)fabs(F(self->current_local_speed.z)),
				tuning->gear_ratios[1]));
		/* 0x007BDD33..0x007BDD54: the active-throttle branch stores
		 * engine_rpm_low_accel * dt + rpm and falls through to the shift
		 * checks. Only the deceleration branches (0x007BDCB2 and
		 * 0x007BDD0C) reach the target comparison at 0x007BDCDA. */
		if (self->engine_limit_flag == 0 && throttle_on) {
			engine->rpm = x87_add(
				x87_mul(tuning->engine_rpm_low_accel, dt),
				engine->rpm);
			goto shift_checks;
		}
		value = x87_sub(
			engine->rpm,
			x87_mul(tuning->engine_rpm_high_decel, dt));
		engine->rpm = value;
		if (F(value) <= F(engine->target_rpm)) {
			engine->rpm = engine->target_rpm;
			self->engine_mode = 0;
			self->engine_limit_flag = 0;
		}
		goto shift_checks;
	}

	if (self->engine_mode == 3) {
		self->engine_limit_flag =
			F(self->current_local_speed.z) < F(tuning->speed_338)
			|| F(tuning->speed_334) < F(self->current_local_speed.z);
		engine->clutch = 1.0f;
		engine->target_rpm = x87_add(
			x87_mul(tuning->gear_aux[0], 0.0f),
			x87_mul(
				(float)fabs(F(self->current_local_speed.z)),
				tuning->gear_ratios[0]));
		/* 0x007BDBC3 jumps to the same unclamped active-throttle store
		 * at 0x007BDD33; only the deceleration branches reach the target
		 * comparison at 0x007BDB8C. */
		if (self->engine_limit_flag == 0 && throttle_on) {
			engine->rpm = x87_add(
				x87_mul(tuning->engine_rpm_low_accel, dt),
				engine->rpm);
			goto shift_checks;
		}
		value = x87_sub(
			engine->rpm,
			x87_mul(tuning->engine_rpm_high_decel, dt));
		engine->rpm = value;
		if (!(F(engine->target_rpm) < F(value))) {
			engine->rpm = engine->target_rpm;
			self->engine_mode = 0;
			self->engine_limit_flag = 0;
		}
		goto shift_checks;
	}

	if (self->engine_mode == 4) {
		engine->target_rpm = engine->max_rpm;
		engine->clutch = 1.15f;
		if (F(engine->max_rpm) <= F(engine->rpm)) {
			if (F(engine->rpm) <= F(engine->max_rpm))
				goto shift_checks;
			value = tuning->engine_rpm_decel;
		}
		else {
			value = tuning->engine_rpm_reverse_accel;
		}
		engine->rpm = x87_add(x87_mul(value, dt), engine->rpm);
		goto shift_checks;
	}

	if (self->turbo_active != 0 && throttle_on) {
		value = 1.15f;
		if (F(engine->clutch) < 1.15) {
			value = x87_add(
				x87_mul(
					x87_mul(x87_sub(1.15f, engine->clutch), 0.3f),
					dt),
				engine->clutch);
		}
	}
	else {
		value = 1.0f;
	}
	engine->clutch = value;
	gear_index = (uint32_t)engine->gear;
	target = x87_add(
		x87_mul(tuning->gear_aux[gear_index], 0.0f),
		x87_mul(
			(float)fabs(F(x87_mul(
				self->current_local_speed.z, engine->clutch))),
			tuning->gear_ratios[gear_index]));
	engine->target_rpm = target;
	if ((engine->reverse == 0 && gear_index == 0)
		|| (engine->reverse != 0 && gear_index != 0)) {
		engine->target_rpm = 0.0f;
		target = 0.0f;
	}
	if (F(target) <= F(engine->rpm)) {
		if (throttle_on) {
			if (self->turbo_active == 0
				|| self->drive_mode == 1 || self->drive_mode == 2) {
				value = tuning->engine_rpm_turbo_decel;
			}
			else {
				value = tuning->engine_rpm_decel;
			}
		}
		else {
			value = tuning->engine_rpm_high_decel;
		}
		engine->rpm = x87_sub(engine->rpm, x87_mul(value, dt));
		if (F(engine->rpm) < F(engine->target_rpm))
			self->engine_mode = 0;
	}
	else {
		engine->rpm = x87_add(
			x87_mul(tuning->engine_rpm_follow_accel, dt),
			engine->rpm);
		if (F(engine->target_rpm) < F(engine->rpm))
			self->engine_mode = 0;
	}

shift_checks:
	if (!throttle_on || engine->reverse != 0 || self->engine_mode != 0) {
		if (F(tuning->speed_338) < F(self->current_local_speed.z)
			&& F(self->current_local_speed.z)
				< F(tuning->speed_334)
			&& throttle_on && engine->reverse != 0
			&& self->engine_mode == 0) {
			self->engine_mode = 3;
			self->engine_limit_flag = 0;
			if (engine->gear != 0) {
				engine->gear = 0;
				engine->shift_timer = 0.025f;
			}
		}
	}
	else if (F(tuning->speed_330) < F(self->current_local_speed.z)
		&& F(self->current_local_speed.z) < F(tuning->speed_32c)) {
		self->engine_mode = 2;
		self->engine_limit_flag = 0;
		if (engine->gear == 0) {
			engine->shift_timer = 0.025f;
			engine->gear = 1;
		}
	}

	if (self->engine_mode == 0 || self->engine_mode == 1) {
		if (engine->reverse == 0) {
			if (engine->gear == 0) {
				self->engine_mode = 1;
				self->gear_downshift_flag = 0;
				if (F(engine->rpm) < 1000.0) {
					engine->shift_timer = 0.025f;
					engine->gear = 1;
				}
			}
			gear_index = (uint32_t)engine->gear;
			if (0 < engine->gear) {
				if (F(engine->target_rpm)
						<= F(x87_mul(
							tuning->gear_upshift[gear_index],
							engine->max_rpm))
					|| 4 < engine->gear) {
					if (F(engine->target_rpm)
							< F(x87_mul(
								tuning->gear_downshift[gear_index],
								engine->max_rpm))
						&& 1 < engine->gear) {
						engine->shift_timer = 0.025f;
						--engine->gear;
						self->engine_mode = 1;
						self->gear_downshift_flag = 1;
					}
				}
				else {
					engine->shift_timer = 0.025f;
					++engine->gear;
					self->engine_mode = 1;
					self->gear_downshift_flag = 0;
				}
			}
		}
		else if (engine->gear != 0) {
			self->engine_mode = 1;
			self->gear_downshift_flag = 1;
			if (F(engine->rpm) < 1000.0) {
				engine->gear = 0;
				engine->shift_timer =
					all_airborne ? 0.002f : 0.025f;
			}
		}
	}

clamp_rpm:
	value = engine->rpm;
	if (0.0 < F(value)) {
		if (F(engine->max_rpm) < F(value))
			engine->rpm = engine->max_rpm;
	}
	else {
		engine->rpm = 0.0f;
	}
}

/* 0x007C1810  Applies one wheel's suspension force to the vehicle. */
TMNF_HD void CSceneVehicleCar_WheelAddForceToVehicle(
	CSceneVehicleCar *self, const CSceneVehicleCarWheel *wheel)
{
	GmVec3 force = { 0.0f, 0.0f, 0.0f };
	float delta;

	if (wheel->real_time.has_ground_contact == 0)
		return;
	delta = x87_sub(
		self->tuning->suspension_rest_length,
		wheel->real_time.damper_absorb);
	switch (self->tuning->suspension_model) {
	case 0:
		force.y = x87_mul(
			x87_mul(
				self->tuning->suspension_stiffness,
				self->tuning->suspension_scale),
			delta);
		break;
	case 1:
	case 2:
		force.y = x87_sub(
			x87_mul(delta, self->tuning->suspension_stiffness),
			x87_mul(
				self->tuning->suspension_damping,
				wheel->real_time.field04));
		break;
	default:
		return;
	}
	CSceneVehicleCar_AddVehicleForce(
		self, &force, &wheel->offset_from_vehicle);
}
