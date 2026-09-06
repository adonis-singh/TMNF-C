#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "track.h"
#include "trace_format.h"
#include "vehicle_model6.h"
#include "../oracle/tracer/model6_trace.h"

enum {
	CAR_SIZE = 0x878,
	TUNING_SIZE = 0x3ac,
	WHEEL_SIZE = 0x2fc,
	DYNA_SIZE = 0x344,
	PARAMS_SIZE = 0x5c,
	STATE_SIZE = 0xb4,
};

typedef struct {
	CSceneVehicleCar vehicle;
	CSceneVehicleCarTuning vehicle_tuning;
	CSceneVehicleCarWheel wheels[4];
	CSceneVehicleCarWheelHistory history[4];
	CHmsStateDyna dyna_state;
	CHmsDynaParams dyna_params;
	CSceneVehicleCarAuxContext aux;
	TMNFVehicleContactContext contact;
	TMNFVehicleContactTimer timer;
	CSceneVehicleCarTuningCurveSet curve_set;
	CFuncKeysReal curves[TMNF_MODEL6_CURVE_COUNT];
	TMNFVehicleGroundMaterial *materials;
	const TMNFVehicleGroundMaterial **material_ptrs;
	uint32_t *ground_ids;
	CSceneVehicleCarModel6Tuning tuning;
	CSceneVehicleCarModel6State state;
	CSceneVehicleCarModel6Context context;
	GmIso4 model_iso;
	GmVec3 body_reference;
} Fixture;

/* The Model6 trace carries no track: an empty water map makes 0x007C2910
 * return 0 for every position, as before the water port. */
static const uint8_t NO_WATER_CELLS[32 * 32];
static const TmnfTrackWater NO_WATER = {
	32.0f, 32.0f, 0.0f, 0.0f, 32, 32, 0, NO_WATER_CELLS, 0, 0.0f, -992.0f,
};
static const CSceneVehicleCarTuningAux NO_WATER_TUNING = {
	.water_map = &NO_WATER,
};

static void die(const char *message)
{
	fprintf(stderr, "%s\n", message);
	exit(2);
}

static uint16_t read_u16(FILE *file)
{
	uint8_t bytes[2];
	if (fread(bytes, 1, 2, file) != 2)
		die("unexpected Model6 trace EOF");
	return (uint16_t)(bytes[0] | (uint16_t)bytes[1] << 8);
}

static uint32_t read_u32(FILE *file)
{
	uint8_t bytes[4];
	if (fread(bytes, 1, 4, file) != 4)
		die("unexpected Model6 trace EOF");
	return (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8
		| (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
}

static uint8_t *read_buffer(FILE *file, uint32_t expected_tag, uint32_t *size)
{
	uint32_t tag = read_u32(file);
	uint32_t address = read_u32(file);
	uint8_t *bytes;
	(void)address;
	*size = read_u32(file);
	if (tag != expected_tag || *size < sizeof(struct TmnfModel6TraceHeader))
		die("invalid Model6 trace buffer");
	bytes = malloc(*size);
	if (bytes == NULL)
		die("out of memory");
	if (fread(bytes, 1, *size, file) != *size)
		die("unexpected Model6 trace blob EOF");
	return bytes;
}

static const uint8_t *section(
	const uint8_t *blob, uint32_t size, uint32_t offset, uint32_t length)
{
	if (offset > size || length > size - offset)
		die("Model6 graph section is out of bounds");
	return blob + offset;
}

static const struct TmnfModel6TraceHeader *validate_blob(
	const uint8_t *blob, uint32_t size, uint32_t phase)
{
	const struct TmnfModel6TraceHeader *header =
		(const struct TmnfModel6TraceHeader *)blob;
	if (memcmp(header->magic, TMNF_MODEL6_TRACE_MAGIC, 8) != 0
	    || header->version != TMNF_MODEL6_TRACE_VERSION
	    || header->total_size != size || header->phase != phase
	    || header->wheel_count != 4)
		die("invalid Model6 graph header");
	(void)section(blob, size, header->car_offset, CAR_SIZE);
	(void)section(blob, size, header->tuning_offset, TUNING_SIZE);
	(void)section(blob, size, header->wheels_offset, 4 * WHEEL_SIZE);
	(void)section(blob, size, header->dyna_offset, DYNA_SIZE);
	(void)section(blob, size, header->params_offset, PARAMS_SIZE);
	(void)section(blob, size, header->state_offset, STATE_SIZE);
	(void)section(blob, size, header->curves_offset,
		TMNF_MODEL6_CURVE_COUNT * sizeof(struct TmnfModel6Curve));
	(void)section(blob, size, header->ground_ids_offset,
		header->ground_id_count * 4);
	(void)section(blob, size, header->ground_materials_offset,
		header->ground_material_count * sizeof(struct TmnfModel6Material));
	(void)section(blob, size, header->model_iso_offset, sizeof(GmIso4));
	(void)section(blob, size, header->body_reference_offset, sizeof(GmVec3));
	return header;
}

static float f32(const uint8_t *bytes, uint32_t offset)
{
	float value;
	memcpy(&value, bytes + offset, 4);
	return value;
}

static int32_t i32(const uint8_t *bytes, uint32_t offset)
{
	int32_t value;
	memcpy(&value, bytes + offset, 4);
	return value;
}

static uint32_t u32(const uint8_t *bytes, uint32_t offset)
{
	uint32_t value;
	memcpy(&value, bytes + offset, 4);
	return value;
}

static void decode_wheel(CSceneVehicleCarWheel *wheel,
	CSceneVehicleCarWheelHistory *history, const uint8_t *raw)
{
	memset(wheel, 0, sizeof(*wheel));
	wheel->history = history;
	wheel->active = i32(raw, 0x000);
	wheel->steerable = i32(raw, 0x004);
	wheel->radius = f32(raw, 0x008);
	memcpy(wheel->field70, raw + 0x070, sizeof(wheel->field70));
	wheel->fielda0 = i32(raw, 0x0a0);
	wheel->fielda4 = i32(raw, 0x0a4);
	memcpy(&wheel->offset_from_vehicle, raw + 0x0a8, sizeof(GmVec3));
	memcpy(&wheel->real_time, raw + 0x0b4, sizeof(wheel->real_time));
	wheel->field15c = i32(raw, 0x15c);
	memcpy(&wheel->contact_relative_local_distance,
		raw + 0x160, sizeof(GmVec3));
	memcpy(&wheel->history->previous_sync, raw + 0x16c, 0x64);
	memcpy(&wheel->history->sync, raw + 0x1d0, 0x64);
	memcpy(&wheel->history->field234, raw + 0x234, 0x64);
	memcpy(&wheel->history->async_state, raw + 0x298, 0x64);
}

static void decode_vehicle_tuning(
	CSceneVehicleCarTuning *tuning, const uint8_t *raw)
{
	memset(tuning, 0, sizeof(*tuning));
	tuning->damper_max = f32(raw, 0x11c);
	tuning->damper_min = f32(raw, 0x120);
	tuning->suspension_model = i32(raw, 0x350);
	tuning->suspension_stiffness = f32(raw, 0x114);
	tuning->suspension_damping = f32(raw, 0x118);
	tuning->suspension_rest_length = f32(raw, 0x124);
	tuning->suspension_scale = f32(raw, 0x128);
}

#define TUNE(field, offset) fixture->tuning.field = f32(raw, offset)

static void decode_model6_tuning(Fixture *fixture, const uint8_t *raw)
{
	TUNE(forward_speed_limit_scale, 0x02c);
	TUNE(reverse_speed_limit_scale, 0x030);
	TUNE(brake_base, 0x040);
	TUNE(brake_speed_scale, 0x044);
	TUNE(forward_brake_limit_sliding, 0x048);
	TUNE(forward_brake_limit, 0x04c);
	TUNE(speed_limit_force, 0x060);
	TUNE(vertical_force_scale, 0x064);
	TUNE(wheel_steer_sine_limit, 0x074);
	TUNE(steer_slowdown_scale, 0x07c);
	TUNE(wheel_torque_scale, 0x098);
	TUNE(sliding_steer_torque_scale, 0x09c);
	TUNE(lateral_force_scale, 0x0a4);
	TUNE(sliding_lateral_limit_scale, 0x0b0);
	TUNE(lateral_overflow_blend, 0x0b4);
	TUNE(wheel_overflow_blend, 0x0e4);
	TUNE(vertical_force_divisor, 0x160);
	TUNE(traction_loss_scale, 0x200);
	TUNE(burnout_trigger_scale, 0x228);
	TUNE(burnout_trigger_limit, 0x22c);
	TUNE(rollover_axis_min_length, 0x234);
	TUNE(rollover_torque_x_scale, 0x238);
	TUNE(rollover_torque_z_scale, 0x23c);
	TUNE(sliding_brake_scale, 0x240);
	TUNE(braking_lateral_limit_scale, 0x244);
	TUNE(reverse_brake_limit_sliding, 0x248);
	TUNE(reverse_brake_limit, 0x24c);
	TUNE(burnout_speed_max, 0x254);
	TUNE(burnout_speed_min, 0x258);
	TUNE(donut_lateral_force_scale, 0x264);
	TUNE(donut_yaw_angle_scale, 0x268);
	TUNE(donut_steer_linear, 0x26c);
	TUNE(donut_steer_quadratic, 0x270);
	TUNE(donut_countersteer_scale, 0x274);
	TUNE(donut_radius_exponent, 0x278);
	TUNE(donut_radial_speed_exponent, 0x27c);
	TUNE(donut_radius_min, 0x280);
	TUNE(donut_lateral_speed_limit, 0x284);
	TUNE(donut_normal_angle_limit, 0x28c);
	TUNE(donut_angle_positive_limit, 0x290);
	TUNE(donut_angle_negative_limit, 0x294);
	fixture->tuning.burnout_enter_ticks = u32(raw, 0x298);
	TUNE(burnout_enter_accel_scale, 0x29c);
	TUNE(burnout_enter_lateral_scale, 0x2a0);
	fixture->tuning.burnout_exit_ticks = u32(raw, 0x2a8);
	TUNE(burnout_exit_accel_scale, 0x2ac);
	TUNE(burnout_exit_extra_accel, 0x2b8);
	TUNE(material6_longitudinal_scale, 0x33c);
	TUNE(material6_gas_denominator, 0x340);
	TUNE(material6_vertical_shape, 0x344);
	TUNE(material6_vertical_scale, 0x348);
}

#undef TUNE

static void decode_model6_state(
	CSceneVehicleCarModel6State *state, const uint8_t *raw)
{
	memcpy(&state->pivot_position, raw + 0x1dc, 12);
	memcpy(&state->pivot_axis, raw + 0x1e8, 12);
	state->reverse_mode = i32(raw, 0x5c4);
	state->reverse_speed_threshold = f32(raw, 0x5cc);
	state->contact_block_count = i32(raw, 0x5d8);
	state->side_contact = i32(raw, 0x5dc);
	state->last_sliding_tick = u32(raw, 0x62c);
	state->sliding_start_tick = u32(raw, 0x630);
	state->sliding_elapsed_ticks = u32(raw, 0x634);
	memcpy(&state->model_iso, raw + 0x6a4, 0x30);
	memcpy(&state->rollover_axis, raw + 0x6d4, 12);
	memcpy(&state->orbit_center, raw + 0x6e0, 12);
	state->orbit_initial_radius = f32(raw, 0x6ec);
	state->orbit_radius = f32(raw, 0x6f0);
	state->burnout_start_tick = u32(raw, 0x6f4);
	state->burnout_transition_tick = u32(raw, 0x6f8);
	memcpy(&state->orbit_axis, raw + 0x6fc, 12);
	state->orbit_sign = f32(raw, 0x708);
	state->axle_width = f32(raw, 0x840);
}

static void decode_curves(
	Fixture *fixture, const uint8_t *blob, uint32_t size,
	const struct TmnfModel6TraceHeader *header)
{
	const struct TmnfModel6Curve *raw = (const struct TmnfModel6Curve *)
		section(blob, size, header->curves_offset,
			TMNF_MODEL6_CURVE_COUNT * sizeof(*raw));
	for (uint32_t i = 0; i < TMNF_MODEL6_CURVE_COUNT; ++i) {
		fixture->curves[i].keys.count = raw[i].count;
		fixture->curves[i].keys.positions = (const float *)section(
			blob, size, raw[i].positions_offset, raw[i].count * 4);
		fixture->curves[i].values = (const float *)section(
			blob, size, raw[i].values_offset, raw[i].count * 4);
		fixture->curves[i].interpolation = raw[i].interpolation;
		CFuncKeys_Compile(&fixture->curves[i].keys);
	}
	fixture->curve_set.accel_from_speed =
		&fixture->curves[TMNF_M6_CURVE_ACCEL];
	fixture->curve_set.steer_slowdown_from_speed =
		&fixture->curves[TMNF_M6_CURVE_STEER_SLOWDOWN];
	fixture->curve_set.steer_drive_torque_from_speed =
		&fixture->curves[TMNF_M6_CURVE_STEER_DRIVE_TORQUE];
	fixture->curve_set.max_side_friction_from_speed =
		&fixture->curves[TMNF_M6_CURVE_MAX_SIDE_FRICTION];
	fixture->curve_set.m5_slipping_accel_from_speed =
		&fixture->curves[TMNF_M6_CURVE_SLIPPING_ACCEL];
	fixture->curve_set.m6_damper_modulation =
		&fixture->curves[TMNF_M6_CURVE_DAMPER_MODULATION];
	fixture->curve_set.m6_rear_gear_accel_from_speed =
		&fixture->curves[TMNF_M6_CURVE_REAR_GEAR_ACCEL];
	fixture->curve_set.m6_rollover_lateral_from_speed_ratio =
		&fixture->curves[TMNF_M6_CURVE_ROLLOVER_RATIO];
	fixture->curve_set.m6_burnout_radius_from_speed =
		&fixture->curves[TMNF_M6_CURVE_BURNOUT_RADIUS];
	fixture->curve_set.m6_lateral_speed_from_burnout_radius =
		&fixture->curves[TMNF_M6_CURVE_BURNOUT_LATERAL_SPEED];
	fixture->curve_set.m6_donut_rollover_from_speed =
		&fixture->curves[TMNF_M6_CURVE_DONUT_ROLLOVER];
	fixture->curve_set.m6_burnout_rollover_from_speed =
		&fixture->curves[TMNF_M6_CURVE_BURNOUT_ROLLOVER];
}

static void init_fixture(
	Fixture *fixture, const uint8_t *blob, uint32_t size,
	const struct TmnfModel6TraceHeader *header)
{
	const uint8_t *car = section(blob, size, header->car_offset, CAR_SIZE);
	const uint8_t *tuning =
		section(blob, size, header->tuning_offset, TUNING_SIZE);
	const uint8_t *wheels =
		section(blob, size, header->wheels_offset, 4 * WHEEL_SIZE);
	const struct TmnfModel6Material *raw_materials;
	memset(fixture, 0, sizeof(*fixture));
	memcpy(&fixture->dyna_state,
		section(blob, size, header->state_offset, STATE_SIZE), STATE_SIZE);
	memcpy(&fixture->dyna_params,
		section(blob, size, header->params_offset, sizeof(CHmsDynaParams)),
		sizeof(CHmsDynaParams));
	decode_vehicle_tuning(&fixture->vehicle_tuning, tuning);
	decode_model6_tuning(fixture, tuning);
	decode_model6_state(&fixture->state, car);
	for (uint32_t i = 0; i < 4; ++i)
		decode_wheel(&fixture->wheels[i], &fixture->history[i], wheels + i * WHEEL_SIZE);
	fixture->vehicle.dyna_state = &fixture->dyna_state;
	fixture->vehicle.dyna_params = &fixture->dyna_params;
	fixture->vehicle.tuning = &fixture->vehicle_tuning;
	fixture->vehicle.input_gas = f32(car, 0x050);
	fixture->vehicle.input_brake = f32(car, 0x054);
	fixture->vehicle.input_steer = f32(car, 0x058);
	fixture->vehicle.wheels = fixture->wheels;
	fixture->vehicle.wheel_count = 4;
	fixture->vehicle.engine_mode = i32(car, 0x2e4);
	fixture->vehicle.turbo_active = i32(car, 0x628);
	fixture->vehicle.drive_mode = i32(car, 0x69c);
	fixture->vehicle.force_wheel_speed = i32(car, 0x6a0);
	fixture->vehicle.flag_60c = i32(car, 0x60c);
	fixture->vehicle.block_wheel_speed = i32(car, 0x73c);
	fixture->vehicle.engine_limit_flag = i32(car, 0x744);
	fixture->vehicle.gear_downshift_flag = i32(car, 0x748);
	memcpy(&fixture->vehicle.current_local_speed, car + 0x70c, 12);
	memcpy(&fixture->vehicle.total_force_added, car + 0x818, 12);
	memcpy(&fixture->vehicle.total_impulse_added, car + 0x824, 12);
	fixture->aux.vehicle = &fixture->vehicle;
	fixture->aux.air_control_locked = i32(car, 0x5e4);
	fixture->aux.steering_value = f32(car, 0x5e8);
	fixture->aux.turbo_factor = f32(car, 0x5f4);
	fixture->aux.turbo_type = (TMNFVehicleTurboType)i32(car, 0x600);
	fixture->aux.tuning = &NO_WATER_TUNING;
	memcpy(&fixture->aux.body_box, car + 0x1dc, sizeof(GmBoxAligned));
	fixture->ground_ids = malloc(header->ground_id_count * 4);
	fixture->materials = calloc(
		header->ground_material_count, sizeof(*fixture->materials));
	fixture->material_ptrs = calloc(
		header->ground_material_count, sizeof(*fixture->material_ptrs));
	if (fixture->ground_ids == NULL || fixture->materials == NULL
	    || fixture->material_ptrs == NULL)
		die("out of memory");
	memcpy(fixture->ground_ids,
		section(blob, size, header->ground_ids_offset,
			header->ground_id_count * 4),
		header->ground_id_count * 4);
	raw_materials = (const struct TmnfModel6Material *)section(
		blob, size, header->ground_materials_offset,
		header->ground_material_count * sizeof(*raw_materials));
	for (uint32_t i = 0; i < header->ground_material_count; ++i) {
		memcpy(fixture->materials[i].values, raw_materials[i].values, 16);
		fixture->material_ptrs[i] = &fixture->materials[i];
	}
	fixture->timer.tick_time = header->tick;
	fixture->contact.vehicle = &fixture->vehicle;
	fixture->contact.ground_material_indices = fixture->ground_ids;
	fixture->contact.ground_material_index_count = header->ground_id_count;
	fixture->contact.ground_materials = fixture->material_ptrs;
	fixture->contact.ground_material_count = header->ground_material_count;
	fixture->contact.timer = &fixture->timer;
	decode_curves(fixture, blob, size, header);
	fixture->curve_set.m5_slipping_accel_scale = f32(tuning, 0x1e4);
	fixture->curve_set.damper_max = f32(tuning, 0x11c);
	fixture->curve_set.damper_min = f32(tuning, 0x120);
	memcpy(&fixture->model_iso,
		section(blob, size, header->model_iso_offset, 0x30), 0x30);
	memcpy(&fixture->body_reference,
		section(blob, size, header->body_reference_offset, 12), 12);
	fixture->context.vehicle = &fixture->vehicle;
	fixture->context.aux = &fixture->aux;
	fixture->context.contact = &fixture->contact;
	fixture->context.curves = &fixture->curve_set;
	fixture->context.tuning = &fixture->tuning;
	fixture->context.state = &fixture->state;
	fixture->context.model_iso_source = &fixture->model_iso;
	fixture->context.body_reference_position = &fixture->body_reference;
}

static void free_fixture(Fixture *fixture)
{
	for (uint32_t i = 0; i < TMNF_MODEL6_CURVE_COUNT; ++i)
		CFuncKeys_Release(&fixture->curves[i].keys);
	free(fixture->ground_ids);
	free(fixture->materials);
	free(fixture->material_ptrs);
}

static void patch_car(uint8_t *raw, const Fixture *fixture)
{
	const CSceneVehicleCar *vehicle = &fixture->vehicle;
	const CSceneVehicleCarModel6State *state = &fixture->state;
	memcpy(raw + 0x050, &vehicle->input_gas, 4);
	memcpy(raw + 0x054, &vehicle->input_brake, 4);
	memcpy(raw + 0x058, &vehicle->input_steer, 4);
	memcpy(raw + 0x2e4, &vehicle->engine_mode, 4);
	memcpy(raw + 0x5e4, &fixture->aux.air_control_locked, 4);
	memcpy(raw + 0x628, &vehicle->turbo_active, 4);
	memcpy(raw + 0x60c, &vehicle->flag_60c, 4);
	memcpy(raw + 0x69c, &vehicle->drive_mode, 4);
	memcpy(raw + 0x6a0, &vehicle->force_wheel_speed, 4);
	memcpy(raw + 0x70c, &vehicle->current_local_speed, 12);
	memcpy(raw + 0x73c, &vehicle->block_wheel_speed, 4);
	memcpy(raw + 0x744, &vehicle->engine_limit_flag, 4);
	memcpy(raw + 0x748, &vehicle->gear_downshift_flag, 4);
	memcpy(raw + 0x818, &vehicle->total_force_added, 12);
	memcpy(raw + 0x824, &vehicle->total_impulse_added, 12);
	memcpy(raw + 0x1dc, &state->pivot_position, 12);
	memcpy(raw + 0x1e8, &state->pivot_axis, 12);
	memcpy(raw + 0x5c4, &state->reverse_mode, 4);
	memcpy(raw + 0x5cc, &state->reverse_speed_threshold, 4);
	memcpy(raw + 0x5d8, &state->contact_block_count, 4);
	memcpy(raw + 0x5dc, &state->side_contact, 4);
	memcpy(raw + 0x62c, &state->last_sliding_tick, 4);
	memcpy(raw + 0x630, &state->sliding_start_tick, 4);
	memcpy(raw + 0x634, &state->sliding_elapsed_ticks, 4);
	memcpy(raw + 0x6a4, &state->model_iso, 0x30);
	memcpy(raw + 0x6d4, &state->rollover_axis, 12);
	memcpy(raw + 0x6e0, &state->orbit_center, 12);
	memcpy(raw + 0x6ec, &state->orbit_initial_radius, 4);
	memcpy(raw + 0x6f0, &state->orbit_radius, 4);
	memcpy(raw + 0x6f4, &state->burnout_start_tick, 4);
	memcpy(raw + 0x6f8, &state->burnout_transition_tick, 4);
	memcpy(raw + 0x6fc, &state->orbit_axis, 12);
	memcpy(raw + 0x708, &state->orbit_sign, 4);
	memcpy(raw + 0x840, &state->axle_width, 4);
}

static int report_difference(
	uint32_t record, const char *region,
	const uint8_t *actual, const uint8_t *expected, uint32_t size)
{
	static uint32_t diagnostic_budget = 1;
	for (uint32_t i = 0; i < size; ++i) {
		if (actual[i] != expected[i]) {
			if (diagnostic_budget != 0) {
				float actual_float;
				float expected_float;
				uint32_t aligned = i & ~3u;
				--diagnostic_budget;
				fprintf(stderr,
					"record %u %s +0x%x: actual=%02x expected=%02x\n",
					record, region, i, actual[i], expected[i]);
				memcpy(&actual_float, actual + aligned, 4);
				memcpy(&expected_float, expected + aligned, 4);
				fprintf(stderr,
					"  float at +0x%x actual=%.9g expected=%.9g\n",
					aligned, actual_float, expected_float);
			}
			return 0;
		}
	}
	return 1;
}

static int replay_record(
	uint32_t record, const uint8_t *input, uint32_t input_size,
	const uint8_t *output, uint32_t output_size)
{
	const struct TmnfModel6TraceHeader *in =
		validate_blob(input, input_size, 0);
	const struct TmnfModel6TraceHeader *out =
		validate_blob(output, output_size, 1);
	Fixture fixture;
	GmVec3 existing_force;
	GmVec3 local_speed;
	GmVec3 local_angular_speed;
	CSceneVehicleMaterialBlendableVals material;
	int sliding = in->sliding;
	float brake = in->brake_force;
	uint8_t actual_car[CAR_SIZE];
	uint8_t actual_wheels[4 * WHEEL_SIZE];
	init_fixture(&fixture, input, input_size, in);
	memcpy(&existing_force, in->existing_force, 12);
	memcpy(&local_speed, in->local_speed, 12);
	memcpy(&local_angular_speed, in->local_angular_speed, 12);
	memcpy(&material, in->material, 16);
	CSceneVehicleCar_ComputeForcesModel6(
		&fixture.context, in->model_value, &existing_force,
		in->lateral_force_factor, in->longitudinal_force_factor,
		&local_speed, &local_angular_speed, in->steering_angle,
		in->grounded, &material, &sliding, &brake);
	memcpy(actual_car,
		section(input, input_size, in->car_offset, CAR_SIZE), CAR_SIZE);
	patch_car(actual_car, &fixture);
	memcpy(actual_wheels,
		section(input, input_size, in->wheels_offset, sizeof(actual_wheels)),
		sizeof(actual_wheels));
	for (uint32_t i = 0; i < 4; ++i)
		memcpy(actual_wheels + i * WHEEL_SIZE + 0xb4,
			&fixture.wheels[i].real_time,
			sizeof(fixture.wheels[i].real_time));
	if (!report_difference(record, "car", actual_car,
			section(output, output_size, out->car_offset, CAR_SIZE),
			CAR_SIZE)
	    || !report_difference(record, "wheels", actual_wheels,
			section(output, output_size, out->wheels_offset,
				sizeof(actual_wheels)), sizeof(actual_wheels))
	    || !report_difference(record, "dyna-state",
			(const uint8_t *)&fixture.dyna_state,
			section(output, output_size, out->state_offset, STATE_SIZE),
			STATE_SIZE)
	    || sliding != out->sliding
	    || memcmp(&brake, &out->brake_force, 4) != 0) {
		if (sliding != out->sliding)
			fprintf(stderr, "record %u sliding: actual=%d expected=%d\n",
				record, sliding, out->sliding);
		if (memcmp(&brake, &out->brake_force, 4) != 0)
			fprintf(stderr, "record %u brake differs\n", record);
		free_fixture(&fixture);
		return 0;
	}
	free_fixture(&fixture);
	return 1;
}

int main(int argc, char **argv)
{
	FILE *file;
	char magic[8];
	uint32_t va;
	uint32_t record_count;
	uint32_t passed = 0;
	if (argc != 2)
		die("usage: replay_model6 TRACE");
	file = fopen(argv[1], "rb");
	if (file == NULL)
		die("cannot open Model6 trace");
	if (fread(magic, 1, 8, file) != 8
	    || memcmp(magic, TMNF_TRACE_MAGIC, 8) != 0)
		die("bad trace magic");
	va = read_u32(file);
	record_count = read_u32(file);
	if (va != 0x007c3e80 || record_count == 0)
		die("invalid Model6 trace header");
	for (uint32_t i = 0; i < record_count; ++i) {
		uint32_t input_size;
		uint32_t output_size;
		uint8_t *input;
		uint8_t *output;
		uint32_t seq = read_u32(file);
		if (read_u16(file) != 1 || read_u16(file) != 1 || seq != i)
			die("invalid Model6 record framing");
		input = read_buffer(file, TAG_THIS, &input_size);
		output = read_buffer(file, TAG_OUT_THIS, &output_size);
		if (replay_record(i, input, input_size, output, output_size))
			++passed;
		free(input);
		free(output);
	}
	if (fgetc(file) != EOF)
		die("trailing Model6 trace bytes");
	fclose(file);
	printf("Model6: %u/%u records byte-exact\n", passed, record_count);
	return passed == record_count ? 0 : 1;
}
