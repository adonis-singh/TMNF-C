#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trace_format.h"
#include "vehicle.h"
#include "vehicle_aux.h"

typedef struct {
	uint32_t tag;
	uint32_t addr;
	uint32_t len;
	uint8_t *bytes;
} Buffer;

typedef struct {
	uint32_t seq;
	int n_in;
	int n_out;
	Buffer *in;
	Buffer *out;
} Record;

typedef struct {
	uint32_t va;
	Record *records;
	int count;
} Trace;

static void free_trace(Trace *trace)
{
	for (int i = 0; i < trace->count; ++i) {
		Record *record = &trace->records[i];
		for (int j = 0; j < record->n_in; ++j)
			free(record->in[j].bytes);
		for (int j = 0; j < record->n_out; ++j)
			free(record->out[j].bytes);
		free(record->in);
		free(record->out);
	}
	free(trace->records);
}


static Buffer *find_buffer(Buffer *buffers, int count, uint32_t tag)
{
	for (int i = 0; i < count; ++i) {
		if (buffers[i].tag == tag)
			return &buffers[i];
	}
	return NULL;
}

static uint32_t read_u32(FILE *file)
{
	uint8_t bytes[4];
	if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
		fprintf(stderr, "unexpected EOF reading u32\n");
		exit(2);
	}
	return (uint32_t)bytes[0]
		| (uint32_t)bytes[1] << 8
		| (uint32_t)bytes[2] << 16
		| (uint32_t)bytes[3] << 24;
}

static uint16_t read_u16(FILE *file)
{
	uint8_t bytes[2];
	if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
		fprintf(stderr, "unexpected EOF reading u16\n");
		exit(2);
	}
	return (uint16_t)(bytes[0] | bytes[1] << 8);
}

static void read_buffers(FILE *file, int count, Buffer *buffers)
{
	for (int i = 0; i < count; ++i) {
		buffers[i].tag = read_u32(file);
		buffers[i].addr = read_u32(file);
		buffers[i].len = read_u32(file);
		buffers[i].bytes = malloc(buffers[i].len);
		if (buffers[i].bytes == NULL) {
			fprintf(stderr, "out of memory reading trace\n");
			exit(2);
		}
		if (fread(buffers[i].bytes, 1, buffers[i].len, file)
			!= buffers[i].len) {
			fprintf(stderr, "unexpected EOF reading trace buffer\n");
			exit(2);
		}
	}
}

static Trace load_trace(const char *path)
{
	FILE *file = fopen(path, "rb");
	char magic[8];
	Trace trace;

	if (file == NULL) {
		fprintf(stderr, "cannot open %s\n", path);
		exit(2);
	}
	if (fread(magic, 1, sizeof(magic), file) != sizeof(magic)
		|| memcmp(magic, TMNF_TRACE_MAGIC, sizeof(magic)) != 0) {
		fprintf(stderr, "bad trace magic in %s\n", path);
		exit(2);
	}
	trace.va = read_u32(file);
	trace.count = (int)read_u32(file);
	trace.records = calloc((size_t)trace.count, sizeof(*trace.records));
	if (trace.records == NULL) {
		fprintf(stderr, "out of memory reading records\n");
		exit(2);
	}
	for (int i = 0; i < trace.count; ++i) {
		Record *record = &trace.records[i];
		record->seq = read_u32(file);
		record->n_in = read_u16(file);
		record->n_out = read_u16(file);
		record->in = calloc((size_t)record->n_in, sizeof(*record->in));
		record->out = calloc((size_t)record->n_out, sizeof(*record->out));
		if (record->in == NULL || record->out == NULL) {
			fprintf(stderr, "out of memory reading buffers\n");
			exit(2);
		}
		read_buffers(file, record->n_in, record->in);
		read_buffers(file, record->n_out, record->out);
	}
	fclose(file);
	return trace;
}

static int replay_input_mapping(const Record *record)
{
	const Buffer *input = find_buffer(record->in, record->n_in, TAG_ARG1);
	const Buffer *vehicle_input =
		find_buffer(record->in, record->n_in, TAG_ARG2);
	const Buffer *expected =
		find_buffer(record->out, record->n_out, TAG_OUT_ARG2);
	CSceneVehicleCar vehicle;
	uint8_t actual[0x0c];

	if (input == NULL || input->len != sizeof(TMNFRaceInputs)
		|| vehicle_input == NULL || vehicle_input->len != sizeof(actual)
		|| expected == NULL || expected->len != sizeof(actual)) {
		fprintf(stderr, "invalid 0x004FE500 trace record shape\n");
		exit(2);
	}
	memset(&vehicle, 0, sizeof(vehicle));
	memcpy(&vehicle.input_gas, vehicle_input->bytes, sizeof(actual));
	CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
		(const TMNFRaceInputs *)input->bytes, &vehicle);
	memcpy(actual, &vehicle.input_gas, sizeof(actual));
	return memcmp(actual, expected->bytes, sizeof(actual)) == 0;
}

static int replay_update_turbo(const Record *record)
{
	const Buffer *input = find_buffer(record->in, record->n_in, TAG_THIS);
	const Buffer *tick = find_buffer(record->in, record->n_in, TAG_SCALAR1);
	const Buffer *expected =
		find_buffer(record->out, record->n_out, TAG_OUT_THIS);
	CSceneVehicleCarAuxContext context;
	uint8_t actual[TMNF_CSCENE_VEHICLE_CAR_GAME_SIZE];
	uint32_t tick_value;

	if (input == NULL || input->len != sizeof(actual)
		|| tick == NULL || tick->len != sizeof(tick_value)
		|| expected == NULL || expected->len != sizeof(actual)) {
		fprintf(stderr, "invalid 0x007BC8B0 trace record shape\n");
		exit(2);
	}
	memset(&context, 0, sizeof(context));
	memcpy(actual, input->bytes, sizeof(actual));
	memcpy(&tick_value, tick->bytes, sizeof(tick_value));
	memcpy(&context.turbo_progress, input->bytes + 0x5f0, 4);
	memcpy(&context.turbo_start_tick, input->bytes + 0x5f8, 4);
	memcpy(&context.turbo_end_tick, input->bytes + 0x5fc, 4);
	memcpy(&context.turbo_type, input->bytes + 0x600, 4);
	CSceneVehicleCar_UpdateTurbo(&context, tick_value);
	memcpy(actual + 0x5f0, &context.turbo_progress, 4);
	memcpy(actual + 0x600, &context.turbo_type, 4);
	return memcmp(actual, expected->bytes, sizeof(actual)) == 0;
}

static int replay_wheel_integrate(const Record *record)
{
	static int reports;
	const Buffer *input = find_buffer(record->in, record->n_in, TAG_THIS);
	const Buffer *dt = find_buffer(record->in, record->n_in, TAG_SCALAR1);
	const Buffer *expected =
		find_buffer(record->out, record->n_out, TAG_OUT_THIS);
	CSceneVehicleCarWheelRealTimeState state;
	float dt_value;

	if (input == NULL || input->len != sizeof(state)
		|| dt == NULL || dt->len != sizeof(dt_value)
		|| expected == NULL || expected->len != sizeof(state)) {
		fprintf(stderr, "invalid 0x007C1060 trace record shape\n");
		exit(2);
	}
	memcpy(&state, input->bytes, sizeof(state));
	memcpy(&dt_value, dt->bytes, sizeof(dt_value));
	CSceneVehicleCarWheelRealTimeState_Integrate(&state, dt_value);
	if (reports < 3 && memcmp(&state, expected->bytes, sizeof(state)) != 0) {
		const uint8_t *actual = (const uint8_t *)&state;

		++reports;
		for (size_t i = 0; i < sizeof(state); ++i) {
			if (actual[i] != expected->bytes[i]) {
				fprintf(stderr,
					"wheel first diff byte %zu: actual=%02x expected=%02x\n",
					i, actual[i], expected->bytes[i]);
				break;
			}
		}
	}
	return memcmp(&state, expected->bytes, sizeof(state)) == 0;
}

static int replay_wheel_speed(const Record *record)
{
	const Buffer *inputs = find_buffer(record->in, record->n_in, TAG_THIS);
	const Buffer *radius = find_buffer(record->in, record->n_in, TAG_ARG1);
	const Buffer *wheel_state = find_buffer(record->in, record->n_in, TAG_ARG2);
	const Buffer *flags = find_buffer(record->in, record->n_in, TAG_ARG3);
	const Buffer *forced_speed = find_buffer(record->in, record->n_in, TAG_ARG4);
	const Buffer *vehicle_speed =
		find_buffer(record->in, record->n_in, TAG_SCALAR1);
	const Buffer *dt = find_buffer(record->in, record->n_in, TAG_SCALAR2);
	const Buffer *expected =
		find_buffer(record->out, record->n_out, TAG_OUT_ARG1);
	CSceneVehicleCar vehicle;
	CSceneVehicleCarWheel wheel;
	float vehicle_speed_value;
	float dt_value;

	if (inputs == NULL || inputs->len != 8
		|| radius == NULL || radius->len != 4
		|| wheel_state == NULL || wheel_state->len != 8
		|| flags == NULL || flags->len != 0x134
		|| forced_speed == NULL || forced_speed->len != 4
		|| vehicle_speed == NULL || vehicle_speed->len != 4
		|| dt == NULL || dt->len != 4
		|| expected == NULL || expected->len != 4) {
		fprintf(stderr, "invalid 0x007C0EC0 trace record shape\n");
		exit(2);
	}
	memset(&vehicle, 0, sizeof(vehicle));
	memset(&wheel, 0, sizeof(wheel));
	memcpy(&vehicle.input_gas, inputs->bytes, 8);
	memcpy(&wheel.radius, radius->bytes, 4);
	memcpy(&wheel.real_time.field6c, wheel_state->bytes, 8);
	memcpy(&vehicle.flag_60c, flags->bytes, 4);
	memcpy(&vehicle.force_wheel_speed, flags->bytes + 0x94, 4);
	memcpy(&vehicle.block_wheel_speed, flags->bytes + 0x130, 4);
	memcpy(&vehicle.forced_wheel_speed, forced_speed->bytes, 4);
	memcpy(&vehicle_speed_value, vehicle_speed->bytes, 4);
	memcpy(&dt_value, dt->bytes, 4);
	CSceneVehicleCar_WheelUpdateSpeedFromVehicleSpeed(
		&vehicle, &wheel, vehicle_speed_value, dt_value);
	return memcmp(
		&wheel.real_time.field6c, expected->bytes, expected->len) == 0;
}

static int replay_spring_integrate(const Record *record)
{
	const Buffer *input = find_buffer(record->in, record->n_in, TAG_THIS);
	const Buffer *dt = find_buffer(record->in, record->n_in, TAG_SCALAR1);
	const Buffer *expected =
		find_buffer(record->out, record->n_out, TAG_OUT_THIS);
	GmSpringFloat spring;
	float dt_value;

	if (input == NULL || input->len != sizeof(spring)
		|| dt == NULL || dt->len != sizeof(dt_value)
		|| expected == NULL || expected->len != sizeof(spring)) {
		fprintf(stderr, "invalid 0x008F46C0 trace record shape\n");
		exit(2);
	}
	memcpy(&spring, input->bytes, sizeof(spring));
	memcpy(&dt_value, dt->bytes, sizeof(dt_value));
	GmSpringFloat_Integrate(&spring, dt_value);
	return memcmp(&spring, expected->bytes, sizeof(spring)) == 0;
}

static int replay_compute_impulse(const Record *record)
{
	const Buffer *mass = find_buffer(record->in, record->n_in, TAG_SCALAR1);
	const Buffer *matrix = find_buffer(record->in, record->n_in, TAG_ARG1);
	const Buffer *restitution =
		find_buffer(record->in, record->n_in, TAG_SCALAR2);
	const Buffer *relative_speed =
		find_buffer(record->in, record->n_in, TAG_ARG2);
	const Buffer *normal = find_buffer(record->in, record->n_in, TAG_ARG3);
	const Buffer *lever_arm = find_buffer(record->in, record->n_in, TAG_ARG4);
	const Buffer *expected =
		find_buffer(record->out, record->n_out, TAG_OUT_THIS);
	GmMat3 inverse_inertia;
	GmVec3 relative_speed_value;
	GmVec3 normal_value;
	GmVec3 lever_arm_value;
	GmVec3 impulse;
	float mass_value;
	float restitution_value;

	if (mass == NULL || mass->len != 4
		|| matrix == NULL || matrix->len != sizeof(inverse_inertia)
		|| restitution == NULL || restitution->len != 4
		|| relative_speed == NULL
		|| relative_speed->len != sizeof(relative_speed_value)
		|| normal == NULL || normal->len != sizeof(normal_value)
		|| lever_arm == NULL || lever_arm->len != sizeof(lever_arm_value)
		|| expected == NULL || expected->len != sizeof(impulse)) {
		fprintf(stderr, "invalid 0x007BD090 trace record shape\n");
		exit(2);
	}
	memcpy(&mass_value, mass->bytes, 4);
	memcpy(&inverse_inertia, matrix->bytes, sizeof(inverse_inertia));
	memcpy(&restitution_value, restitution->bytes, 4);
	memcpy(&relative_speed_value, relative_speed->bytes, sizeof(relative_speed_value));
	memcpy(&normal_value, normal->bytes, sizeof(normal_value));
	memcpy(&lever_arm_value, lever_arm->bytes, sizeof(lever_arm_value));
	SDynaMath_ComputeImpulse(
		mass_value, &inverse_inertia, restitution_value,
		&relative_speed_value, &normal_value, &lever_arm_value, &impulse);
	return memcmp(&impulse, expected->bytes, sizeof(impulse)) == 0;
}

static int replay_add_central_force(const Record *record)
{
	const Buffer *total = find_buffer(record->in, record->n_in, TAG_THIS);
	const Buffer *force = find_buffer(record->in, record->n_in, TAG_ARG1);
	const Buffer *state = find_buffer(record->in, record->n_in, TAG_ARG2);
	const Buffer *expected_total =
		find_buffer(record->out, record->n_out, TAG_OUT_THIS);
	const Buffer *expected_dyna =
		find_buffer(record->out, record->n_out, TAG_OUT_ARG1);
	CSceneVehicleCar vehicle;
	CHmsStateDyna dyna_state;

	if (total == NULL || total->len != sizeof(vehicle.total_force_added)
		|| force == NULL || force->len != sizeof(GmVec3)
		|| state == NULL || state->len != sizeof(dyna_state)
		|| expected_total == NULL
		|| expected_total->len != sizeof(vehicle.total_force_added)
		|| expected_dyna == NULL || expected_dyna->len != 24) {
		fprintf(stderr, "invalid 0x007BE310 trace record shape\n");
		exit(2);
	}
	memset(&vehicle, 0, sizeof(vehicle));
	memcpy(&vehicle.total_force_added, total->bytes, total->len);
	memcpy(&dyna_state, state->bytes, sizeof(dyna_state));
	vehicle.dyna_state = &dyna_state;
	CSceneVehicleCar_AddVehicleCentralForce(
		&vehicle, (const GmVec3 *)force->bytes);
	return memcmp(
			&vehicle.total_force_added,
			expected_total->bytes, expected_total->len) == 0
		&& memcmp(
			&dyna_state.force, expected_dyna->bytes,
			expected_dyna->len) == 0;
}

static int replay_add_vehicle_force(const Record *record)
{
	const Buffer *total = find_buffer(record->in, record->n_in, TAG_THIS);
	const Buffer *force = find_buffer(record->in, record->n_in, TAG_ARG1);
	const Buffer *point = find_buffer(record->in, record->n_in, TAG_ARG2);
	const Buffer *state = find_buffer(record->in, record->n_in, TAG_ARG3);
	const Buffer *params = find_buffer(record->in, record->n_in, TAG_ARG4);
	const Buffer *expected_total =
		find_buffer(record->out, record->n_out, TAG_OUT_THIS);
	const Buffer *expected_dyna =
		find_buffer(record->out, record->n_out, TAG_OUT_ARG1);
	CSceneVehicleCar vehicle;
	CHmsStateDyna dyna_state;
	CHmsDynaParams dyna_params;

	if (total == NULL || total->len != sizeof(vehicle.total_force_added)
		|| force == NULL || force->len != sizeof(GmVec3)
		|| point == NULL || point->len != sizeof(GmVec3)
		|| state == NULL || state->len != sizeof(dyna_state)
		|| params == NULL || params->len != sizeof(dyna_params)
		|| expected_total == NULL
		|| expected_total->len != sizeof(vehicle.total_force_added)
		|| expected_dyna == NULL || expected_dyna->len != 24) {
		fprintf(stderr, "invalid 0x007BE2C0 trace record shape\n");
		exit(2);
	}
	memset(&vehicle, 0, sizeof(vehicle));
	memcpy(&vehicle.total_force_added, total->bytes, total->len);
	memcpy(&dyna_state, state->bytes, sizeof(dyna_state));
	memcpy(&dyna_params, params->bytes, sizeof(dyna_params));
	vehicle.dyna_state = &dyna_state;
	vehicle.dyna_params = &dyna_params;
	CSceneVehicleCar_AddVehicleForce(
		&vehicle, (const GmVec3 *)force->bytes,
		(const GmVec3 *)point->bytes);
	return memcmp(
			&vehicle.total_force_added,
			expected_total->bytes, expected_total->len) == 0
		&& memcmp(
			&dyna_state.force, expected_dyna->bytes,
			expected_dyna->len) == 0;
}

static int replay_engine_integrate(const Record *record)
{
	const Buffer *car = find_buffer(record->in, record->n_in, TAG_THIS);
	const Buffer *tuning = find_buffer(record->in, record->n_in, TAG_ARG1);
	const Buffer *wheels = find_buffer(record->in, record->n_in, TAG_ARG2);
	const Buffer *ratios = find_buffer(record->in, record->n_in, TAG_ARG3);
	const Buffer *upshift = find_buffer(record->in, record->n_in, TAG_ARG4);
	const Buffer *downshift =
		find_buffer(record->in, record->n_in, TAG_SCALAR3);
	const Buffer *aux = find_buffer(record->in, record->n_in, TAG_OUT_ARG4);
	const Buffer *throttle =
		find_buffer(record->in, record->n_in, TAG_SCALAR1);
	const Buffer *dt = find_buffer(record->in, record->n_in, TAG_SCALAR2);
	const Buffer *expected =
		find_buffer(record->out, record->n_out, TAG_OUT_THIS);
	CSceneVehicleCar vehicle;
	CSceneVehicleCarTuning tuning_value;
	CSceneVehicleCarWheel wheel_values[4];
	float throttle_value;
	float dt_value;

	if (car == NULL || car->len != 0x878
		|| tuning == NULL || tuning->len != 0x3ac
		|| wheels == NULL || wheels->len != 0xbf0
		|| ratios == NULL || ratios->len != 0x18
		|| upshift == NULL || upshift->len != 0x18
		|| downshift == NULL || downshift->len != 0x18
		|| aux == NULL || aux->len != 0x18
		|| throttle == NULL || throttle->len != 4
		|| dt == NULL || dt->len != 4
		|| expected == NULL || expected->len != 0x878) {
		fprintf(stderr, "invalid 0x007BD700 trace record shape\n");
		exit(2);
	}
	memset(&vehicle, 0, sizeof(vehicle));
	memset(&tuning_value, 0, sizeof(tuning_value));
	memset(wheel_values, 0, sizeof(wheel_values));
	memcpy(&vehicle.engine_mode, car->bytes + 0x2e4, 4);
	memcpy(&vehicle.engine, car->bytes + 0x59c, sizeof(vehicle.engine));
	memcpy(&vehicle.turbo_active, car->bytes + 0x628, 4);
	memcpy(&vehicle.drive_mode, car->bytes + 0x69c, 4);
	memcpy(&vehicle.current_local_speed, car->bytes + 0x70c, 12);
	memcpy(&vehicle.engine_limit_flag, car->bytes + 0x744, 4);
	memcpy(&vehicle.gear_downshift_flag, car->bytes + 0x748, 4);
	for (int i = 0; i < 4; ++i) {
		memcpy(
			&wheel_values[i].real_time.has_ground_contact,
			wheels->bytes + i * 0x2fc + 0x124, 4);
	}
	memcpy(&tuning_value.engine_model, tuning->bytes + 0x354, 4);
	memcpy(&tuning_value.engine_rpm_reverse_accel, tuning->bytes + 0x2ec, 4);
	memcpy(&tuning_value.engine_rpm_accel, tuning->bytes + 0x2f0, 4);
	memcpy(&tuning_value.engine_rpm_decel, tuning->bytes + 0x2f4, 4);
	memcpy(&tuning_value.engine_rpm_turbo_decel, tuning->bytes + 0x31c, 4);
	memcpy(&tuning_value.engine_rpm_follow_accel, tuning->bytes + 0x320, 4);
	memcpy(&tuning_value.engine_rpm_low_accel, tuning->bytes + 0x324, 4);
	memcpy(&tuning_value.engine_rpm_high_decel, tuning->bytes + 0x328, 4);
	memcpy(&tuning_value.speed_32c, tuning->bytes + 0x32c, 4);
	memcpy(&tuning_value.speed_330, tuning->bytes + 0x330, 4);
	memcpy(&tuning_value.speed_334, tuning->bytes + 0x334, 4);
	memcpy(&tuning_value.speed_338, tuning->bytes + 0x338, 4);
	tuning_value.gear_ratios = (const float *)ratios->bytes;
	tuning_value.gear_upshift = (const float *)upshift->bytes;
	tuning_value.gear_downshift = (const float *)downshift->bytes;
	tuning_value.gear_aux = (const float *)aux->bytes;
	vehicle.tuning = &tuning_value;
	vehicle.wheels = wheel_values;
	vehicle.wheel_count = 4;
	memcpy(&throttle_value, throttle->bytes, 4);
	memcpy(&dt_value, dt->bytes, 4);
	CSceneVehicleCar_EngineIntegrate(&vehicle, throttle_value, dt_value);
	return memcmp(&vehicle.engine_mode, expected->bytes + 0x2e4, 4) == 0
		&& memcmp(
			&vehicle.engine, expected->bytes + 0x59c,
			sizeof(vehicle.engine)) == 0
		&& memcmp(
			&vehicle.engine_limit_flag,
			expected->bytes + 0x744, 4) == 0
		&& memcmp(
			&vehicle.gear_downshift_flag,
			expected->bytes + 0x748, 4) == 0;
}

static int replay_wheel_add_force(const Record *record)
{
	const Buffer *total = find_buffer(record->in, record->n_in, TAG_THIS);
	const Buffer *tuning = find_buffer(record->in, record->n_in, TAG_ARG1);
	const Buffer *wheel = find_buffer(record->in, record->n_in, TAG_ARG2);
	const Buffer *state = find_buffer(record->in, record->n_in, TAG_ARG3);
	const Buffer *params = find_buffer(record->in, record->n_in, TAG_ARG4);
	const Buffer *expected_total =
		find_buffer(record->out, record->n_out, TAG_OUT_THIS);
	const Buffer *expected_dyna =
		find_buffer(record->out, record->n_out, TAG_OUT_ARG1);
	CSceneVehicleCar vehicle;
	CSceneVehicleCarTuning tuning_value;
	CSceneVehicleCarWheel wheel_value;
	CHmsStateDyna dyna_state;
	CHmsDynaParams dyna_params;

	if (total == NULL || total->len != 12
		|| tuning == NULL || tuning->len != 0x3ac
		|| wheel == NULL || wheel->len != 0x2fc
		|| state == NULL || state->len != sizeof(dyna_state)
		|| params == NULL || params->len != sizeof(dyna_params)
		|| expected_total == NULL || expected_total->len != 12
		|| expected_dyna == NULL || expected_dyna->len != 24) {
		fprintf(stderr, "invalid 0x007C1810 trace record shape\n");
		exit(2);
	}
	memset(&vehicle, 0, sizeof(vehicle));
	memset(&tuning_value, 0, sizeof(tuning_value));
	memset(&wheel_value, 0, sizeof(wheel_value));
	memcpy(&vehicle.total_force_added, total->bytes, 12);
	memcpy(&dyna_state, state->bytes, sizeof(dyna_state));
	memcpy(&dyna_params, params->bytes, sizeof(dyna_params));
	memcpy(&tuning_value.suspension_stiffness, tuning->bytes + 0x114, 4);
	memcpy(&tuning_value.suspension_damping, tuning->bytes + 0x118, 4);
	memcpy(&tuning_value.suspension_rest_length, tuning->bytes + 0x124, 4);
	memcpy(&tuning_value.suspension_scale, tuning->bytes + 0x128, 4);
	memcpy(&tuning_value.suspension_model, tuning->bytes + 0x350, 4);
	memcpy(&wheel_value.offset_from_vehicle, wheel->bytes + 0xa8, 12);
	memcpy(
		&wheel_value.real_time.damper_absorb,
		wheel->bytes + 0xb4, 8);
	memcpy(
		&wheel_value.real_time.has_ground_contact,
		wheel->bytes + 0x124, 4);
	vehicle.tuning = &tuning_value;
	vehicle.dyna_state = &dyna_state;
	vehicle.dyna_params = &dyna_params;
	CSceneVehicleCar_WheelAddForceToVehicle(&vehicle, &wheel_value);
	return memcmp(&vehicle.total_force_added, expected_total->bytes, 12) == 0
		&& memcmp(&dyna_state.force, expected_dyna->bytes, 24) == 0;
}

int main(int argc, char **argv)
{
	DIR *directory;
	struct dirent *entry;
	int input_records = 0;
	int wheel_records = 0;
	int wheel_speed_records = 0;
	int spring_records = 0;
	int impulse_records = 0;
	int central_force_records = 0;
	int vehicle_force_records = 0;
	int engine_records = 0;
	int wheel_force_records = 0;
	int turbo_records = 0;
	int input_failures = 0;
	int wheel_failures = 0;
	int wheel_speed_failures = 0;
	int spring_failures = 0;
	int impulse_failures = 0;
	int central_force_failures = 0;
	int vehicle_force_failures = 0;
	int engine_failures = 0;
	int wheel_force_failures = 0;
	int turbo_failures = 0;
	int failures = 0;

	if (argc != 2) {
		fprintf(stderr, "usage: %s TRACE_DIR\n", argv[0]);
		return 2;
	}
	directory = opendir(argv[1]);
	if (directory == NULL) {
		fprintf(stderr, "cannot open trace directory %s\n", argv[1]);
		return 2;
	}
	while ((entry = readdir(directory)) != NULL) {
		char path[4096];
		size_t name_len = strlen(entry->d_name);
		Trace trace;

		if (name_len < 4
			|| strcmp(entry->d_name + name_len - 4, ".bin") != 0)
			continue;
		snprintf(path, sizeof(path), "%s/%s", argv[1], entry->d_name);
		trace = load_trace(path);
		if (trace.va != 0x004FE500 && trace.va != 0x007C1060
			&& trace.va != 0x007C0EC0 && trace.va != 0x008F46C0
			&& trace.va != 0x007BD090 && trace.va != 0x007BE310
			&& trace.va != 0x007BE2C0 && trace.va != 0x007BD700
			&& trace.va != 0x007C1810 && trace.va != 0x007BC8B0) {
			free_trace(&trace);
			continue;
		}
		for (int i = 0; i < trace.count; ++i) {
			int matched;

			if (trace.va == 0x004FE500) {
				++input_records;
				matched = replay_input_mapping(&trace.records[i]);
			}
			else if (trace.va == 0x007BC8B0) {
				++turbo_records;
				matched = replay_update_turbo(&trace.records[i]);
			}
			else if (trace.va == 0x007C1060) {
				++wheel_records;
				matched = replay_wheel_integrate(&trace.records[i]);
			}
			else if (trace.va == 0x007C0EC0) {
				++wheel_speed_records;
				matched = replay_wheel_speed(&trace.records[i]);
			}
			else if (trace.va == 0x008F46C0) {
				++spring_records;
				matched = replay_spring_integrate(&trace.records[i]);
			}
			else if (trace.va == 0x007BD090) {
				++impulse_records;
				matched = replay_compute_impulse(&trace.records[i]);
			}
			else if (trace.va == 0x007BE310) {
				++central_force_records;
				matched = replay_add_central_force(&trace.records[i]);
			}
			else if (trace.va == 0x007BE2C0) {
				++vehicle_force_records;
				matched = replay_add_vehicle_force(&trace.records[i]);
			}
			else if (trace.va == 0x007BD700) {
				++engine_records;
				matched = replay_engine_integrate(&trace.records[i]);
			}
			else {
				++wheel_force_records;
				matched = replay_wheel_add_force(&trace.records[i]);
			}
			if (!matched) {
				++failures;
				if (trace.va == 0x004FE500)
					++input_failures;
				else if (trace.va == 0x007BC8B0)
					++turbo_failures;
				else if (trace.va == 0x007C1060)
					++wheel_failures;
				else if (trace.va == 0x007C0EC0)
					++wheel_speed_failures;
				else if (trace.va == 0x008F46C0)
					++spring_failures;
				else if (trace.va == 0x007BD090)
					++impulse_failures;
				else if (trace.va == 0x007BE310)
					++central_force_failures;
				else if (trace.va == 0x007BE2C0)
					++vehicle_force_failures;
				else if (trace.va == 0x007BD700)
					++engine_failures;
				else
					++wheel_force_failures;
				if (failures <= 3) {
					printf("FAIL  0x%08X seq=%u\n",
						trace.va, trace.records[i].seq);
				}
			}
		}
		free_trace(&trace);
	}
	closedir(directory);
	if (input_records == 0 || wheel_records == 0
		|| wheel_speed_records == 0 || spring_records == 0
		|| impulse_records == 0 || central_force_records == 0
		|| vehicle_force_records == 0 || engine_records == 0
		|| wheel_force_records == 0 || turbo_records == 0) {
		fprintf(stderr, "missing required vehicle trace records\n");
		return 2;
	}
	printf("0x004FE500: %d records, %d matched, %d failed\n",
		input_records, input_records - input_failures, input_failures);
	printf("0x007BC8B0: %d records, %d matched, %d failed\n",
		turbo_records, turbo_records - turbo_failures, turbo_failures);
	printf("0x007C1060: %d records, %d matched, %d failed\n",
		wheel_records, wheel_records - wheel_failures, wheel_failures);
	printf("0x007C0EC0: %d records, %d matched, %d failed\n",
		wheel_speed_records,
		wheel_speed_records - wheel_speed_failures, wheel_speed_failures);
	printf("0x008F46C0: %d records, %d matched, %d failed\n",
		spring_records, spring_records - spring_failures, spring_failures);
	printf("0x007BD090: %d records, %d matched, %d failed\n",
		impulse_records, impulse_records - impulse_failures, impulse_failures);
	printf("0x007BE310: %d records, %d matched, %d failed\n",
		central_force_records,
		central_force_records - central_force_failures,
		central_force_failures);
	printf("0x007BE2C0: %d records, %d matched, %d failed\n",
		vehicle_force_records,
		vehicle_force_records - vehicle_force_failures,
		vehicle_force_failures);
	printf("0x007BD700: %d records, %d matched, %d failed\n",
		engine_records, engine_records - engine_failures, engine_failures);
	printf("0x007C1810: %d records, %d matched, %d failed\n",
		wheel_force_records,
		wheel_force_records - wheel_force_failures,
		wheel_force_failures);
	return failures != 0;
}
