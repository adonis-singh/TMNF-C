#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trace_format.h"
#include "vehicle_aux.h"

typedef struct {
	uint32_t tag;
	uint32_t len;
	uint8_t *bytes;
} Buffer;

static void fail(const char *message)
{
	fprintf(stderr, "%s\n", message);
	exit(2);
}

static uint32_t read_u32(FILE *file)
{
	uint32_t value;
	if (fread(&value, sizeof(value), 1, file) != 1)
		fail("truncated wheel integration trace");
	return value;
}

static uint16_t read_u16(FILE *file)
{
	uint16_t value;
	if (fread(&value, sizeof(value), 1, file) != 1)
		fail("truncated wheel integration trace");
	return value;
}

static Buffer read_buffer(FILE *file)
{
	Buffer buffer;
	buffer.tag = read_u32(file);
	(void)read_u32(file);
	buffer.len = read_u32(file);
	buffer.bytes = malloc(buffer.len == 0 ? 1 : buffer.len);
	if (buffer.bytes == NULL ||
		fread(buffer.bytes, 1, buffer.len, file) != buffer.len) {
		fail("truncated wheel integration buffer");
	}
	return buffer;
}

static const Buffer *find_buffer(
	const Buffer *buffers, uint16_t count, uint32_t tag)
{
	for (uint16_t i = 0; i < count; ++i) {
		if (buffers[i].tag == tag)
			return &buffers[i];
	}
	return NULL;
}

static int32_t raw_i32(const uint8_t *raw, uint32_t offset)
{
	int32_t value;
	memcpy(&value, raw + offset, 4);
	return value;
}

static float raw_f32(const uint8_t *raw, uint32_t offset)
{
	float value;
	memcpy(&value, raw + offset, 4);
	return value;
}

static void capture_location(
	void *user, void *surface_tree, const GmIso4 *location)
{
	GmIso4 *captured = user;
	(void)surface_tree;
	*captured = *location;
}

static int replay_record(
	uint32_t seq, const Buffer *inputs, uint16_t input_count,
	const Buffer *outputs, uint16_t output_count)
{
	const Buffer *car =
		find_buffer(inputs, input_count, TAG_THIS);
	const Buffer *wheel_input =
		find_buffer(inputs, input_count, TAG_ARG1);
	const Buffer *dt_input =
		find_buffer(inputs, input_count, TAG_SCALAR1);
	const Buffer *tuning_input =
		find_buffer(inputs, input_count, TAG_ARG2);
	const Buffer *wheel_expected =
		find_buffer(outputs, output_count, TAG_OUT_ARG1);
	if (car == NULL || car->len != 0x878 ||
		wheel_input == NULL || wheel_input->len != 0x2fc ||
		dt_input == NULL || dt_input->len != 4 ||
		tuning_input == NULL || tuning_input->len != 0x3ac ||
		wheel_expected == NULL || wheel_expected->len != 0x2fc) {
		fail("invalid wheel integration record shape");
	}

	CSceneVehicleCarWheel wheel;
	CSceneVehicleCarTuning vehicle_tuning;
	CSceneVehicleCar vehicle;
	CSceneVehicleCarTuningAux aux_tuning;
	CSceneVehicleCarWheelAux aux_wheel;
	CSceneVehicleCarAuxContext context;
	GmIso4 captured_location;
	float dt;
	uint8_t actual[0x2fc];
	memset(&wheel, 0, sizeof(wheel));
	wheel.active = raw_i32(wheel_input->bytes, 0x000);
	wheel.steerable = raw_i32(wheel_input->bytes, 0x004);
	wheel.radius = raw_f32(wheel_input->bytes, 0x008);
	wheel.surface_handler = (void *)(uintptr_t)1;
	memcpy(&wheel.real_time, wheel_input->bytes + 0x0b4,
		sizeof(wheel.real_time));
	memset(&vehicle_tuning, 0, sizeof(vehicle_tuning));
	vehicle_tuning.suspension_model =
		raw_i32(tuning_input->bytes, 0x350);
	vehicle_tuning.suspension_stiffness =
		raw_f32(tuning_input->bytes, 0x114);
	vehicle_tuning.suspension_damping =
		raw_f32(tuning_input->bytes, 0x118);
	vehicle_tuning.suspension_rest_length =
		raw_f32(tuning_input->bytes, 0x124);
	vehicle_tuning.suspension_scale =
		raw_f32(tuning_input->bytes, 0x128);
	memset(&vehicle, 0, sizeof(vehicle));
	vehicle.tuning = &vehicle_tuning;
	vehicle.wheels = &wheel;
	vehicle.wheel_count = 1;
	memset(&aux_tuning, 0, sizeof(aux_tuning));
	aux_tuning.suspension_follow_rate =
		raw_f32(tuning_input->bytes, 0x194);
	memcpy(&aux_wheel.surface_source,
		wheel_input->bytes + 0x010, sizeof(GmIso4));
	memcpy(&aux_wheel.surface_location,
		wheel_input->bytes + 0x040, sizeof(GmIso4));
	memset(&context, 0, sizeof(context));
	context.vehicle = &vehicle;
	context.tuning = &aux_tuning;
	context.wheels = &aux_wheel;
	context.wheel_count = 1;
	context.runtime = &captured_location;
	context.set_surface_location = capture_location;
	memcpy(&dt, dt_input->bytes, 4);

	CSceneVehicleCar_WheelIntegrate(&context, 0, dt);

	memcpy(actual, wheel_input->bytes, sizeof(actual));
	memcpy(actual + 0x040, &aux_wheel.surface_location,
		sizeof(aux_wheel.surface_location));
	memcpy(actual + 0x0b4, &wheel.real_time,
		sizeof(wheel.real_time));
	if (memcmp(actual, wheel_expected->bytes, sizeof(actual)) == 0)
		return 1;
	for (uint32_t i = 0; i < sizeof(actual); ++i) {
		if (actual[i] != wheel_expected->bytes[i]) {
			uint32_t offset = i & ~3u;
			uint32_t actual_bits;
			uint32_t expected_bits;
			memcpy(&actual_bits, actual + offset, 4);
			memcpy(&expected_bits,
				wheel_expected->bytes + offset, 4);
			fprintf(stderr,
				"seq %u wheel +0x%x actual=%08x expected=%08x\n",
				seq, offset, actual_bits, expected_bits);
			break;
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 2)
		fail("usage: replay_wheel_integrate TRACE");
	FILE *file = fopen(argv[1], "rb");
	if (file == NULL)
		fail("cannot open wheel integration trace");
	char magic[8];
	if (fread(magic, 1, 8, file) != 8 ||
		memcmp(magic, TMNF_TRACE_MAGIC, 8) != 0 ||
		read_u32(file) != 0x007bd3f0u) {
		fail("invalid wheel integration trace header");
	}
	uint32_t record_count = read_u32(file);
	uint32_t passed = 0;
	for (uint32_t record = 0; record < record_count; ++record) {
		uint32_t seq = read_u32(file);
		uint16_t input_count = read_u16(file);
		uint16_t output_count = read_u16(file);
		Buffer *inputs = calloc(input_count, sizeof(*inputs));
		Buffer *outputs = calloc(output_count, sizeof(*outputs));
		if (inputs == NULL || outputs == NULL)
			fail("out of memory reading wheel integration trace");
		for (uint16_t i = 0; i < input_count; ++i)
			inputs[i] = read_buffer(file);
		for (uint16_t i = 0; i < output_count; ++i)
			outputs[i] = read_buffer(file);
		passed += replay_record(
			seq, inputs, input_count, outputs, output_count);
		for (uint16_t i = 0; i < input_count; ++i)
			free(inputs[i].bytes);
		for (uint16_t i = 0; i < output_count; ++i)
			free(outputs[i].bytes);
		free(inputs);
		free(outputs);
	}
	if (fgetc(file) != EOF)
		fail("trailing wheel integration trace bytes");
	fclose(file);
	printf("WheelIntegrate: %u/%u records byte-exact\n",
		passed, record_count);
	return passed == record_count ? 0 : 1;
}
