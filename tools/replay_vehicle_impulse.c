#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trace_format.h"
#include "vehicle_contact.h"

typedef struct {
	uint32_t tag;
	uint32_t address;
	uint32_t size;
	uint8_t *bytes;
} Buffer;

static void fail(const char *message)
{
	fprintf(stderr, "replay_vehicle_impulse: %s\n", message);
	exit(2);
}

static uint32_t read_u32(FILE *file)
{
	uint32_t value;
	if (fread(&value, sizeof(value), 1, file) != 1)
		fail("truncated u32");
	return value;
}

static uint16_t read_u16(FILE *file)
{
	uint16_t value;
	if (fread(&value, sizeof(value), 1, file) != 1)
		fail("truncated u16");
	return value;
}

static Buffer read_buffer(FILE *file)
{
	Buffer buffer = {
		.tag = read_u32(file),
		.address = read_u32(file),
		.size = read_u32(file),
	};
	buffer.bytes = malloc(buffer.size);
	if (buffer.bytes == NULL)
		fail("out of memory");
	if (fread(buffer.bytes, buffer.size, 1, file) != 1)
		fail("truncated buffer");
	return buffer;
}

static Buffer *find_buffer(Buffer *buffers, uint16_t count, uint32_t tag)
{
	for (uint16_t index = 0; index < count; ++index) {
		if (buffers[index].tag == tag)
			return &buffers[index];
	}
	return NULL;
}

static float raw_f32(const uint8_t *bytes, uint32_t offset)
{
	float value;
	memcpy(&value, bytes + offset, sizeof(value));
	return value;
}

static void report_mismatch(
	uint32_t sequence, const char *name, const void *actual_data,
	const void *expected_data, uint32_t size)
{
	const uint8_t *actual = actual_data;
	const uint8_t *expected = expected_data;
	uint32_t offset = 0;
	while (offset < size && actual[offset] == expected[offset])
		++offset;
	uint32_t word = offset & ~3u;
	uint32_t actual_word;
	uint32_t expected_word;
	memcpy(&actual_word, actual + word, 4);
	memcpy(&expected_word, expected + word, 4);
	printf(
		"FAIL seq=%u %s byte=%u actual=%08x expected=%08x\n",
		sequence, name, offset, actual_word, expected_word);
}

int main(int argc, char **argv)
{
	if (argc != 2)
		fail("usage: replay_vehicle_impulse TRACE");
	FILE *file = fopen(argv[1], "rb");
	if (file == NULL)
		fail("cannot open trace");
	char magic[8];
	if (fread(magic, sizeof(magic), 1, file) != 1 ||
		memcmp(magic, TMNF_TRACE_MAGIC, sizeof(magic)) != 0) {
		fail("bad trace magic");
	}
	if (read_u32(file) != 0x007BE390u)
		fail("trace has the wrong target VA");
	uint32_t record_count = read_u32(file);
	uint32_t failures = 0;
	for (uint32_t record_index = 0;
	     record_index < record_count; ++record_index) {
		uint32_t sequence = read_u32(file);
		uint16_t input_count = read_u16(file);
		uint16_t output_count = read_u16(file);
		Buffer *inputs = calloc(input_count, sizeof(*inputs));
		Buffer *outputs = calloc(output_count, sizeof(*outputs));
		if (inputs == NULL || outputs == NULL)
			fail("out of memory");
		for (uint16_t index = 0; index < input_count; ++index)
			inputs[index] = read_buffer(file);
		for (uint16_t index = 0; index < output_count; ++index)
			outputs[index] = read_buffer(file);

		Buffer *car = find_buffer(inputs, input_count, TAG_THIS);
		Buffer *impulse = find_buffer(inputs, input_count, TAG_ARG1);
		Buffer *point = find_buffer(inputs, input_count, TAG_ARG2);
		Buffer *state_input = find_buffer(inputs, input_count, TAG_ARG3);
		Buffer *params_input = find_buffer(inputs, input_count, TAG_ARG4);
		Buffer *raw_tuning =
			find_buffer(inputs, input_count, TAG_SCALAR1);
		Buffer *total_expected =
			find_buffer(outputs, output_count, TAG_OUT_THIS);
		Buffer *state_expected =
			find_buffer(outputs, output_count, TAG_OUT_ARG1);
		if (car == NULL || car->size != 0x878 ||
			impulse == NULL || impulse->size != sizeof(GmVec3) ||
			point == NULL || point->size != sizeof(GmVec3) ||
			state_input == NULL ||
				state_input->size != sizeof(CHmsStateDyna) ||
			params_input == NULL ||
				params_input->size != sizeof(CHmsDynaParams) ||
			raw_tuning == NULL || raw_tuning->size != 0x3ac ||
			total_expected == NULL ||
				total_expected->size != sizeof(GmVec3) ||
			state_expected == NULL ||
				state_expected->size != sizeof(CHmsStateDyna)) {
			fail("invalid record shape");
		}

		CHmsStateDyna state;
		CHmsDynaParams params;
		CSceneVehicleCar vehicle = { 0 };
		TMNFVehicleContactTuning tuning = { 0 };
		TMNFVehicleContactContext context = { 0 };
		memcpy(&state, state_input->bytes, sizeof(state));
		memcpy(&params, params_input->bytes, sizeof(params));
		vehicle.dyna_state = &state;
		vehicle.dyna_params = &params;
		memcpy(&vehicle.total_impulse_added, car->bytes + 0x824, 12);
		tuning.angular_y_scale = raw_f32(raw_tuning->bytes, 0x0e8);
		tuning.angular_xz_scale = raw_f32(raw_tuning->bytes, 0x0ec);
		tuning.max_angular_speed = raw_f32(raw_tuning->bytes, 0x14c);
		tuning.max_linear_speed_delta =
			raw_f32(raw_tuning->bytes, 0x150);
		context.vehicle = &vehicle;
		context.tuning = &tuning;

		CSceneVehicleCar_AddVehicleImpulse(
			&context, (const GmVec3 *)impulse->bytes,
			(const GmVec3 *)point->bytes);
		if (memcmp(&state, state_expected->bytes, sizeof(state)) != 0) {
			if (failures < 10) {
				report_mismatch(
					sequence, "dyna", &state,
					state_expected->bytes, sizeof(state));
			}
			++failures;
		} else if (memcmp(
				   &vehicle.total_impulse_added,
				   total_expected->bytes, 12) != 0) {
			if (failures < 10) {
				report_mismatch(
					sequence, "total impulse",
					&vehicle.total_impulse_added,
					total_expected->bytes, 12);
			}
			++failures;
		}
		for (uint16_t index = 0; index < input_count; ++index)
			free(inputs[index].bytes);
		for (uint16_t index = 0; index < output_count; ++index)
			free(outputs[index].bytes);
		free(inputs);
		free(outputs);
	}
	if (fclose(file) != 0)
		fail("cannot close trace");
	printf(
		"vehicle impulse: %u/%u records byte-exact\n",
		record_count - failures, record_count);
	return failures == 0 ? 0 : 1;
}
