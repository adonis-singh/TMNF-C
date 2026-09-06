#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trace_format.h"
#include "vehicle.h"

typedef struct {
	uint32_t tag;
	uint32_t size;
	uint8_t *bytes;
} Buffer;

static void fail(const char *message)
{
	fprintf(stderr, "diagnose_wheel_impulse: %s\n", message);
	exit(2);
}

static uint32_t read_u32(FILE *file)
{
	uint32_t value;
	if (fread(&value, 4, 1, file) != 1)
		fail("truncated u32");
	return value;
}

static uint16_t read_u16(FILE *file)
{
	uint16_t value;
	if (fread(&value, 2, 1, file) != 1)
		fail("truncated u16");
	return value;
}

static Buffer read_buffer(FILE *file)
{
	Buffer buffer;
	buffer.tag = read_u32(file);
	(void)read_u32(file);
	buffer.size = read_u32(file);
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
	fail("missing trace buffer");
	return NULL;
}

static float raw_f32(const uint8_t *bytes, uint32_t offset)
{
	float value;
	memcpy(&value, bytes + offset, 4);
	return value;
}

static uint32_t raw_u32(const uint8_t *bytes, uint32_t offset)
{
	uint32_t value;
	memcpy(&value, bytes + offset, 4);
	return value;
}

static uint32_t bits(float value)
{
	uint32_t result;
	memcpy(&result, &value, 4);
	return result;
}

int main(int argc, char **argv)
{
	if (argc != 3)
		fail("usage: diagnose_wheel_impulse TRACE SEQUENCE");
	uint32_t wanted = (uint32_t)strtoul(argv[2], NULL, 10);
	FILE *file = fopen(argv[1], "rb");
	if (file == NULL)
		fail("cannot open trace");
	char magic[8];
	if (fread(magic, sizeof(magic), 1, file) != 1 ||
		memcmp(magic, TMNF_TRACE_MAGIC, sizeof(magic)) != 0 ||
		read_u32(file) != 0x007C11D0u) {
		fail("wrong trace type");
	}
	uint32_t count = read_u32(file);
	for (uint32_t record = 0; record < count; ++record) {
		uint32_t sequence = read_u32(file);
		uint16_t input_count = read_u16(file);
		uint16_t output_count = read_u16(file);
		Buffer *inputs = calloc(input_count, sizeof(*inputs));
		if (inputs == NULL)
			fail("out of memory");
		for (uint16_t index = 0; index < input_count; ++index)
			inputs[index] = read_buffer(file);
		for (uint16_t index = 0; index < output_count; ++index) {
			Buffer output = read_buffer(file);
			free(output.bytes);
		}
		if (sequence != wanted) {
			for (uint16_t index = 0; index < input_count; ++index)
				free(inputs[index].bytes);
			free(inputs);
			continue;
		}
		Buffer *wheel = find_buffer(inputs, input_count, TAG_ARG1);
		Buffer *contact = find_buffer(inputs, input_count, TAG_ARG2);
		Buffer *tuning = find_buffer(inputs, input_count, TAG_ARG3);
		Buffer *params = find_buffer(inputs, input_count, TAG_ARG4);
		GmVec3 normal;
		GmVec3 relative_speed;
		GmVec3 points[3];
		GmVec3 lever;
		GmVec3 impulse;
		CHmsDynaParams dyna_params;
		memcpy(&normal, contact->bytes + 0x0c, 12);
		memcpy(&relative_speed, contact->bytes + 0x24, 12);
		memcpy(&points[0], contact->bytes + 0x18, 12);
		memcpy(&points[1], wheel->bytes + 0x64, 12);
		memcpy(&dyna_params, params->bytes, sizeof(dyna_params));
		points[2] = points[0];
		points[2].y = dyna_params.comOffset.y;
		uint32_t material = raw_u32(contact->bytes, 0x48);
		float restitution = raw_f32(
			tuning->bytes, material == 4 ? 0x178 : 0x17c);
		printf(
			"center=(%.9g,%.9g,%.9g) restitution=%.9g\n",
			dyna_params.comOffset.x, dyna_params.comOffset.y,
			dyna_params.comOffset.z, restitution);
		for (uint32_t point_index = 0; point_index < 3; ++point_index) {
			lever.x = points[point_index].x - dyna_params.comOffset.x;
			lever.y = points[point_index].y - dyna_params.comOffset.y;
			lever.z = points[point_index].z - dyna_params.comOffset.z;
			SDynaMath_ComputeImpulse(
				dyna_params.mass, &dyna_params.invInertiaBody,
				-restitution, &relative_speed, &normal, &lever,
				&impulse);
			printf(
				"%s point=(%.9g,%.9g,%.9g) "
				"impulse=%08x,%08x,%08x\n",
				point_index == 0 ? "contact"
					: point_index == 1 ? "wheel" : "side",
				points[point_index].x, points[point_index].y,
				points[point_index].z,
				bits(impulse.x), bits(impulse.y),
				bits(impulse.z));
		}
		return 0;
	}
	fail("sequence not found");
	return 2;
}
