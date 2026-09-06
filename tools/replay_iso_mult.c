#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gm.h"
#include "trace_format.h"

typedef struct {
	uint32_t tag;
	uint32_t size;
	uint8_t *bytes;
} Buffer;

static void fail(const char *message)
{
	fprintf(stderr, "replay_iso_mult: %s\n", message);
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
	fail("missing buffer");
	return NULL;
}

static void iso_mult(GmIso4 *self, const GmIso4 *parent)
{
	GmMat3 matrix;
	GmVec3 translation = {
		self->t[0],
		self->t[1],
		self->t[2],
	};
	memcpy(matrix.m, self->m, sizeof(matrix.m));
	GmMat3_Mult(&matrix, (const GmMat3 *)parent);
	memcpy(self->m, matrix.m, sizeof(matrix.m));
	GmVec3_Mult_Iso4(&translation, parent);
	self->t[0] = translation.x;
	self->t[1] = translation.y;
	self->t[2] = translation.z;
}

int main(int argc, char **argv)
{
	if (argc != 2)
		fail("usage: replay_iso_mult TRACE");
	FILE *file = fopen(argv[1], "rb");
	if (file == NULL)
		fail("cannot open trace");
	char magic[8];
	if (fread(magic, 8, 1, file) != 1 ||
		memcmp(magic, TMNF_TRACE_MAGIC, 8) != 0) {
		fail("wrong trace type");
	}
	uint32_t va = read_u32(file);
	if (va != 0x008E2570u && va != 0x008E2970u)
		fail("wrong trace type");
	uint32_t count = read_u32(file);
	uint32_t failures = 0;
	for (uint32_t record = 0; record < count; ++record) {
		uint32_t sequence = read_u32(file);
		uint16_t input_count = read_u16(file);
		uint16_t output_count = read_u16(file);
		Buffer *inputs = calloc(input_count, sizeof(*inputs));
		Buffer *outputs = calloc(output_count, sizeof(*outputs));
		if (inputs == NULL || outputs == NULL)
			fail("out of memory");
		for (uint16_t i = 0; i < input_count; ++i)
			inputs[i] = read_buffer(file);
		for (uint16_t i = 0; i < output_count; ++i)
			outputs[i] = read_buffer(file);
		Buffer *self = find_buffer(inputs, input_count, TAG_THIS);
		Buffer *parent = find_buffer(inputs, input_count, TAG_ARG1);
		Buffer *expected =
			find_buffer(outputs, output_count, TAG_OUT_THIS);
		if (self->size != 48 || parent->size != 48 ||
			expected->size != 48) {
			fail("invalid record shape");
		}
		GmIso4 actual;
		memcpy(&actual, self->bytes, sizeof(actual));
		const GmIso4 *expected_iso =
			(const GmIso4 *)expected->bytes;
		if (199.9f < expected_iso->t[0] &&
			expected_iso->t[0] < 200.1f) {
			printf(
				"candidate seq=%u m=%a,%a,%a t=%a,%a,%a\n",
				sequence,
				expected_iso->m[0], expected_iso->m[1],
				expected_iso->m[2], expected_iso->t[0],
				expected_iso->t[1], expected_iso->t[2]);
		}
		if (va == 0x008E2570u)
			GmIso4_SetInverse(
				&actual, (const GmIso4 *)parent->bytes);
		else
			iso_mult(&actual, (const GmIso4 *)parent->bytes);
		if (memcmp(&actual, expected->bytes, sizeof(actual)) != 0) {
			if (failures < 10) {
				uint32_t word = 0;
				while (((uint32_t *)&actual)[word] ==
					((uint32_t *)expected->bytes)[word]) {
					++word;
				}
				printf(
					"FAIL seq=%u word=%u actual=%08x expected=%08x\n",
					sequence, word, ((uint32_t *)&actual)[word],
					((uint32_t *)expected->bytes)[word]);
			}
			++failures;
		}
		for (uint16_t i = 0; i < input_count; ++i)
			free(inputs[i].bytes);
		for (uint16_t i = 0; i < output_count; ++i)
			free(outputs[i].bytes);
		free(inputs);
		free(outputs);
	}
	printf("isometry 0x%08x: %u/%u records byte-exact\n",
		va, count - failures, count);
	return failures == 0 ? 0 : 1;
}
