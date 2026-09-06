#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hms_dyna.h"
#include "trace_format.h"

typedef struct {
	uint32_t tag;
	uint32_t size;
	uint8_t *bytes;
} Buffer;

static void fail(const char *message)
{
	fprintf(stderr, "replay_replacement: %s\n", message);
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
	Buffer buffer;
	buffer.tag = read_u32(file);
	(void)read_u32(file);
	buffer.size = read_u32(file);
	buffer.bytes = malloc(buffer.size == 0 ? 1 : buffer.size);
	if (buffer.bytes == NULL)
		fail("out of memory");
	if (buffer.size != 0 &&
		fread(buffer.bytes, buffer.size, 1, file) != 1) {
		fail("truncated buffer");
	}
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

static uint32_t raw_u32(const uint8_t *bytes)
{
	uint32_t value;
	memcpy(&value, bytes, sizeof(value));
	return value;
}

static uint32_t bits(float value)
{
	uint32_t result;
	memcpy(&result, &value, sizeof(result));
	return result;
}

int main(int argc, char **argv)
{
	if (argc != 2)
		fail("usage: replay_replacement TRACE");
	FILE *file = fopen(argv[1], "rb");
	if (file == NULL)
		fail("cannot open trace");
	char magic[8];
	if (fread(magic, sizeof(magic), 1, file) != 1 ||
		memcmp(magic, TMNF_TRACE_MAGIC, sizeof(magic)) != 0 ||
		read_u32(file) != 0x00535D50u) {
		fail("wrong trace type");
	}
	uint32_t record_count = read_u32(file);
	uint32_t failures = 0;
	for (uint32_t record = 0; record < record_count; ++record) {
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
		Buffer *metadata = find_buffer(inputs, input_count, TAG_THIS);
		Buffer *replacements = find_buffer(inputs, input_count, TAG_ARG1);
		Buffer *expected =
			find_buffer(outputs, output_count, TAG_OUT_THIS);
		if (metadata == NULL || metadata->size != 12 ||
			replacements == NULL || expected == NULL ||
			expected->size != sizeof(GmVec3)) {
			fail("invalid record shape");
		}
		uint32_t count = raw_u32(metadata->bytes);
		if (replacements->size != count * sizeof(GmVec3))
			fail("replacement count differs from payload");
		CHmsDyna dyna = { 0 };
		dyna.replacementBuf.count = count;
		dyna.replacementBuf.data = (GmVec3 *)replacements->bytes;
		dyna.replacementBuf.capacity = count;
		GmVec3 actual;
		CHmsDyna_ComputeSynthetizedReplacement(&dyna, &actual);
		if (memcmp(&actual, expected->bytes, sizeof(actual)) != 0) {
			if (failures < 10) {
				const GmVec3 *want = (const GmVec3 *)expected->bytes;
				printf(
					"FAIL seq=%u count=%u "
					"actual=%08x,%08x,%08x "
					"expected=%08x,%08x,%08x\n",
					sequence, count,
					bits(actual.x), bits(actual.y), bits(actual.z),
					bits(want->x), bits(want->y), bits(want->z));
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
		"replacement synthesis: %u/%u records byte-exact\n",
		record_count - failures, record_count);
	return failures == 0 ? 0 : 1;
}
