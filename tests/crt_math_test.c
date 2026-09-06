#include "gm.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define IMAGE_ASIN_SAFE UINT32_C(0x004575b0)
#define IMAGE_ACOS_DISPATCHER UINT32_C(0x009c1c40)
#define IMAGE_MATH_GUARD UINT32_C(0x00d7aff8)
#define MAX_DOMAIN_MAGNITUDE UINT32_C(0x3f800000)

typedef struct {
	const char *name;
	uint64_t tested;
	uint64_t mismatches;
	uint32_t worst_ulp;
	uint32_t first_input;
	uint32_t first_expected;
	uint32_t first_actual;
} Comparison;

static float float_from_bits(uint32_t bits)
{
	float value;
	memcpy(&value, &bits, sizeof(value));
	return value;
}

static uint32_t float_bits(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return bits;
}

static uint16_t read_u16(const unsigned char *data, size_t offset)
{
	uint16_t value;
	memcpy(&value, data + offset, sizeof(value));
	return value;
}

static uint32_t read_u32(const unsigned char *data, size_t offset)
{
	uint32_t value;
	memcpy(&value, data + offset, sizeof(value));
	return value;
}

static void load_image(const char *path)
{
	FILE *file = fopen(path, "rb");
	long file_size;
	unsigned char *image;
	uint32_t pe;
	uint16_t section_count;
	uint16_t optional_size;
	uint32_t image_base;
	size_t section_table;
	long page_size;

	if (file == NULL) {
		perror(path);
		exit(2);
	}
	if (fseek(file, 0, SEEK_END) != 0) {
		perror("fseek");
		exit(2);
	}
	file_size = ftell(file);
	if (file_size < 0 || fseek(file, 0, SEEK_SET) != 0) {
		perror("ftell/fseek");
		exit(2);
	}
	image = malloc((size_t)file_size);
	if (image == NULL
		|| fread(image, 1, (size_t)file_size, file) != (size_t)file_size) {
		fprintf(stderr, "failed to read %s\n", path);
		exit(2);
	}
	fclose(file);

	pe = read_u32(image, 0x3c);
	section_count = read_u16(image, pe + 6);
	optional_size = read_u16(image, pe + 20);
	image_base = read_u32(image, pe + 24 + 28);
	section_table = (size_t)pe + 24 + optional_size;
	page_size = sysconf(_SC_PAGESIZE);
	if (image_base != UINT32_C(0x00400000) || page_size <= 0) {
		fprintf(stderr, "unexpected PE image layout\n");
		exit(2);
	}

	for (uint16_t i = 0; i < section_count; ++i) {
		size_t header = section_table + (size_t)i * 40;
		uint32_t virtual_size = read_u32(image, header + 8);
		uint32_t virtual_address = read_u32(image, header + 12);
		uint32_t raw_size = read_u32(image, header + 16);
		uint32_t raw_offset = read_u32(image, header + 20);
		uint32_t characteristics = read_u32(image, header + 36);
		uint32_t mapped_size =
			virtual_size > raw_size ? virtual_size : raw_size;
		uintptr_t address = (uintptr_t)image_base + virtual_address;
		uintptr_t page = address & ~((uintptr_t)page_size - 1);
		size_t leading = address - page;
		size_t length =
			(leading + mapped_size + (size_t)page_size - 1)
			& ~((size_t)page_size - 1);
		void *mapping = mmap(
			(void *)page, length, PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
		int protection = 0;

		if (mapping == MAP_FAILED) {
			fprintf(stderr, "mmap 0x%08" PRIxPTR ": %s\n",
				page, strerror(errno));
			exit(2);
		}
		if ((uint64_t)raw_offset + raw_size > (uint64_t)file_size) {
			fprintf(stderr, "truncated PE section\n");
			exit(2);
		}
		memcpy((void *)address, image + raw_offset, raw_size);
		if ((characteristics & UINT32_C(0x40000000)) != 0)
			protection |= PROT_READ;
		if ((characteristics & UINT32_C(0x80000000)) != 0)
			protection |= PROT_WRITE;
		if ((characteristics & UINT32_C(0x20000000)) != 0)
			protection |= PROT_EXEC;
		if (mprotect((void *)page, length, protection) != 0) {
			perror("mprotect");
			exit(2);
		}
	}
	free(image);

	if (*(const uint32_t *)(uintptr_t)IMAGE_MATH_GUARD != 0) {
		fprintf(stderr, "image math guard is not zero\n");
		exit(2);
	}
}

static float image_call_x87_float(uint32_t address, float value)
{
	float result;
	uint16_t saved_control;
	const uint16_t game_control = UINT16_C(0x007f);

	__asm__ volatile(
		"fnstcw %[saved_control]\n\t"
		"fldcw %[game_control]\n\t"
		"flds %[value]\n\t"
		"call *%[address]\n\t"
		"fstps %[result]\n\t"
		"fldcw %[saved_control]"
		: [result] "=m" (result), [saved_control] "=m" (saved_control)
		: [value] "m" (value), [address] "r" ((uintptr_t)address),
		  [game_control] "m" (game_control)
		: "eax", "ecx", "edx", "cc", "memory",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7", "st");
	return result;
}

static float image_call_stack_float(uint32_t address, float value)
{
	float result;
	uint32_t value_bits = float_bits(value);
	uint16_t saved_control;
	const uint16_t game_control = UINT16_C(0x007f);

	__asm__ volatile(
		"fnstcw %[saved_control]\n\t"
		"fldcw %[game_control]\n\t"
		"pushl %[value_bits]\n\t"
		"call *%[address]\n\t"
		"addl $4, %%esp\n\t"
		"fstps %[result]\n\t"
		"fldcw %[saved_control]"
		: [result] "=m" (result), [saved_control] "=m" (saved_control)
		: [value_bits] "r" (value_bits),
		  [address] "r" ((uintptr_t)address),
		  [game_control] "m" (game_control)
		: "eax", "ecx", "edx", "cc", "memory",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7", "st");
	return result;
}

static uint32_t ordered_bits(uint32_t bits)
{
	return (bits & UINT32_C(0x80000000)) != 0
		? ~bits
		: bits | UINT32_C(0x80000000);
}

static uint32_t ulp_distance(uint32_t left, uint32_t right)
{
	uint32_t a = ordered_bits(left);
	uint32_t b = ordered_bits(right);
	return a > b ? a - b : b - a;
}

static void compare_result(
	Comparison *comparison, uint32_t input,
	uint32_t expected, uint32_t actual)
{
	uint32_t distance;

	comparison->tested++;
	if (actual == expected)
		return;
	distance = ulp_distance(expected, actual);
	if (comparison->mismatches == 0) {
		comparison->first_input = input;
		comparison->first_expected = expected;
		comparison->first_actual = actual;
	}
	comparison->mismatches++;
	if (distance > comparison->worst_ulp)
		comparison->worst_ulp = distance;
}

static void test_input(Comparison results[2], uint32_t bits)
{
	float value = float_from_bits(bits);

	compare_result(
		&results[0], bits,
		float_bits(image_call_stack_float(IMAGE_ASIN_SAFE, value)),
		float_bits(GmFunc_AsinSafe(value)));
	compare_result(
		&results[1], bits,
		float_bits(image_call_x87_float(IMAGE_ACOS_DISPATCHER, value)),
		float_bits(GmFunc_Acos(value)));
}

static void test_signed_magnitude(
	Comparison results[2], uint32_t magnitude)
{
	test_input(results, magnitude);
	if (magnitude != 0)
		test_input(results, magnitude | UINT32_C(0x80000000));
}

static void test_directed(Comparison results[2])
{
	static const uint32_t boundaries[] = {
		UINT32_C(0x00000000),
		UINT32_C(0x00000001),
		UINT32_C(0x007fffff),
		UINT32_C(0x00800000),
		UINT32_C(0x3d7625d0),
		UINT32_C(0x3f7fffef),
		UINT32_C(0x3f7fffff),
		UINT32_C(0x3f800000),
	};

	for (size_t i = 0; i < sizeof(boundaries) / sizeof(boundaries[0]); ++i) {
		uint32_t boundary = boundaries[i];
		for (int delta = -4; delta <= 4; ++delta) {
			uint32_t magnitude;

			if (delta < 0 && boundary < (uint32_t)-delta)
				continue;
			magnitude = boundary + (uint32_t)delta;
			if (magnitude > MAX_DOMAIN_MAGNITUDE)
				continue;
			test_signed_magnitude(results, magnitude);
		}
	}
}

static uint32_t next_random(uint32_t *state)
{
	uint32_t value = *state;
	value ^= value << 13;
	value ^= value >> 17;
	value ^= value << 5;
	*state = value;
	return value;
}

static void test_sampled(Comparison results[2], uint64_t count)
{
	uint32_t state = UINT32_C(0x243f6a88);

	for (uint64_t i = 0; i < count; ++i) {
		uint32_t random = next_random(&state);
		uint32_t magnitude = (uint32_t)(
			((uint64_t)random
				* (MAX_DOMAIN_MAGNITUDE + UINT64_C(1))) >> 32);
		test_signed_magnitude(results, magnitude);
	}
}

static void test_exhaustive(Comparison results[2])
{
	test_input(results, 0);
	for (uint32_t magnitude = 1;
		magnitude <= MAX_DOMAIN_MAGNITUDE; ++magnitude) {
		test_input(results, magnitude);
		test_input(results, magnitude | UINT32_C(0x80000000));
	}
}

static void print_result(const Comparison *result)
{
	printf("%s: tested=%" PRIu64 " mismatches=%" PRIu64
		" worst_ulp=%" PRIu32 "\n",
		result->name, result->tested,
		result->mismatches, result->worst_ulp);
	if (result->mismatches != 0) {
		printf("%s first: input=0x%08" PRIx32
			" expected=0x%08" PRIx32 " actual=0x%08" PRIx32 "\n",
			result->name, result->first_input,
			result->first_expected, result->first_actual);
	}
}

int main(int argc, char **argv)
{
	Comparison results[2] = {
		{ .name = "asin-safe vs 0x004575b0" },
		{ .name = "acos vs 0x009c1c40" },
	};
	uint32_t regression = UINT32_C(0x3d7625d0);
	uint32_t dispatcher_regression;

	if (argc < 2 || argc > 3) {
		fprintf(stderr,
			"usage: %s TmForever.exe [sample-count|exhaustive]\n",
			argv[0]);
		return 2;
	}
	load_image(argv[1]);
	dispatcher_regression = float_bits(image_call_x87_float(
		UINT32_C(0x009c1d90), float_from_bits(regression)));
	if (dispatcher_regression != UINT32_C(0x3d764bce)) {
		fprintf(stderr,
			"asin dispatcher regression: got 0x%08" PRIx32
			", expected 0x3d764bce\n", dispatcher_regression);
		return 2;
	}
	printf("asin dispatcher: input=0x3d7625d0 output=0x%08" PRIx32 "\n",
		dispatcher_regression);

	if (argc == 3 && strcmp(argv[2], "exhaustive") == 0) {
		test_exhaustive(results);
	} else {
		uint64_t count = argc == 3
			? strtoull(argv[2], NULL, 0)
			: UINT64_C(1000000);
		test_directed(results);
		test_sampled(results, count);
	}

	print_result(&results[0]);
	print_result(&results[1]);
	return results[0].mismatches != 0 || results[1].mismatches != 0;
}
