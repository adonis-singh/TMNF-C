/* Checks the identities src/tmnf_fp.h relies on: for binary32 operands, one
 * operation carried out in binary64 and rounded to binary32 equals the
 * directly rounded binary32 operation. sqrt is checked for every non-negative
 * bit pattern;
 * + - * / for every exponent pair with pseudo-random significands. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t bits(float value) {
	uint32_t result;
	memcpy(&result, &value, sizeof(result));
	return result;
}

static float from_bits(uint32_t value) {
	float result;
	memcpy(&result, &value, sizeof(result));
	return result;
}

static uint64_t next(uint64_t *state) {
	*state ^= *state << 13;
	*state ^= *state >> 7;
	*state ^= *state << 17;
	return *state;
}

/* Two NaNs compare equal when both are NaN with the same bits, and any two
 * default NaNs count as equal since only their sign/payload could differ and
 * the game never observes NaN payloads. */
static int same(float a, float b) {
	if (isnan(a) && isnan(b)) {
		return 1;
	}
	return bits(a) == bits(b);
}

static unsigned long long failures;

static void report(const char *op, float a, float b, float wide, float narrow) {
	if (failures++ < 10) {
		fprintf(stderr, "%s %08x %08x: wide %08x narrow %08x\n", op, bits(a),
			bits(b), bits(wide), bits(narrow));
	}
}

int main(void) {
	volatile double wide_result;
	volatile float narrow_result;

	/* Every non-negative pattern; negatives (all NaN results apart from
	 * -0) strided. */
	for (uint64_t i = 0; i < UINT64_C(0x100000000);
		i += i < UINT64_C(0x80000000) ? 1 : 0x101) {
		float x = from_bits((uint32_t)i);
		wide_result = sqrt((double)x);
		narrow_result = sqrtf(x);
		if (!same((float)wide_result, narrow_result)) {
			report("sqrt", x, 0.0f, (float)wide_result, narrow_result);
		}
	}

	uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
	for (uint32_t ea = 0; ea < 256; ++ea) {
		for (uint32_t eb = 0; eb < 256; ++eb) {
			for (int k = 0; k < 64; ++k) {
				uint64_t r = next(&state);
				uint32_t ma = (uint32_t)r & 0x7fffffu;
				uint32_t mb = (uint32_t)(r >> 23) & 0x7fffffu;
				uint32_t sa = (uint32_t)(r >> 46) & 1u;
				uint32_t sb = (uint32_t)(r >> 47) & 1u;
				float a = from_bits((sa << 31) | (ea << 23) | ma);
				float b = from_bits((sb << 31) | (eb << 23) | mb);
				wide_result = (double)a + (double)b;
				narrow_result = a + b;
				if (!same((float)wide_result, narrow_result)) {
					report("add", a, b, (float)wide_result, narrow_result);
				}
				wide_result = (double)a - (double)b;
				narrow_result = a - b;
				if (!same((float)wide_result, narrow_result)) {
					report("sub", a, b, (float)wide_result, narrow_result);
				}
				wide_result = (double)a * (double)b;
				narrow_result = a * b;
				if (!same((float)wide_result, narrow_result)) {
					report("mul", a, b, (float)wide_result, narrow_result);
				}
				wide_result = (double)a / (double)b;
				narrow_result = a / b;
				if (!same((float)wide_result, narrow_result)) {
					report("div", a, b, (float)wide_result, narrow_result);
				}
				wide_result = 1.0 / (double)a;
				narrow_result = 1.0f / a;
				if (!same((float)wide_result, narrow_result)) {
					report("rcp", a, 0.0f, (float)wide_result, narrow_result);
				}
			}
		}
	}

	if (failures != 0) {
		fprintf(stderr, "%llu identity failures\n", failures);
		return 1;
	}
	printf("x87_fp_identity: sqrt exhaustive, 4194304 binary samples: ok\n");
	return 0;
}
