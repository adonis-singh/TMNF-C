/* Enumerates every float input of the device transcendentals used by the
 * physics port and compares the device result, rounded to float, with this
 * machine's glibc. Inputs where they differ are the hard cases that
 * src/cuda/tmnf_dev_libm.h corrects.
 *
 *   cuda_libm_hardcases generate OUTPUT.h   write the merged table
 *   cuda_libm_hardcases verify              expect zero residual mismatches
 *
 * The device kernels already apply the compiled-in table, so "generate" finds
 * the residual mismatches, merges them into the table it was built with, and
 * writes the result; a rebuild followed by "verify" is the proof. Both scans
 * cover all 2^32 bit patterns of the argument. atan2 is only reached from
 * GmFunc_Acos and GmFunc_AsinSafe with (sqrt((1+v)(1-v)), v) and
 * (v, sqrt((1+v)(1-v))), so its domain is enumerated through v.
 *
 * NaN results compare equal regardless of payload: no valid race produces
 * one, and x86 and CUDA propagate payloads differently in plain arithmetic
 * anyway.
 */
#include <gnu/libc-version.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <vector>

#include <cuda_runtime.h>

#include "tmnf_fp.h"
#include "cuda/tmnf_dev_libm.h"

enum {
	CHUNK_BITS = 26,
	CHUNK = 1u << CHUNK_BITS,
	CHUNKS = 1u << (32 - CHUNK_BITS),
};

enum Kind {
	KIND_SIN = 0,
	KIND_COS = 1,
	KIND_EXP = 2,
	KIND_ATAN2_ACOS = 3,
	KIND_ATAN2_ASIN = 4,
	KIND_COUNT = 5,
};

static const char *KIND_NAMES[KIND_COUNT] = {
	"sin", "cos", "exp", "atan2(acos form)", "atan2(asin form)",
};

static void check(cudaError_t error, const char *what)
{
	if (error != cudaSuccess) {
		fprintf(stderr, "cuda_libm_hardcases: %s: %s\n", what,
			cudaGetErrorString(error));
		exit(2);
	}
}

__host__ __device__ static float acos_root(float value)
{
	float product = x87_mul(x87_add(1.0f, value), x87_sub(1.0f, value));
	return x87_sqrt(product);
}

__global__ static void scan_kernel(int kind, uint32_t base, float *out)
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	float x = __uint_as_float(base + i);
	float result;
	switch (kind) {
	case KIND_SIN:
		result = x87_sin(x);
		break;
	case KIND_COS:
		result = x87_cos(x);
		break;
	case KIND_EXP:
		result = x87_exp(x);
		break;
	case KIND_ATAN2_ACOS:
		result = x87_atan2(acos_root(x), x);
		break;
	default:
		result = x87_atan2(x, acos_root(x));
		break;
	}
	out[i] = result;
}

static float host_reference(int kind, float x)
{
	switch (kind) {
	case KIND_SIN:
		return x87_sin(x);
	case KIND_COS:
		return x87_cos(x);
	case KIND_EXP:
		return x87_exp(x);
	case KIND_ATAN2_ACOS:
		return x87_atan2(acos_root(x), x);
	default:
		return x87_atan2(x, acos_root(x));
	}
}

static uint32_t bits(float value)
{
	uint32_t result;
	memcpy(&result, &value, sizeof(result));
	return result;
}

struct Mismatch {
	uint32_t input;
	uint32_t device;
	uint32_t host;
};

/* Residual mismatches of one function over the whole float domain. */
static std::vector<Mismatch> scan(int kind, float *device_out, float *host_out)
{
	std::vector<Mismatch> mismatches;
	for (uint32_t chunk = 0; chunk < CHUNKS; ++chunk) {
		uint32_t base = chunk << CHUNK_BITS;
		scan_kernel<<<CHUNK / 256, 256>>>(kind, base, device_out);
		check(cudaGetLastError(), "scan launch");
		check(cudaMemcpy(host_out, device_out, CHUNK * sizeof(float),
			cudaMemcpyDeviceToHost), "scan copy");
		std::vector<std::vector<Mismatch>> partial(omp_get_max_threads());
#pragma omp parallel
		{
			std::vector<Mismatch> &mine = partial[omp_get_thread_num()];
#pragma omp for schedule(static)
			for (int64_t i = 0; i < (int64_t)CHUNK; ++i) {
				uint32_t input = base + (uint32_t)i;
				float x;
				memcpy(&x, &input, sizeof(x));
				float reference = host_reference(kind, x);
				float device = host_out[i];
				if (bits(reference) == bits(device))
					continue;
				if (isnan(reference) && isnan(device))
					continue;
				mine.push_back({ input, bits(device), bits(reference) });
			}
		}
		for (auto &part : partial)
			mismatches.insert(mismatches.end(), part.begin(), part.end());
	}
	std::sort(mismatches.begin(), mismatches.end(),
		[](const Mismatch &a, const Mismatch &b) {
			return a.input < b.input;
		});
	return mismatches;
}

static void write_table1(
	FILE *out, const char *name, const TmnfLibmHardCase1 *existing,
	uint32_t existing_count, const std::vector<Mismatch> &fresh)
{
	std::map<uint32_t, uint32_t> merged;
	for (uint32_t i = 0; i < existing_count; ++i)
		merged[existing[i].input] = existing[i].output;
	for (const Mismatch &m : fresh)
		merged[m.input] = m.host;
	fprintf(out, "enum { TMNF_LIBM_%s_HARD_COUNT = %zu };\n", name,
		merged.size());
	fprintf(out,
		"__device__ static const TmnfLibmHardCase1 TMNF_LIBM_%s_HARD[%zu] = {\n",
		name, merged.empty() ? (size_t)1 : merged.size());
	if (merged.empty())
		fprintf(out, "\t{ 0, 0 },\n");
	for (auto &entry : merged)
		fprintf(out, "\t{ 0x%08xu, 0x%08xu },\n", entry.first, entry.second);
	fprintf(out, "};\n\n");
}

struct Key2 {
	uint32_t y;
	uint32_t x;
	bool operator<(const Key2 &other) const
	{
		return y < other.y || (y == other.y && x < other.x);
	}
};

static void write_table2(
	FILE *out, const char *name, const TmnfLibmHardCase2 *existing,
	uint32_t existing_count, const std::vector<Mismatch> &acos_form,
	const std::vector<Mismatch> &asin_form)
{
	std::map<Key2, uint32_t> merged;
	for (uint32_t i = 0; i < existing_count; ++i)
		merged[{ existing[i].y, existing[i].x }] = existing[i].output;
	for (const Mismatch &m : acos_form) {
		float v;
		memcpy(&v, &m.input, sizeof(v));
		merged[{ bits(acos_root(v)), m.input }] = m.host;
	}
	for (const Mismatch &m : asin_form) {
		float v;
		memcpy(&v, &m.input, sizeof(v));
		merged[{ m.input, bits(acos_root(v)) }] = m.host;
	}
	fprintf(out, "enum { TMNF_LIBM_%s_HARD_COUNT = %zu };\n", name,
		merged.size());
	fprintf(out,
		"__device__ static const TmnfLibmHardCase2 TMNF_LIBM_%s_HARD[%zu] = {\n",
		name, merged.empty() ? (size_t)1 : merged.size());
	if (merged.empty())
		fprintf(out, "\t{ 0, 0, 0 },\n");
	for (auto &entry : merged) {
		fprintf(out, "\t{ 0x%08xu, 0x%08xu, 0x%08xu },\n",
			entry.first.y, entry.first.x, entry.second);
	}
	fprintf(out, "};\n\n");
}

int main(int argc, char **argv)
{
	if (argc < 2 || (strcmp(argv[1], "generate") != 0 &&
		strcmp(argv[1], "verify") != 0) ||
		(strcmp(argv[1], "generate") == 0 && argc != 3)) {
		fprintf(stderr,
			"usage: cuda_libm_hardcases generate OUTPUT.h | verify\n");
		return 2;
	}
	int generate = strcmp(argv[1], "generate") == 0;

	float *device_out;
	float *host_out;
	check(cudaMalloc(&device_out, CHUNK * sizeof(float)), "cudaMalloc");
	check(cudaMallocHost(&host_out, CHUNK * sizeof(float)), "cudaMallocHost");

	std::vector<Mismatch> results[KIND_COUNT];
	size_t total = 0;
	for (int kind = 0; kind < KIND_COUNT; ++kind) {
		double start = omp_get_wtime();
		results[kind] = scan(kind, device_out, host_out);
		printf("%-18s residual mismatches: %zu (%.1f s)\n",
			KIND_NAMES[kind], results[kind].size(),
			omp_get_wtime() - start);
		for (size_t i = 0; i < results[kind].size() && i < 8; ++i) {
			printf("  input 0x%08x device 0x%08x glibc 0x%08x\n",
				results[kind][i].input, results[kind][i].device,
				results[kind][i].host);
		}
		total += results[kind].size();
	}

	if (!generate) {
		printf("cuda_libm: %zu residual mismatches\n", total);
		return total == 0 ? 0 : 1;
	}

	/* The compiled-in tables are device symbols; fetch them for the merge. */
	std::vector<TmnfLibmHardCase1> sin_existing(
		TMNF_LIBM_SIN_HARD_COUNT > 0 ? TMNF_LIBM_SIN_HARD_COUNT : 1);
	std::vector<TmnfLibmHardCase1> cos_existing(
		TMNF_LIBM_COS_HARD_COUNT > 0 ? TMNF_LIBM_COS_HARD_COUNT : 1);
	std::vector<TmnfLibmHardCase1> exp_existing(
		TMNF_LIBM_EXP_HARD_COUNT > 0 ? TMNF_LIBM_EXP_HARD_COUNT : 1);
	std::vector<TmnfLibmHardCase2> atan2_existing(
		TMNF_LIBM_ATAN2_HARD_COUNT > 0 ? TMNF_LIBM_ATAN2_HARD_COUNT : 1);
	check(cudaMemcpyFromSymbol(sin_existing.data(), TMNF_LIBM_SIN_HARD,
		sin_existing.size() * sizeof(TmnfLibmHardCase1)), "sin table");
	check(cudaMemcpyFromSymbol(cos_existing.data(), TMNF_LIBM_COS_HARD,
		cos_existing.size() * sizeof(TmnfLibmHardCase1)), "cos table");
	check(cudaMemcpyFromSymbol(exp_existing.data(), TMNF_LIBM_EXP_HARD,
		exp_existing.size() * sizeof(TmnfLibmHardCase1)), "exp table");
	check(cudaMemcpyFromSymbol(atan2_existing.data(), TMNF_LIBM_ATAN2_HARD,
		atan2_existing.size() * sizeof(TmnfLibmHardCase2)), "atan2 table");

	FILE *out = fopen(argv[2], "w");
	if (out == NULL) {
		fprintf(stderr, "cuda_libm_hardcases: cannot open %s\n", argv[2]);
		return 2;
	}
	int runtime_version = 0;
	check(cudaRuntimeGetVersion(&runtime_version), "runtime version");
	cudaDeviceProp prop;
	check(cudaGetDeviceProperties(&prop, 0), "device properties");
	fprintf(out,
		"/* Generated by tools/cuda_libm_hardcases.cu; do not edit.\n"
		" * glibc %s, CUDA runtime %d, %s (sm_%d%d).\n"
		" * Inputs where CUDA's double libm rounded to float differs from\n"
		" * glibc's; see src/cuda/tmnf_dev_libm.h. */\n"
		"#ifndef TMNF_LIBM_HARDCASES_H\n"
		"#define TMNF_LIBM_HARDCASES_H\n\n"
		"/* The table is only valid for the glibc that produced the host side\n"
		" * and the CUDA toolkit that produced the device side; "
		"TmnfCudaVecEnv_Create\n"
		" * checks both and the cuda_libm_hardcases test rescans. */\n"
		"#define TMNF_LIBM_GLIBC_VERSION \"%s\"\n"
		"#define TMNF_LIBM_CUDART_VERSION %d\n\n",
		gnu_get_libc_version(), runtime_version, prop.name, prop.major,
		prop.minor, gnu_get_libc_version(), runtime_version);
	write_table1(out, "SIN", sin_existing.data(), TMNF_LIBM_SIN_HARD_COUNT,
		results[KIND_SIN]);
	write_table1(out, "COS", cos_existing.data(), TMNF_LIBM_COS_HARD_COUNT,
		results[KIND_COS]);
	write_table1(out, "EXP", exp_existing.data(), TMNF_LIBM_EXP_HARD_COUNT,
		results[KIND_EXP]);
	write_table2(out, "ATAN2", atan2_existing.data(),
		TMNF_LIBM_ATAN2_HARD_COUNT, results[KIND_ATAN2_ACOS],
		results[KIND_ATAN2_ASIN]);
	fprintf(out, "#endif\n");
	if (fclose(out) != 0) {
		fprintf(stderr, "cuda_libm_hardcases: cannot write %s\n", argv[2]);
		return 2;
	}
	printf("wrote %s\n", argv[2]);
	return 0;
}
