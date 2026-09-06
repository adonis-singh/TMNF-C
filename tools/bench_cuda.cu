/*
 * GPU environment throughput: N environments, held random discrete actions
 * (the same generator as tests/cuda_lockstep.c so cars drive, slide and hit
 * walls), device-resident actions and results, no host copies inside the
 * timed region.
 *
 *   bench_cuda TRACK VEHICLE ROUTE SHA256 ENVS[,ENVS...] [TICKS] [REPEAT] [HOLD]
 *
 * HOLD (default REPEAT) is how many ticks each action vector is held, so
 * "600 1 3" and "600 3 3" drive identical per-tick actions and differ only
 * in launches per tick.
 *
 * Prints one line per environment count: ticks/s, wall µs per batch step,
 * ns per environment tick, and the termination mix over the timed region.
 */
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cuda/tmnf_cuda_env.h"
#include "route.h"
#include "track.h"
#include "vec_env.h"
#include "world.h"

enum {
	WARMUP_TICKS = 200,
	ACTION_ROTATION = 64,
};

static void fail(const char *message)
{
	fprintf(stderr, "bench_cuda: %s\n", message);
	exit(2);
}

static void check(cudaError_t error, const char *what)
{
	if (error != cudaSuccess) {
		fprintf(stderr, "bench_cuda: %s: %s\n", what,
			cudaGetErrorString(error));
		exit(2);
	}
}

static uint8_t hex_nibble(char c)
{
	if (c >= '0' && c <= '9')
		return (uint8_t)(c - '0');
	if (c >= 'a' && c <= 'f')
		return (uint8_t)(c - 'a' + 10);
	if (c >= 'A' && c <= 'F')
		return (uint8_t)(c - 'A' + 10);
	fail("SHA-256 is not hexadecimal");
	return 0;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint32_t rng_next(void)
{
	rng_state ^= rng_state >> 12;
	rng_state ^= rng_state << 25;
	rng_state ^= rng_state >> 27;
	return (uint32_t)((rng_state * 0x2545F4914F6CDD1Dull) >> 32);
}

static double now_seconds(void)
{
	struct timespec time;
	clock_gettime(CLOCK_MONOTONIC_RAW, &time);
	return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

/* longitudinal * 3 + steering: gas 65%, coast 15%, brake 10%, both 10%. */
static uint8_t random_action(void)
{
	uint32_t roll = rng_next() % 100;
	uint32_t longitudinal = roll < 65 ? 1 : roll < 80 ? 0 : roll < 90 ? 2 : 3;
	return (uint8_t)(longitudinal * 3 + rng_next() % 3);
}

/* ACTION_ROTATION action vectors per env count, held 1..60 ticks each. */
static uint8_t *build_actions(uint32_t count)
{
	uint8_t *actions = (uint8_t *)malloc((size_t)count * ACTION_ROTATION);
	uint32_t *hold = (uint32_t *)calloc(count, sizeof(uint32_t));
	uint8_t *current = (uint8_t *)calloc(count, 1);
	if (actions == NULL || hold == NULL || current == NULL)
		fail("out of memory");
	for (uint32_t step = 0; step < ACTION_ROTATION; ++step) {
		for (uint32_t i = 0; i < count; ++i) {
			if (hold[i] == 0) {
				current[i] = random_action();
				hold[i] = 1 + rng_next() % 60;
			}
			hold[i]--;
			actions[(size_t)step * count + i] = current[i];
		}
	}
	free(hold);
	free(current);
	return actions;
}

/* Action vector of step s: vectors change every `hold` ticks, so that
 * repeat 1 with hold 3 drives the same per-tick actions as repeat 3. */
static const uint8_t *step_actions(
	const uint8_t *device_actions, uint32_t count, uint32_t s,
	uint32_t repeat, uint32_t hold)
{
	uint32_t vector = ((s * repeat) / hold) % ACTION_ROTATION;
	return device_actions + (size_t)vector * count;
}

static void bench(
	TmnfTrack *track, TmnfWorld *template_world, TmnfRoute *route,
	uint32_t count, uint32_t ticks, uint32_t repeat, uint32_t hold)
{
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	TmnfCudaVecEnvLimits limits = TmnfCudaVecEnv_DefaultLimits();
	size_t free_before = 0, free_after = 0, total = 0;
	check(cudaMemGetInfo(&free_before, &total), "memory info");
	TmnfCudaVecEnv *env = TmnfCudaVecEnv_Create(
		track, template_world, route, count, &config, &limits);
	if (env == NULL)
		fail(TmnfCudaVecEnv_LastError());
	check(cudaMemGetInfo(&free_after, &total), "memory info");
	printf("  create: %.0f MiB of device memory (stack %u B per thread)\n",
		(double)(free_before - free_after) / 1048576.0, limits.stack_bytes);

	uint8_t *host_actions = build_actions(count);
	uint8_t *device_actions = NULL;
	check(cudaMalloc((void **)&device_actions, (size_t)count * ACTION_ROTATION),
		"action buffer");
	check(cudaMemcpy(device_actions, host_actions,
			(size_t)count * ACTION_ROTATION, cudaMemcpyHostToDevice),
		"action upload");
	free(host_actions);

	uint32_t steps = (ticks + repeat - 1) / repeat;
	uint32_t warmup_steps = (WARMUP_TICKS + repeat - 1) / repeat;
	for (uint32_t s = 0; s < warmup_steps; ++s) {
		TmnfCudaVecEnv_StepDiscreteDevice(env,
			step_actions(device_actions, count, s, repeat, hold), repeat);
	}
	check(cudaDeviceSynchronize(), "warmup");

	cudaEvent_t begin, end;
	check(cudaEventCreate(&begin), "event");
	check(cudaEventCreate(&end), "event");
	double wall_begin = now_seconds();
	check(cudaEventRecord(begin), "event record");
	for (uint32_t s = 0; s < steps; ++s) {
		TmnfCudaVecEnv_StepDiscreteDevice(env,
			step_actions(device_actions, count, s, repeat, hold), repeat);
	}
	check(cudaEventRecord(end), "event record");
	check(cudaDeviceSynchronize(), "timed region");
	double wall = now_seconds() - wall_begin;
	float gpu_ms = 0.0f;
	check(cudaEventElapsedTime(&gpu_ms, begin, end), "event elapsed");

	/* Termination mix: sample the last step's results. */
	TmnfStepResult *results =
		(TmnfStepResult *)malloc((size_t)count * sizeof(TmnfStepResult));
	if (results == NULL)
		fail("out of memory");
	check(cudaMemcpy(results, TmnfCudaVecEnv_DeviceResults(env),
			(size_t)count * sizeof(TmnfStepResult), cudaMemcpyDeviceToHost),
		"results download");
	double speed_sum = 0.0;
	uint32_t moving = 0;
	for (uint32_t i = 0; i < count; ++i) {
		const GmVec3 *v = &results[i].observation.linear_speed;
		double speed = sqrt((double)v->x * v->x + (double)v->y * v->y +
			(double)v->z * v->z);
		speed_sum += speed;
		moving += speed > 1.0;
	}
	free(results);

	double env_ticks = (double)count * (double)steps * (double)repeat;
	printf("%7u envs repeat %u: %10.3f M ticks/s  %9.1f us/step  "
		"%7.1f ns/env-tick  (gpu %.3f s, wall %.3f s, mean speed %.1f m/s, "
		"%.0f%% moving)\n",
		count, repeat, env_ticks / wall / 1e6,
		wall / steps * 1e6, wall / env_ticks * 1e9,
		gpu_ms / 1000.0, wall, speed_sum / count,
		100.0 * moving / count);
	fflush(stdout);

	check(cudaFree(device_actions), "action free");
	TmnfCudaVecEnv_Destroy(env);
}

int main(int argc, char **argv)
{
	if (argc < 6) {
		fprintf(stderr,
			"usage: bench_cuda TRACK VEHICLE ROUTE SHA256 ENVS[,ENVS...] "
			"[TICKS] [REPEAT] [HOLD]\n");
		return 2;
	}
	if (strlen(argv[4]) != 64)
		fail("SHA-256 must have 64 hexadecimal characters");
	uint8_t sha256[32];
	for (uint32_t i = 0; i < 32; ++i) {
		sha256[i] = (uint8_t)(hex_nibble(argv[4][i * 2]) << 4 |
			hex_nibble(argv[4][i * 2 + 1]));
	}
	uint32_t ticks = argc > 6 ? (uint32_t)strtoul(argv[6], NULL, 10) : 2000;
	uint32_t repeat = argc > 7 ? (uint32_t)strtoul(argv[7], NULL, 10) : 1;
	uint32_t hold = argc > 8 ? (uint32_t)strtoul(argv[8], NULL, 10) : repeat;
	if (ticks == 0 || repeat == 0 || hold < repeat || hold % repeat != 0)
		fail("ticks and repeat must be positive; hold a multiple of repeat");

	TmnfTrack *track = TmnfTrack_Load(argv[1], sha256);
	TmnfRoute *route = TmnfRoute_Load(argv[3], sha256);
	TmnfWorld *template_world = World_Create(track, argv[2]);

	char *list = argv[5];
	for (char *token = strtok(list, ","); token != NULL;
		token = strtok(NULL, ",")) {
		uint32_t count = (uint32_t)strtoul(token, NULL, 10);
		if (count == 0)
			fail("environment count must be positive");
		bench(track, template_world, route, count, ticks, repeat, hold);
	}

	World_Destroy(template_world);
	TmnfRoute_Unload(route);
	TmnfTrack_Unload(track);
	return 0;
}
