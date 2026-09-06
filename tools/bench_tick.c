/* Repeated full-environment throughput and cost-split benchmark.
 *
 * Build after the CMake Release build:
 *   cc -std=c11 -O3 -DNDEBUG -Wall -Wextra -Werror \
 *     -ffp-contract=off -fno-fast-math -Isrc tools/bench_tick.c \
 *     build/libtmnf_physics.a -pthread -lm \
 *     -Wl,--wrap=CHmsZoneDynamic_PhysicsStep2 \
 *     -Wl,--wrap=CHmsCollisionManager_SZone_PrepareCollisions \
 *     -Wl,--wrap=CHmsCollisionManager_SZone_DetectCollisionsCorpus \
 *     -Wl,--wrap=CHmsZoneDynamic_ComputeCollisionResponse \
 *     -Wl,--wrap=TmnfRace_Step -o build/bench_tick
 */
#define _GNU_SOURCE

#include <inttypes.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "race.h"
#include "vec_env.h"
#include "world.h"

enum {
	DRIVE_TICKS = 3000,
	DRIVE_RECORD_SIZE = 1668,
	DRIVE_LINEAR_SPEED_OFFSET = 68,
	DRIVE_INPUT_OFFSET = 176,
	WARMUP_TICKS = 300,
	PROFILE_REPETITIONS = 5,
	SCALING_COUNT = 6,
	EMPTY_DISPATCHES = 10000,
};

static const uint32_t SCALING_THREADS[SCALING_COUNT] = {
	1, 2, 4, 8, 16, 32,
};

static const uint8_t A01_SHA256[32] = {
	0xf0, 0xa8, 0x70, 0x80, 0x9b, 0xe9, 0x9d, 0xa2,
	0xcb, 0x36, 0xad, 0x5d, 0xf4, 0x3a, 0x2c, 0xf6,
	0x3d, 0x8f, 0x74, 0xfe, 0x4a, 0xc3, 0x47, 0x0e,
	0xca, 0xc6, 0x8b, 0x9e, 0x97, 0x62, 0x5d, 0xc3,
};

typedef struct {
	double mean;
	double standard_deviation;
	double minimum;
	double maximum;
} SampleStats;

typedef struct {
	uint8_t actions[DRIVE_TICKS];
	double mean_native_speed;
	double native_active_fraction;
	double final_native_speed;
	double cycle_mean_speed;
	double cycle_active_fraction;
	uint32_t cycle_ticks;
	uint32_t gas_ticks;
	uint32_t brake_ticks;
	uint32_t coast_ticks;
} DriveStream;

typedef struct {
	double wall_ns;
	double physics_ns;
	double collision_ns;
	double race_ns;
} ProfileSample;

static int profile_enabled;
static uint64_t profile_physics_ns;
static uint64_t profile_collision_ns;
static uint64_t profile_race_ns;
static uint64_t profile_physics_calls;
static uint64_t profile_collision_calls;
static uint64_t profile_race_calls;

static void fail(const char *message)
{
	fprintf(stderr, "bench_tick: %s\n", message);
	exit(2);
}

static void *allocate(size_t count, size_t size)
{
	if (count == 0 || size == 0 || count > SIZE_MAX / size)
		fail("invalid allocation");
	void *memory = calloc(count, size);
	if (memory == NULL)
		fail("out of memory");
	return memory;
}

static uint64_t monotonic_ns(void)
{
	struct timespec time;
	if (clock_gettime(CLOCK_MONOTONIC_RAW, &time) != 0)
		fail("clock_gettime failed");
	return (uint64_t)time.tv_sec * UINT64_C(1000000000) +
		(uint64_t)time.tv_nsec;
}

void __real_CHmsZoneDynamic_PhysicsStep2(
	TmnfPhysicsWorld *world, uint32_t tick_ms);

void __wrap_CHmsZoneDynamic_PhysicsStep2(
	TmnfPhysicsWorld *world, uint32_t tick_ms)
{
	if (!profile_enabled) {
		__real_CHmsZoneDynamic_PhysicsStep2(world, tick_ms);
		return;
	}
	uint64_t start = monotonic_ns();
	__real_CHmsZoneDynamic_PhysicsStep2(world, tick_ms);
	profile_physics_ns += monotonic_ns() - start;
	profile_physics_calls++;
}

void __real_CHmsCollisionManager_SZone_PrepareCollisions(
	CHmsCollisionManager_SZone *zone);

void __wrap_CHmsCollisionManager_SZone_PrepareCollisions(
	CHmsCollisionManager_SZone *zone)
{
	if (!profile_enabled) {
		__real_CHmsCollisionManager_SZone_PrepareCollisions(zone);
		return;
	}
	uint64_t start = monotonic_ns();
	__real_CHmsCollisionManager_SZone_PrepareCollisions(zone);
	profile_collision_ns += monotonic_ns() - start;
	profile_collision_calls++;
}

void __real_CHmsCollisionManager_SZone_DetectCollisionsCorpus(
	CHmsCollisionManager_SZone *zone, CHmsCollisionBuffer *buffer,
	CHmsCorpus *corpus, int active);

void __wrap_CHmsCollisionManager_SZone_DetectCollisionsCorpus(
	CHmsCollisionManager_SZone *zone, CHmsCollisionBuffer *buffer,
	CHmsCorpus *corpus, int active)
{
	if (!profile_enabled) {
		__real_CHmsCollisionManager_SZone_DetectCollisionsCorpus(
			zone, buffer, corpus, active);
		return;
	}
	uint64_t start = monotonic_ns();
	__real_CHmsCollisionManager_SZone_DetectCollisionsCorpus(
		zone, buffer, corpus, active);
	profile_collision_ns += monotonic_ns() - start;
	profile_collision_calls++;
}

void __real_CHmsZoneDynamic_ComputeCollisionResponse(
	CHmsResponseZone *zone);

void __wrap_CHmsZoneDynamic_ComputeCollisionResponse(
	CHmsResponseZone *zone)
{
	if (!profile_enabled) {
		__real_CHmsZoneDynamic_ComputeCollisionResponse(zone);
		return;
	}
	uint64_t start = monotonic_ns();
	__real_CHmsZoneDynamic_ComputeCollisionResponse(zone);
	profile_collision_ns += monotonic_ns() - start;
	profile_collision_calls++;
}

TmnfRaceStepResult __real_TmnfRace_Step(
	const TmnfRoute *route, TmnfRaceState *state,
	uint64_t trigger_contacts,
	const GmIso4 *car_world_transform);

/* The trigger test itself now runs inside PhysicsStep2's detection passes
 * and is counted under the physics step, not here. */
TmnfRaceStepResult __wrap_TmnfRace_Step(
	const TmnfRoute *route, TmnfRaceState *state,
	uint64_t trigger_contacts,
	const GmIso4 *car_world_transform)
{
	if (!profile_enabled) {
		return __real_TmnfRace_Step(
			route, state, trigger_contacts, car_world_transform);
	}
	uint64_t start = monotonic_ns();
	TmnfRaceStepResult result = __real_TmnfRace_Step(
		route, state, trigger_contacts, car_world_transform);
	profile_race_ns += monotonic_ns() - start;
	profile_race_calls++;
	return result;
}

static float read_float(const uint8_t *bytes)
{
	float value;
	memcpy(&value, bytes, sizeof(value));
	return value;
}

static double vector_speed(const uint8_t *bytes)
{
	double x = read_float(bytes);
	double y = read_float(bytes + 4);
	double z = read_float(bytes + 8);
	return sqrt(x * x + y * y + z * z);
}

static DriveStream load_drive_stream(const char *path)
{
	FILE *file = fopen(path, "rb");
	if (file == NULL)
		fail("cannot open driving stream");
	uint8_t *records = allocate(
		DRIVE_TICKS, DRIVE_RECORD_SIZE);
	size_t count = fread(
		records, DRIVE_RECORD_SIZE, DRIVE_TICKS, file);
	if (count != DRIVE_TICKS || fgetc(file) != EOF)
		fail("driving stream must contain exactly 3000 records");
	if (fclose(file) != 0)
		fail("cannot close driving stream");

	DriveStream stream = {0};
	double speeds[DRIVE_TICKS];
	uint32_t active_ticks = 0;
	double speed_sum = 0.0;
	for (uint32_t tick = 0; tick < DRIVE_TICKS; ++tick) {
		const uint8_t *record =
			records + (size_t)tick * DRIVE_RECORD_SIZE;
		const uint8_t *input = record + DRIVE_INPUT_OFFSET;
		float gas = read_float(input);
		float brake = read_float(input + 4);
		float steer = read_float(input + 8);
		if ((gas != 0.0f && gas != 1.0f) ||
			(brake != 0.0f && brake != 1.0f) ||
			(steer != -1.0f && steer != 0.0f &&
			 steer != 1.0f)) {
			fail("driving stream has non-discrete controls");
		}

		uint32_t longitudinal;
		if (gas != 0.0f && brake != 0.0f)
			longitudinal = 3;
		else if (gas != 0.0f)
			longitudinal = 1;
		else if (brake != 0.0f)
			longitudinal = 2;
		else
			longitudinal = 0;
		uint32_t steering = steer < 0.0f
			? 0 : (steer > 0.0f ? 2 : 1);
		stream.actions[tick] =
			(uint8_t)(longitudinal * 3 + steering);
		if (longitudinal == 1)
			stream.gas_ticks++;
		else if (longitudinal == 2)
			stream.brake_ticks++;
		else
			stream.coast_ticks++;

		double speed = vector_speed(
			record + DRIVE_LINEAR_SPEED_OFFSET);
		speeds[tick] = speed;
		speed_sum += speed;
		if (speed > 1.0)
			active_ticks++;
		stream.final_native_speed = speed;
	}
	free(records);
	stream.mean_native_speed = speed_sum / DRIVE_TICKS;
	stream.native_active_fraction =
		(double)active_ticks / DRIVE_TICKS;
	for (uint32_t tick = 0; tick < DRIVE_TICKS; ++tick) {
		if (speeds[tick] > 5.0)
			stream.cycle_ticks = tick + 1;
	}
	if (stream.cycle_ticks < 100 ||
		stream.cycle_ticks == DRIVE_TICKS) {
		fail("cannot isolate the moving prefix");
	}
	uint32_t cycle_active_ticks = 0;
	for (uint32_t tick = 0;
		tick < stream.cycle_ticks; ++tick) {
		stream.cycle_mean_speed += speeds[tick];
		if (speeds[tick] > 1.0)
			cycle_active_ticks++;
	}
	stream.cycle_mean_speed /= stream.cycle_ticks;
	stream.cycle_active_fraction =
		(double)cycle_active_ticks / stream.cycle_ticks;
	if (stream.cycle_active_fraction < 0.80)
		fail("moving prefix is mostly stationary");
	stream.gas_ticks = 0;
	stream.brake_ticks = 0;
	stream.coast_ticks = 0;
	for (uint32_t tick = 0;
		tick < stream.cycle_ticks; ++tick) {
		uint32_t longitudinal = stream.actions[tick] / 3;
		if (longitudinal == 1)
			stream.gas_ticks++;
		else if (longitudinal == 2)
			stream.brake_ticks++;
		else
			stream.coast_ticks++;
	}
	return stream;
}

static SampleStats summarize(const double *samples, uint32_t count)
{
	SampleStats stats = {
		.minimum = samples[0],
		.maximum = samples[0],
	};
	for (uint32_t i = 0; i < count; ++i) {
		stats.mean += samples[i];
		if (samples[i] < stats.minimum)
			stats.minimum = samples[i];
		if (samples[i] > stats.maximum)
			stats.maximum = samples[i];
	}
	stats.mean /= count;
	double sum_squared = 0.0;
	for (uint32_t i = 0; i < count; ++i) {
		double difference = samples[i] - stats.mean;
		sum_squared += difference * difference;
	}
	stats.standard_deviation =
		sqrt(sum_squared / (count - 1));
	return stats;
}

/* The collection horizon must allow a finish (TmnfVecEnv_Init aborts
 * otherwise), so the moving prefix is repeated by restoring the initial
 * snapshot at every cycle boundary instead of by horizon autoreset. */
static uint32_t workload_cycle_ticks;

static void restore_at_cycle_boundary(
	TmnfVecEnv *env, const TmnfEnvSnapshot *snapshot, uint32_t tick)
{
	if (snapshot != NULL && tick != 0 &&
		tick % workload_cycle_ticks == 0)
		TmnfVecEnv_Restore(env, snapshot);
}

static double run_steps(
	TmnfVecEnv *env, const TmnfEnvSnapshot *snapshot,
	const uint8_t *actions, TmnfStepResult *results)
{
	uint64_t start = monotonic_ns();
	for (uint32_t tick = 0; tick < DRIVE_TICKS; ++tick) {
		restore_at_cycle_boundary(env, snapshot, tick);
		TmnfVecEnv_StepDiscrete(
			env, actions + (size_t)tick * env->count,
			1, results);
	}
	return (double)(monotonic_ns() - start);
}

static void warm_up(
	TmnfVecEnv *env, const TmnfEnvSnapshot *initial,
	const uint8_t *drive_actions, TmnfStepResult *results)
{
	TmnfVecEnv_Restore(env, initial);
	for (uint32_t tick = 0; tick < WARMUP_TICKS; ++tick) {
		restore_at_cycle_boundary(env, initial, tick);
		TmnfVecEnv_StepDiscrete(
			env,
			drive_actions + (size_t)tick * env->count,
			1, results);
	}
}

static SampleStats benchmark_workload(
	TmnfVecEnv *env, const TmnfEnvSnapshot *snapshot,
	const uint8_t *actions, TmnfStepResult *results,
	uint32_t repetitions)
{
	double *samples = allocate(repetitions, sizeof(*samples));
	double tick_count = (double)env->count * DRIVE_TICKS;
	for (uint32_t repetition = 0;
		repetition < repetitions; ++repetition) {
		TmnfVecEnv_Restore(env, snapshot);
		samples[repetition] =
			run_steps(env, snapshot, actions, results) /
			tick_count / 1000.0;
	}
	SampleStats stats = summarize(samples, repetitions);
	free(samples);
	return stats;
}

static SampleStats benchmark_empty_dispatch(
	TmnfVecEnv *env, uint32_t repetitions)
{
	for (uint32_t i = 0; i < 1000; ++i)
		TmnfVecEnv_DispatchEmpty(env);
	double *samples = allocate(repetitions, sizeof(*samples));
	for (uint32_t repetition = 0;
		repetition < repetitions; ++repetition) {
		uint64_t start = monotonic_ns();
		for (uint32_t i = 0; i < EMPTY_DISPATCHES; ++i)
			TmnfVecEnv_DispatchEmpty(env);
		samples[repetition] =
			(double)(monotonic_ns() - start) /
			EMPTY_DISPATCHES / 1000.0;
	}
	SampleStats stats = summarize(samples, repetitions);
	free(samples);
	return stats;
}

static uint32_t requested_thread_count(void)
{
	const char *text = getenv("TMNF_BENCH_THREADS");
	if (text == NULL)
		return 0;
	char *end = NULL;
	unsigned long value = strtoul(text, &end, 10);
	if (end == text || *end != '\0' || value > UINT32_MAX)
		fail("TMNF_BENCH_THREADS is invalid");
	for (uint32_t i = 0; i < SCALING_COUNT; ++i) {
		if (value == SCALING_THREADS[i])
			return (uint32_t)value;
	}
	fail("TMNF_BENCH_THREADS must be 1, 2, 4, 8, 16, or 32");
	return 0;
}

static ProfileSample profile_workload(
	TmnfVecEnv *env, const TmnfEnvSnapshot *snapshot,
	const uint8_t *actions, TmnfStepResult *results)
{
	TmnfVecEnv_Restore(env, snapshot);
	profile_physics_ns = 0;
	profile_collision_ns = 0;
	profile_race_ns = 0;
	profile_physics_calls = 0;
	profile_collision_calls = 0;
	profile_race_calls = 0;
	profile_enabled = 1;
	double wall_ns = run_steps(env, snapshot, actions, results);
	profile_enabled = 0;
	uint64_t expected_calls =
		(uint64_t)env->count * DRIVE_TICKS;
	if (profile_physics_calls != expected_calls ||
		profile_race_calls != expected_calls ||
		profile_collision_calls == 0) {
		fail("linker profiling wrappers did not observe all calls");
	}
	return (ProfileSample){
		.wall_ns = wall_ns,
		.physics_ns = (double)profile_physics_ns,
		.collision_ns = (double)profile_collision_ns,
		.race_ns = (double)profile_race_ns,
	};
}

static void report_profile(
	const char *name, TmnfVecEnv *env,
	const TmnfEnvSnapshot *snapshot, const uint8_t *actions,
	TmnfStepResult *results, double unprofiled_us)
{
	ProfileSample samples[PROFILE_REPETITIONS];
	double wall_us[PROFILE_REPETITIONS];
	double tick_count = (double)env->count * DRIVE_TICKS;
	for (uint32_t repetition = 0;
		repetition < PROFILE_REPETITIONS; ++repetition) {
		samples[repetition] = profile_workload(
			env, snapshot, actions, results);
		wall_us[repetition] =
			samples[repetition].wall_ns / tick_count / 1000.0;
	}
	SampleStats wall = summarize(
		wall_us, PROFILE_REPETITIONS);
	double physics = 0.0;
	double collision = 0.0;
	double race = 0.0;
	double total = 0.0;
	for (uint32_t i = 0; i < PROFILE_REPETITIONS; ++i) {
		physics += samples[i].physics_ns -
			samples[i].collision_ns;
		collision += samples[i].collision_ns;
		race += samples[i].race_ns;
		total += samples[i].wall_ns;
	}
	double physics_fraction = physics / total;
	double collision_fraction = collision / total;
	double race_observation_fraction =
		1.0 - physics_fraction - collision_fraction;
	printf(
		"cost %-10s physics_ex_collision %.3f us %5.1f%%  "
		"collision %.3f us %5.1f%%  "
		"race_observation %.3f us %5.1f%%\n",
		name,
		unprofiled_us * physics_fraction,
		physics_fraction * 100.0,
		unprofiled_us * collision_fraction,
		collision_fraction * 100.0,
		unprofiled_us * race_observation_fraction,
		race_observation_fraction * 100.0);
	printf(
		"cost %-10s race_step_subset %.3f us %5.1f%%  "
		"profiled %.3f us sd %.3f cv %.2f%% overhead %.1f%%\n",
		name,
		unprofiled_us * race / total,
		race / total * 100.0,
		wall.mean, wall.standard_deviation,
		wall.standard_deviation / wall.mean * 100.0,
		(wall.mean / unprofiled_us - 1.0) * 100.0);
}

static double observation_speed(const TmnfObservation *observation)
{
	double x = observation->linear_speed.x;
	double y = observation->linear_speed.y;
	double z = observation->linear_speed.z;
	return sqrt(x * x + y * y + z * z);
}

static void verify_port_drive(
	TmnfVecEnv *env, const TmnfEnvSnapshot *initial,
	const uint8_t *drive_actions, TmnfStepResult *results)
{
	TmnfVecEnv_Restore(env, initial);
	double speed_sum = 0.0;
	uint32_t active_ticks = 0;
	uint32_t endings = 0;
	for (uint32_t tick = 0; tick < DRIVE_TICKS; ++tick) {
		restore_at_cycle_boundary(env, initial, tick);
		TmnfVecEnv_StepDiscrete(
			env,
			drive_actions + (size_t)tick * env->count,
			1, results);
		double speed = observation_speed(
			&results[0].observation);
		speed_sum += speed;
		if (speed > 1.0)
			active_ticks++;
		endings += results[0].terminated ||
			results[0].truncated;
	}
	double active_fraction = (double)active_ticks / DRIVE_TICKS;
	printf(
		"ported drive       mean_speed %.3f m/s active_gt_1mps %.2f%% "
		"final_speed %.3f m/s endings %u\n",
		speed_sum / DRIVE_TICKS, active_fraction * 100.0,
		observation_speed(&results[0].observation), endings);
	if (active_fraction < 0.80 || endings != 0) {
		fail("ported driving workload was not sustained");
	}
}

static uint32_t allowed_cpu_count(void)
{
	cpu_set_t allowed;
	if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
		fail("sched_getaffinity failed");
	return (uint32_t)CPU_COUNT(&allowed);
}

static uint64_t resident_kb(void)
{
	FILE *status = fopen("/proc/self/status", "r");
	if (status == NULL)
		fail("cannot open /proc/self/status");
	char *line = NULL;
	size_t capacity = 0;
	uint64_t result = 0;
	while (getline(&line, &capacity, status) >= 0) {
		if (sscanf(line, "VmRSS: %" SCNu64 " kB", &result) == 1)
			break;
	}
	free(line);
	if (fclose(status) != 0)
		fail("cannot close /proc/self/status");
	if (result == 0)
		fail("cannot read VmRSS");
	return result;
}

static int report_world_rss(
	const char *track_path, const char *vehicle_path,
	uint32_t world_count)
{
	if (world_count != 256)
		fail("RSS mode requires exactly 256 worlds");
	uint64_t baseline = resident_kb();
	printf("rss baseline_kb %" PRIu64 "\n", baseline);
	TmnfWorld **worlds = allocate(world_count, sizeof(*worlds));
	TmnfTrack *track = TmnfTrack_Load(track_path, A01_SHA256);
	for (uint32_t i = 0; i < world_count; ++i) {
		worlds[i] = World_Create(track, vehicle_path);
		uint32_t count = i + 1;
		if (count == 1 || count == 64 || count == 256) {
			uint64_t rss = resident_kb();
			printf(
				"rss worlds %3u total_kb %" PRIu64
				" delta_kb %" PRIu64 "\n",
				count, rss, rss - baseline);
		}
	}
	for (uint32_t i = 0; i < world_count; ++i)
		World_Destroy(worlds[i]);
	TmnfTrack_Unload(track);
	free(worlds);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc == 5 && strcmp(argv[1], "--rss") == 0) {
		uint32_t world_count =
			(uint32_t)strtoul(argv[4], NULL, 10);
		return report_world_rss(argv[2], argv[3], world_count);
	}
	if (argc != 7) {
		fprintf(stderr,
			"usage: %s TRACK VEHICLE ROUTE DRIVE ENVS REPETITIONS\n"
			"       %s --rss TRACK VEHICLE 256\n",
			argv[0],
			argv[0]);
		return 2;
	}
	uint32_t env_count = (uint32_t)strtoul(argv[5], NULL, 10);
	uint32_t repetitions = (uint32_t)strtoul(argv[6], NULL, 10);
	if (env_count < 32 || env_count % 32 != 0)
		fail("ENVS must be a positive multiple of 32");
	if (repetitions < 3)
		fail("REPETITIONS must be at least 3");

	DriveStream drive = load_drive_stream(argv[4]);
	TmnfWorld **owners = allocate(env_count, sizeof(*owners));
	TmnfPhysicsWorld **worlds =
		allocate(env_count, sizeof(*worlds));
	uint32_t *player_indices =
		allocate(env_count, sizeof(*player_indices));
	TmnfTrack *track = TmnfTrack_Load(argv[1], A01_SHA256);
	for (uint32_t i = 0; i < env_count; ++i) {
		owners[i] = World_Create(track, argv[2]);
		worlds[i] = World_GetPhysicsWorld(owners[i]);
	}
	TmnfRoute *route = TmnfRoute_Load(argv[3], A01_SHA256);
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	uint32_t requested_threads = requested_thread_count();
	config.thread_count =
		requested_threads == 0 ? 1 : requested_threads;
	config.max_race_ticks = 100000;
	config.horizon_ticks = 0;
	workload_cycle_ticks = drive.cycle_ticks;
	config.off_track_grace_ticks = 100000;
	config.stuck_grace_ticks = 100000;
	TmnfVecEnv env;
	TmnfVecEnv_Init(
		&env, worlds, player_indices, env_count, route, &config);

	TmnfEnvSnapshot *initial =
		allocate(env_count, sizeof(*initial));
	TmnfEnvSnapshot *stationary =
		allocate(env_count, sizeof(*stationary));
	TmnfStepResult *results =
		allocate(env_count, sizeof(*results));
	uint8_t *drive_actions =
		allocate((size_t)DRIVE_TICKS, env_count);
	uint8_t *stationary_actions =
		allocate((size_t)DRIVE_TICKS, env_count);
	for (uint32_t tick = 0; tick < DRIVE_TICKS; ++tick) {
		memset(
			drive_actions + (size_t)tick * env_count,
			drive.actions[tick % drive.cycle_ticks],
			env_count);
	}
	TmnfVecEnv_Capture(&env, initial);

	TmnfVecEnv_Restore(&env, initial);
	for (uint32_t tick = 0;
		tick < drive.cycle_ticks; ++tick) {
		TmnfVecEnv_StepDiscrete(
			&env, stationary_actions, 1, results);
	}
	if (observation_speed(&results[0].observation) > 0.5)
		fail("stationary workload did not settle");
	TmnfVecEnv_Capture(&env, stationary);
	verify_port_drive(&env, initial, drive_actions, results);

	printf("method             full TmnfVecEnv_StepDiscrete, repeat=1\n");
	printf("workload           moving native prefix, repeated by snapshot restore\n");
	printf("clock              CLOCK_MONOTONIC_RAW\n");
	printf("environments       %u\n", env_count);
	printf("ticks_per_env      %u\n", DRIVE_TICKS);
	printf("repetitions        %u scaling, %u cost profile\n",
		repetitions, PROFILE_REPETITIONS);
	printf("allowed_logical_cpus %u\n", allowed_cpu_count());
	const char *pinned_cpus = getenv("TMNF_PIN_CPUS");
	printf("pinned_cpus        %s\n",
		pinned_cpus == NULL ? "none" : pinned_cpus);
	printf(
		"native source      mean_speed %.3f m/s active_gt_1mps %.2f%% "
		"final_speed %.3f m/s\n",
		drive.mean_native_speed,
		drive.native_active_fraction * 100.0,
		drive.final_native_speed);
	printf(
		"driving cycle      ticks %u gas %u brake %u coast %u "
		"mean_speed %.3f m/s active_gt_1mps %.2f%% "
		"source_cutoff 5.000 m/s\n",
		drive.cycle_ticks, drive.gas_ticks,
		drive.brake_ticks, drive.coast_ticks,
		drive.cycle_mean_speed,
		drive.cycle_active_fraction * 100.0);

	SampleStats scaling[SCALING_COUNT] = {{0}};
	printf("threads  us_per_tick  sd_us  cv_pct  min_us  max_us  "
		"steps_per_second\n");
	for (uint32_t index = 0; index < SCALING_COUNT; ++index) {
		uint32_t threads = SCALING_THREADS[index];
		if (requested_threads != 0 && threads != requested_threads)
			continue;
		TmnfVecEnv_SetThreadCount(&env, threads);
		warm_up(&env, initial, drive_actions, results);
		scaling[index] = benchmark_workload(
			&env, initial, drive_actions, results, repetitions);
		double rate = 1.0e6 / scaling[index].mean;
		printf(
			"%7u  %11.3f  %5.3f  %6.2f  %6.3f  %6.3f  "
			"%16.0f\n",
			threads, scaling[index].mean,
			scaling[index].standard_deviation,
			scaling[index].standard_deviation /
				scaling[index].mean * 100.0,
			scaling[index].minimum, scaling[index].maximum,
			rate);
		SampleStats empty = benchmark_empty_dispatch(
			&env, repetitions);
		printf(
			"empty_dispatch threads %u mean_us %.3f sd_us %.3f "
			"cv_pct %.2f min_us %.3f max_us %.3f\n",
			threads, empty.mean, empty.standard_deviation,
			empty.standard_deviation / empty.mean * 100.0,
			empty.minimum, empty.maximum);
	}

	if (requested_threads != 0 && requested_threads != 1)
		goto cleanup;
	TmnfVecEnv_SetThreadCount(&env, 1);
	warm_up(&env, initial, drive_actions, results);
	SampleStats stationary_stats = benchmark_workload(
		&env, stationary, stationary_actions, results, repetitions);
	printf(
		"stationary         %.3f us/tick sd %.3f cv %.2f%% "
		"%.0f steps/s\n",
		stationary_stats.mean,
		stationary_stats.standard_deviation,
		stationary_stats.standard_deviation /
			stationary_stats.mean * 100.0,
		1.0e6 / stationary_stats.mean);
	printf(
		"driving_vs_stationary %.3fx slower, +%.1f%%\n",
		scaling[0].mean / stationary_stats.mean,
		(scaling[0].mean / stationary_stats.mean - 1.0) * 100.0);

	report_profile(
		"driving", &env, initial, drive_actions, results,
		scaling[0].mean);
	report_profile(
		"stationary", &env, stationary, stationary_actions, results,
		stationary_stats.mean);

cleanup:
	free(stationary_actions);
	free(drive_actions);
	free(results);
	free(stationary);
	free(initial);
	TmnfVecEnv_Destroy(&env);
	TmnfRoute_Unload(route);
	for (uint32_t i = 0; i < env_count; ++i)
		World_Destroy(owners[i]);
	TmnfTrack_Unload(track);
	free(player_indices);
	free(worlds);
	free(owners);
	return 0;
}
