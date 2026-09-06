/*
 * CPU-vs-GPU lockstep. TmnfVecEnv and TmnfCudaVecEnv are driven with the same
 * actions and must agree byte for byte every tick: every TmnfStepResult
 * (observations, reward, flags) and every TmnfEnvSnapshot v6 field.
 *
 *   cuda_lockstep random TRACK VEHICLE ROUTE SHA256 ENVS TICKS
 *     ENVS environments, one random discrete action per env per tick, with a
 *     snapshot round trip in both directions halfway (CPU snapshots restored
 *     into the GPU and GPU snapshots into the CPU, rotated across envs), then
 *     an analog phase over TICKS / 5 ticks. Tight budgets so episodes end
 *     through every termination reason and reset often.
 *
 *   cuda_lockstep replay TRACK VEHICLE ROUTE SHA256 INPUTS
 *     Replays a committed digital input schedule through the discrete action
 *     space of both envs under the default training configuration, action
 *     repeat 1 and 5, comparing rewards, terminations and race times per step.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda/tmnf_cuda_env.h"
#include "route.h"
#include "vec_env.h"
#include "world.h"

enum {
	MAX_ENVS = 4096,
	CPU_THREADS = 16,
};

static void fail(const char *message)
{
	fprintf(stderr, "cuda lockstep: %s\n", message);
	exit(1);
}

static uint8_t hex_nibble(char character)
{
	if (character >= '0' && character <= '9')
		return (uint8_t)(character - '0');
	if (character >= 'a' && character <= 'f')
		return (uint8_t)(character - 'a' + 10);
	if (character >= 'A' && character <= 'F')
		return (uint8_t)(character - 'A' + 10);
	fail("track SHA-256 is not hexadecimal");
	return 0;
}

static void parse_sha256(const char *text, uint8_t output[32])
{
	if (strlen(text) != 64)
		fail("track SHA-256 must contain 64 hexadecimal characters");
	for (uint32_t i = 0; i < 32; ++i) {
		output[i] = (uint8_t)(
			hex_nibble(text[i * 2]) << 4 | hex_nibble(text[i * 2 + 1]));
	}
}

static void *allocate(size_t count, size_t size)
{
	void *memory = calloc(count, size);
	if (memory == NULL)
		fail("out of memory");
	return memory;
}

/* xorshift64*: deterministic across runs and machines. */
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint32_t rng_next(void)
{
	rng_state ^= rng_state >> 12;
	rng_state ^= rng_state << 25;
	rng_state ^= rng_state >> 27;
	return (uint32_t)((rng_state * 0x2545F4914F6CDD1Dull) >> 32);
}

/* longitudinal * 3 + steering: gas 65%, coast 15%, brake 10%, both 10%. */
static uint8_t random_action(void)
{
	uint32_t roll = rng_next() % 100;
	uint32_t longitudinal = roll < 65 ? 1 : roll < 80 ? 0 : roll < 90 ? 2 : 3;
	return (uint8_t)(longitudinal * 3 + rng_next() % 3);
}

typedef struct {
	const char *name;
	size_t offset;
	size_t size;
} Region;

#define REGION(type, field) { #field, offsetof(type, field), sizeof(((type *)0)->field) }

static const Region SNAPSHOT_REGIONS[] = {
	REGION(TmnfEnvSnapshot, version),
	REGION(TmnfEnvSnapshot, live_state),
	REGION(TmnfEnvSnapshot, committed_state),
	REGION(TmnfEnvSnapshot, temp_state),
	REGION(TmnfEnvSnapshot, dyna_params),
	REGION(TmnfEnvSnapshot, car),
	REGION(TmnfEnvSnapshot, wheels),
	REGION(TmnfEnvSnapshot, aux),
	REGION(TmnfEnvSnapshot, aux_wheels),
	REGION(TmnfEnvSnapshot, contact),
	REGION(TmnfEnvSnapshot, contact_wheels),
	REGION(TmnfEnvSnapshot, contact_body_refs),
	REGION(TmnfEnvSnapshot, model6_state),
	REGION(TmnfEnvSnapshot, compute_state),
	REGION(TmnfEnvSnapshot, contact_timer_tick),
	REGION(TmnfEnvSnapshot, physics_tick_time),
	REGION(TmnfEnvSnapshot, dyna_dirty),
	REGION(TmnfEnvSnapshot, race_state),
	REGION(TmnfEnvSnapshot, episode_id),
	REGION(TmnfEnvSnapshot, episode_return),
	REGION(TmnfEnvSnapshot, race_state_valid),
	REGION(TmnfEnvSnapshot, reset_pending),
	REGION(TmnfEnvSnapshot, vehicle_context_valid),
};

static const Region RESULT_REGIONS[] = {
	REGION(TmnfStepResult, observation),
	REGION(TmnfStepResult, final_observation),
	REGION(TmnfStepResult, reward),
	REGION(TmnfStepResult, transition_discount),
	REGION(TmnfStepResult, completed_episode_return),
	REGION(TmnfStepResult, completed_episode_ticks),
	REGION(TmnfStepResult, race_time_ms),
	REGION(TmnfStepResult, executed_ticks),
	REGION(TmnfStepResult, episode_id),
	REGION(TmnfStepResult, terminated),
	REGION(TmnfStepResult, truncated),
	REGION(TmnfStepResult, final_observation_valid),
	REGION(TmnfStepResult, reset_only),
	REGION(TmnfStepResult, termination_reason),
};

static void report_regions(
	const char *what, uint32_t tick, uint32_t env_index,
	const uint8_t *cpu, const uint8_t *gpu, size_t size,
	const Region *regions, size_t region_count)
{
	fprintf(stderr,
		"cuda lockstep: %s differs at tick %u env %u:\n", what, tick,
		env_index);
	for (size_t r = 0; r < region_count; ++r) {
		const Region *region = &regions[r];
		if (memcmp(cpu + region->offset, gpu + region->offset,
				region->size) == 0)
			continue;
		size_t at = 0;
		while (cpu[region->offset + at] == gpu[region->offset + at])
			++at;
		at &= ~(size_t)3;
		uint32_t wa = 0, wb = 0;
		memcpy(&wa, cpu + region->offset + at, 4);
		memcpy(&wb, gpu + region->offset + at, 4);
		fprintf(stderr, "  %-24s +%-5zu cpu %08x gpu %08x\n",
			region->name, at, wa, wb);
	}
	size_t differing = 0;
	for (size_t i = 0; i < size; ++i)
		differing += cpu[i] != gpu[i];
	fprintf(stderr, "  %zu of %zu bytes differ\n", differing, size);
	exit(1);
}

static void compare_snapshots(
	uint32_t tick, uint32_t count, const TmnfEnvSnapshot *cpu,
	const TmnfEnvSnapshot *gpu)
{
	for (uint32_t i = 0; i < count; ++i) {
		if (memcmp(&cpu[i], &gpu[i], sizeof(*cpu)) != 0) {
			report_regions("snapshot", tick, i,
				(const uint8_t *)&cpu[i], (const uint8_t *)&gpu[i],
				sizeof(*cpu), SNAPSHOT_REGIONS,
				sizeof(SNAPSHOT_REGIONS) / sizeof(SNAPSHOT_REGIONS[0]));
		}
	}
}

static void compare_results(
	uint32_t tick, uint32_t count, const TmnfStepResult *cpu,
	const TmnfStepResult *gpu)
{
	for (uint32_t i = 0; i < count; ++i) {
		if (memcmp(&cpu[i], &gpu[i], sizeof(*cpu)) != 0) {
			report_regions("step result", tick, i,
				(const uint8_t *)&cpu[i], (const uint8_t *)&gpu[i],
				sizeof(*cpu), RESULT_REGIONS,
				sizeof(RESULT_REGIONS) / sizeof(RESULT_REGIONS[0]));
		}
	}
}

static void compare_observations(
	uint32_t tick, uint32_t count, const TmnfObservation *cpu,
	const TmnfObservation *gpu)
{
	for (uint32_t i = 0; i < count; ++i) {
		if (memcmp(&cpu[i], &gpu[i], sizeof(*cpu)) != 0) {
			fprintf(stderr,
				"cuda lockstep: observation differs at tick %u env %u\n",
				tick, i);
			exit(1);
		}
	}
}

typedef struct {
	TmnfTrack *track;
	TmnfRoute *route;
	const char *vehicle_path;
	uint32_t count;
	TmnfWorld **owners;
	TmnfPhysicsWorld **physics;
	uint32_t *player_indices;
	TmnfVecEnv cpu;
	TmnfCudaVecEnv *gpu;
	TmnfEnvSnapshot *cpu_snapshots;
	TmnfEnvSnapshot *gpu_snapshots;
	TmnfStepResult *cpu_results;
	TmnfStepResult *gpu_results;
	uint64_t compared_snapshot_bytes;
	uint64_t compared_result_bytes;
	uint32_t terminations[7];
} Pair;

static void pair_create(
	Pair *pair, TmnfTrack *track, TmnfRoute *route, const char *vehicle_path,
	uint32_t count, const TmnfVecEnvConfig *config)
{
	memset(pair, 0, sizeof(*pair));
	pair->track = track;
	pair->route = route;
	pair->vehicle_path = vehicle_path;
	pair->count = count;
	pair->owners = allocate(count, sizeof(*pair->owners));
	pair->physics = allocate(count, sizeof(*pair->physics));
	pair->player_indices = allocate(count, sizeof(*pair->player_indices));
	for (uint32_t i = 0; i < count; ++i) {
		pair->owners[i] = World_Create(track, vehicle_path);
		pair->physics[i] = World_GetPhysicsWorld(pair->owners[i]);
	}
	TmnfVecEnvConfig cpu_config = *config;
	cpu_config.thread_count = CPU_THREADS;
	TmnfVecEnv_Init(&pair->cpu, pair->physics, pair->player_indices, count,
		route, &cpu_config);
	TmnfCudaVecEnvLimits limits = TmnfCudaVecEnv_DefaultLimits();
	pair->gpu = TmnfCudaVecEnv_Create(
		track, pair->owners[0], route, count, config, &limits);
	if (pair->gpu == NULL)
		fail(TmnfCudaVecEnv_LastError());
	pair->cpu_snapshots = allocate(count, sizeof(TmnfEnvSnapshot));
	pair->gpu_snapshots = allocate(count, sizeof(TmnfEnvSnapshot));
	pair->cpu_results = allocate(count, sizeof(TmnfStepResult));
	pair->gpu_results = allocate(count, sizeof(TmnfStepResult));
}

static void pair_destroy(Pair *pair)
{
	TmnfCudaVecEnv_Destroy(pair->gpu);
	TmnfVecEnv_Destroy(&pair->cpu);
	for (uint32_t i = 0; i < pair->count; ++i)
		World_Destroy(pair->owners[i]);
	free(pair->owners);
	free(pair->physics);
	free(pair->player_indices);
	free(pair->cpu_snapshots);
	free(pair->gpu_snapshots);
	free(pair->cpu_results);
	free(pair->gpu_results);
}

static void pair_compare_snapshots(Pair *pair, uint32_t tick)
{
	TmnfVecEnv_Capture(&pair->cpu, pair->cpu_snapshots);
	TmnfCudaVecEnv_Capture(pair->gpu, pair->gpu_snapshots);
	compare_snapshots(tick, pair->count, pair->cpu_snapshots,
		pair->gpu_snapshots);
	pair->compared_snapshot_bytes +=
		(uint64_t)pair->count * sizeof(TmnfEnvSnapshot);
}

static void pair_account_results(Pair *pair, uint32_t tick)
{
	compare_results(tick, pair->count, pair->cpu_results, pair->gpu_results);
	pair->compared_result_bytes +=
		(uint64_t)pair->count * sizeof(TmnfStepResult);
	for (uint32_t i = 0; i < pair->count; ++i) {
		if (pair->cpu_results[i].terminated)
			pair->terminations[pair->cpu_results[i].termination_reason]++;
	}
}

static void pair_step_discrete(
	Pair *pair, uint32_t tick, const uint8_t *actions, uint32_t repeat)
{
	TmnfVecEnv_StepDiscrete(&pair->cpu, actions, repeat, pair->cpu_results);
	TmnfCudaVecEnv_StepDiscrete(pair->gpu, actions, repeat, pair->gpu_results);
	pair_account_results(pair, tick);
}

static void pair_step_analog(
	Pair *pair, uint32_t tick, const TmnfAnalogAction *actions,
	uint32_t repeat)
{
	TmnfVecEnv_StepAnalog(&pair->cpu, actions, repeat, pair->cpu_results);
	TmnfCudaVecEnv_StepAnalog(pair->gpu, actions, repeat, pair->gpu_results);
	pair_account_results(pair, tick);
}

/*
 * Round trip in both directions: CPU snapshots go into the GPU and GPU
 * snapshots into the CPU, each rotated by one environment so a restore that
 * silently kept the old state would show up on the next tick. Restore
 * observations must match too.
 */
static void pair_round_trip(Pair *pair, uint32_t tick)
{
	uint32_t count = pair->count;
	TmnfVecEnv_Capture(&pair->cpu, pair->cpu_snapshots);
	TmnfCudaVecEnv_Capture(pair->gpu, pair->gpu_snapshots);
	compare_snapshots(tick, count, pair->cpu_snapshots, pair->gpu_snapshots);

	TmnfEnvSnapshot *rotated = allocate(count, sizeof(*rotated));
	uint32_t *indices = allocate(count, sizeof(*indices));
	TmnfObservation *cpu_obs = allocate(count, sizeof(*cpu_obs));
	TmnfObservation *gpu_obs = allocate(count, sizeof(*gpu_obs));
	for (uint32_t i = 0; i < count; ++i) {
		rotated[i] = pair->cpu_snapshots[(i + 1) % count];
		indices[i] = count - 1 - i;
	}
	/* CPU snapshots into the GPU, indexed and reversed. */
	TmnfCudaVecEnv_RestoreIndices(pair->gpu, indices, count, rotated, gpu_obs);
	/* GPU snapshots (identical bytes) into the CPU the same way. */
	for (uint32_t i = 0; i < count; ++i)
		rotated[i] = pair->gpu_snapshots[(i + 1) % count];
	TmnfVecEnv_RestoreIndices(&pair->cpu, indices, count, rotated, cpu_obs);
	compare_observations(tick, count, cpu_obs, gpu_obs);
	pair_compare_snapshots(pair, tick);
	free(rotated);
	free(indices);
	free(cpu_obs);
	free(gpu_obs);
}

static void run_random(
	TmnfTrack *track, TmnfRoute *route, const char *vehicle_path,
	uint32_t count, uint32_t ticks)
{
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	config.max_race_ticks = 1500;
	config.horizon_ticks = 1200;
	config.off_track_grace_ticks = 60;
	config.stuck_grace_ticks = 120;
	config.respawn_action = 1;

	Pair pair;
	pair_create(&pair, track, route, vehicle_path, count, &config);
	TmnfObservation *cpu_obs = allocate(count, sizeof(*cpu_obs));
	TmnfObservation *gpu_obs = allocate(count, sizeof(*gpu_obs));
	TmnfVecEnv_Reset(&pair.cpu, NULL, cpu_obs);
	TmnfCudaVecEnv_Reset(pair.gpu, NULL, gpu_obs);
	compare_observations(0, count, cpu_obs, gpu_obs);
	pair_compare_snapshots(&pair, 0);

	/* Gas-biased actions held 1..60 ticks: cars reach speed, slide, hit
	 * walls, leave the corridor and stall, so every termination reason and
	 * the reset path are exercised. One hold in 40 starts with a respawn
	 * press: after a checkpoint the car moves to its spawn, before one the
	 * episode ends with TMNF_TERMINATION_RESTART. */
	uint8_t *actions = allocate(count, 1);
	uint32_t *hold = allocate(count, sizeof(*hold));
	for (uint32_t tick = 1; tick <= ticks; ++tick) {
		for (uint32_t i = 0; i < count; ++i) {
			actions[i] &= (uint8_t)~TMNF_DISCRETE_RESPAWN_FLAG;
			if (hold[i] == 0) {
				actions[i] = random_action();
				if (rng_next() % 40 == 0)
					actions[i] |= TMNF_DISCRETE_RESPAWN_FLAG;
				hold[i] = 1 + rng_next() % 60;
			}
			hold[i]--;
		}
		pair_step_discrete(&pair, tick, actions, 1);
		pair_compare_snapshots(&pair, tick);
		if (tick == ticks / 2)
			pair_round_trip(&pair, tick);
	}
	printf("cuda lockstep: discrete %u envs x %u ticks, %llu snapshot bytes "
		"and %llu result bytes compared, 0 differ; terminations finish %u "
		"timeout %u off_track %u stuck %u fell %u restart %u\n",
		count, ticks,
		(unsigned long long)pair.compared_snapshot_bytes,
		(unsigned long long)pair.compared_result_bytes,
		pair.terminations[TMNF_TERMINATION_FINISH],
		pair.terminations[TMNF_TERMINATION_TIMEOUT],
		pair.terminations[TMNF_TERMINATION_OFF_TRACK],
		pair.terminations[TMNF_TERMINATION_STUCK],
		pair.terminations[TMNF_TERMINATION_FELL],
		pair.terminations[TMNF_TERMINATION_RESTART]);
	pair_destroy(&pair);

	/* Analog action space, NEXT_STEP autoreset, action repeat 1..3. */
	config.action_space = TMNF_ACTION_SPACE_ANALOG;
	config.autoreset_mode = TMNF_AUTORESET_NEXT_STEP;
	pair_create(&pair, track, route, vehicle_path, count, &config);
	TmnfAnalogAction *analog = allocate(count, sizeof(*analog));
	uint32_t analog_ticks = ticks / 5;
	for (uint32_t tick = 1; tick <= analog_ticks; ++tick) {
		for (uint32_t i = 0; i < count; ++i) {
			analog[i].respawn = 0;
			if (hold[i] == 0) {
				int32_t steer = (int32_t)(rng_next() % 131073) - 65536;
				analog[i].steer = (float)steer / 65536.0f;
				analog[i].gas = (uint8_t)(rng_next() % 4 != 0);
				analog[i].brake = (uint8_t)(rng_next() % 5 == 0);
				analog[i].respawn = (uint8_t)(rng_next() % 40 == 0);
				hold[i] = 1 + rng_next() % 30;
			}
			hold[i]--;
		}
		pair_step_analog(&pair, tick, analog, 1 + tick % 3);
		pair_compare_snapshots(&pair, tick);
	}
	printf("cuda lockstep: analog %u envs x %u decisions, %llu snapshot bytes "
		"and %llu result bytes compared, 0 differ\n",
		count, analog_ticks,
		(unsigned long long)pair.compared_snapshot_bytes,
		(unsigned long long)pair.compared_result_bytes);
	pair_destroy(&pair);
	free(analog);
	free(actions);
	free(hold);
	free(cpu_obs);
	free(gpu_obs);
}

static TMNFRaceInputs *read_inputs(const char *path, uint32_t *count)
{
	FILE *file = fopen(path, "rb");
	if (file == NULL)
		fail("cannot open input schedule");
	if (fseek(file, 0, SEEK_END) != 0)
		fail("cannot seek input schedule");
	long size = ftell(file);
	if (size <= 0 || size % sizeof(TMNFRaceInputs) != 0)
		fail("input schedule is not whole records");
	rewind(file);
	*count = (uint32_t)(size / sizeof(TMNFRaceInputs));
	TMNFRaceInputs *inputs = malloc((size_t)size);
	if (inputs == NULL || fread(inputs, sizeof(*inputs), *count, file) != *count)
		fail("cannot read input schedule");
	fclose(file);
	return inputs;
}

/* tests/vec_env_matches_replay.c: the game's mapper decides the action. */
/* Digital-steer schedules replay through the discrete space; analog-steer
 * ones (TMInterface's (float)(-q) / 65536 packets with digital gas and
 * brake) through the analog space, whose quantizer reproduces q exactly. */
static int analog_schedule(const TMNFRaceInputs *inputs, uint32_t count)
{
	int analog = 0;
	for (uint32_t i = 0; i < count; ++i) {
		if (inputs[i].gas_analog_time != 0)
			fail("replay mode has no analog gas");
		if (inputs[i].steer_analog_time != 0)
			analog = 1;
	}
	return analog;
}

static TmnfAnalogAction analog_action(const TMNFRaceInputs *input)
{
	if (input->steer_left != 0 || input->steer_right != 0)
		fail("analog schedule mixes digital steering");
	TmnfAnalogAction action;
	memset(&action, 0, sizeof(action));
	action.steer = -input->steer_analog;
	action.gas = input->accelerate != 0;
	action.brake = input->brake != 0;
	action.respawn = input->respawn != 0;
	return action;
}

static uint8_t discrete_action(const TMNFRaceInputs *input)
{
	CSceneVehicleCar scratch;
	memset(&scratch, 0, sizeof(scratch));
	CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
		input, &scratch);
	if (input->steer_analog_time != 0)
		fail("discrete replay needs a digital schedule");
	uint32_t steering = scratch.input_steer < 0.0f ? 0
		: scratch.input_steer > 0.0f ? 2 : 1;
	uint32_t longitudinal = (scratch.input_gas != 0.0f ? 1u : 0u) +
		(scratch.input_brake != 0.0f ? 2u : 0u);
	uint8_t action = (uint8_t)(longitudinal * 3 + steering);
	if (input->respawn != 0)
		action |= TMNF_DISCRETE_RESPAWN_FLAG;
	return action;
}

static void run_replay(
	TmnfTrack *track, TmnfRoute *route, const char *vehicle_path,
	const char *inputs_path)
{
	uint32_t tick_count;
	TMNFRaceInputs *inputs = read_inputs(inputs_path, &tick_count);
	int analog = analog_schedule(inputs, tick_count);
	static const uint32_t REPEATS[] = { 1, 5 };
	for (uint32_t r = 0; r < 2; ++r) {
		uint32_t repeat = REPEATS[r];
		TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
		config.respawn_action = 1;
		if (analog)
			config.action_space = TMNF_ACTION_SPACE_ANALOG;
		Pair pair;
		pair_create(&pair, track, route, vehicle_path, 1, &config);
		uint32_t tick = 0;
		uint32_t finishes = 0;
		uint32_t finish_ms = 0;
		float episode_return = 0.0f;
		while (tick < tick_count) {
			if (analog) {
				TmnfAnalogAction action = analog_action(&inputs[tick]);
				pair_step_analog(&pair, tick, &action, repeat);
			} else {
				uint8_t action = discrete_action(&inputs[tick]);
				pair_step_discrete(&pair, tick, &action, repeat);
			}
			pair_compare_snapshots(&pair, tick);
			const TmnfStepResult *result = &pair.cpu_results[0];
			if (result->executed_ticks == 0 ||
				result->executed_ticks > repeat)
				fail("executed tick count is out of range");
			tick += result->executed_ticks;
			if (result->terminated &&
				result->termination_reason == TMNF_TERMINATION_FINISH) {
				finishes++;
				finish_ms = result->race_time_ms;
				episode_return = result->completed_episode_return;
			}
		}
		printf("cuda lockstep: %s replay repeat %u, %u ticks, %llu snapshot "
			"bytes and %llu result bytes compared, 0 differ; finishes %u "
			"(first %u ms, return %.6f), terminations finish %u timeout %u "
			"off_track %u stuck %u fell %u restart %u\n",
			analog ? "analog" : "discrete", repeat, tick,
			(unsigned long long)pair.compared_snapshot_bytes,
			(unsigned long long)pair.compared_result_bytes,
			finishes, finish_ms, (double)episode_return,
			pair.terminations[TMNF_TERMINATION_FINISH],
			pair.terminations[TMNF_TERMINATION_TIMEOUT],
			pair.terminations[TMNF_TERMINATION_OFF_TRACK],
			pair.terminations[TMNF_TERMINATION_STUCK],
			pair.terminations[TMNF_TERMINATION_FELL],
			pair.terminations[TMNF_TERMINATION_RESTART]);
		pair_destroy(&pair);
	}
	free(inputs);
}

int main(int argc, char **argv)
{
	if (argc < 6) {
		fprintf(stderr,
			"usage: cuda_lockstep random TRACK VEHICLE ROUTE SHA256 ENVS TICKS\n"
			"       cuda_lockstep replay TRACK VEHICLE ROUTE SHA256 INPUTS\n");
		return 2;
	}
	uint8_t sha256[32];
	parse_sha256(argv[5], sha256);
	TmnfTrack *track = TmnfTrack_Load(argv[2], sha256);
	TmnfRoute *route = TmnfRoute_Load(argv[4], sha256);
	if (strcmp(argv[1], "random") == 0 && argc == 8) {
		uint32_t count = (uint32_t)strtoul(argv[6], NULL, 10);
		uint32_t ticks = (uint32_t)strtoul(argv[7], NULL, 10);
		if (count == 0 || count > MAX_ENVS || ticks == 0)
			fail("invalid env count or tick count");
		run_random(track, route, argv[3], count, ticks);
	} else if (strcmp(argv[1], "replay") == 0 && argc == 7) {
		run_replay(track, route, argv[3], argv[6]);
	} else {
		fail("unknown mode");
	}
	TmnfRoute_Unload(route);
	TmnfTrack_Unload(track);
	return 0;
}
