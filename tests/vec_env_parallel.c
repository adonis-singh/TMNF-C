#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "route.h"
#include "vec_env.h"
#include "world.h"

enum {
	MAX_BATCH_SIZE = 16,
	ROLLOUT_STEPS = 3000,
};

typedef struct {
	uint32_t environment_count;
	uint32_t thread_count;
} Sharding;

static const Sharding SHARDINGS[] = {
	{13, 3},
	{5, 8},
	{16, 32},
};

static const uint8_t A01_SHA256[32] = {
	0xf0, 0xa8, 0x70, 0x80, 0x9b, 0xe9, 0x9d, 0xa2,
	0xcb, 0x36, 0xad, 0x5d, 0xf4, 0x3a, 0x2c, 0xf6,
	0x3d, 0x8f, 0x74, 0xfe, 0x4a, 0xc3, 0x47, 0x0e,
	0xca, 0xc6, 0x8b, 0x9e, 0x97, 0x62, 0x5d, 0xc3,
};

static void fail(uint32_t step, uint32_t env_index, const char *field)
{
	fprintf(stderr,
		"vec env parallel: step %u env %u changed %s bytes\n",
		step, env_index, field);
	exit(1);
}

static void compare_result(
	uint32_t step, uint32_t env_index,
	const TmnfStepResult *serial, const TmnfStepResult *parallel)
{
	if (memcmp(
			&serial->observation, &parallel->observation,
			sizeof(serial->observation)) != 0) {
		fail(step, env_index, "observation");
	}
	if (memcmp(
			&serial->final_observation, &parallel->final_observation,
			sizeof(serial->final_observation)) != 0) {
		fail(step, env_index, "final observation");
	}
	if (memcmp(
			&serial->reward, &parallel->reward,
			sizeof(serial->reward)) != 0) {
		fail(step, env_index, "reward");
	}
	if (serial->terminated != parallel->terminated ||
		serial->truncated != parallel->truncated ||
		serial->termination_reason != parallel->termination_reason ||
		serial->final_observation_valid !=
			parallel->final_observation_valid ||
		serial->reset_only != parallel->reset_only) {
		fail(step, env_index, "terminal flag");
	}
	if (memcmp(serial, parallel, sizeof(*serial)) != 0)
		fail(step, env_index, "complete result");
}

static void check_gate_sidecars(TmnfVecEnv *env, uint32_t step,
 const TmnfStepResult *results, const TmnfGateObservations *gates,
 const TmnfGateObservations *final_gates)
{
 TmnfGateObservations current[MAX_BATCH_SIZE], zero = {0};
 TmnfVecEnv_ObserveGates(env, current);
 for (uint32_t i = 0; i < env->count; ++i) {
  const TmnfStepResult *r = &results[i];
  if (!r->final_observation_valid) {
   if (memcmp(&final_gates[i], &zero, sizeof(zero))) fail(step,i,"nonterminal gates");
  } else if (!final_gates[i].gates[7].valid && !final_gates[i].finished) {
   fail(step,i,"missing terminal finish gate");
  }
  const TmnfGateObservations *expected =
   r->final_observation_valid && env->config.autoreset_mode == TMNF_AUTORESET_NEXT_STEP
   ? &final_gates[i] : &current[i];
  if (memcmp(&gates[i], expected, sizeof(*expected))) fail(step,i,"gate autoreset semantics");
 }
 /* Derived observations must survive snapshot capture/restore without adding
  * fields to the snapshot or inheriting the previous query's output. */
 if (step % 500 == 0) {
  TmnfEnvSnapshot snapshots[MAX_BATCH_SIZE];
  TmnfVecEnv_Capture(env, snapshots);
  TmnfVecEnv_Restore(env, snapshots);
  TmnfGateObservations restored[MAX_BATCH_SIZE];
  TmnfVecEnv_ObserveGates(env, restored);
  if (memcmp(current, restored, env->count * sizeof(current[0]))) fail(step,0,"restored gates");
 }
}

typedef struct {
	TmnfVecEnv *env;
	const uint8_t *actions;
	TmnfStepResult *results;
} ConcurrentStep;

static void *run_concurrent_step(void *argument)
{
	ConcurrentStep *step = argument;
	TmnfVecEnv_StepDiscrete(
		step->env, step->actions, UINT32_MAX, step->results);
	return NULL;
}

static void require_concurrent_step_aborts(
	const char *track_path,
	const char *vehicle_path,
	const char *route_path)
{
	pid_t child = fork();
	if (child < 0) {
		perror("vec env parallel: fork");
		exit(1);
	}
	if (child == 0) {
		TmnfTrack *track = TmnfTrack_Load(track_path, A01_SHA256);
		TmnfWorld *owner = World_Create(track, vehicle_path);
		TmnfPhysicsWorld *world = World_GetPhysicsWorld(owner);
		uint32_t player_index = 0;
		TmnfRoute *route = TmnfRoute_Load(route_path, A01_SHA256);
		TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
		config.max_race_ticks = UINT32_MAX;
		config.horizon_ticks = UINT32_MAX;
		config.off_track_grace_ticks = UINT32_MAX;
		config.stuck_grace_ticks = UINT32_MAX;
		TmnfVecEnv env;
		TmnfVecEnv_Init(
			&env, &world, &player_index, 1, route, &config);

		uint8_t action = 4;
		TmnfStepResult first_result;
		ConcurrentStep step = {
			.env = &env,
			.actions = &action,
			.results = &first_result,
		};
		pthread_t worker;
		if (pthread_create(
				&worker, NULL, run_concurrent_step, &step) != 0) {
			_exit(4);
		}
		while (atomic_load_explicit(
				&env.active_operation,
				memory_order_acquire) == 0) {
			sched_yield();
		}

		TmnfStepResult second_result;
		TmnfVecEnv_StepDiscrete(
			&env, &action, 1, &second_result);
		_exit(5);
	}

	int status;
	if (waitpid(child, &status, 0) != child) {
		perror("vec env parallel: waitpid");
		exit(1);
	}
	if (!WIFSIGNALED(status) || WTERMSIG(status) != SIGABRT) {
		fprintf(
			stderr,
			"vec env parallel: concurrent step status %#x, "
			"expected SIGABRT\n",
			status);
		exit(1);
	}
	printf("vec env parallel: concurrent step aborted\n");
}

static void run_sharding(
	const char *track_path,
	const char *vehicle_path,
	const char *route_path,
	const Sharding *sharding, uint32_t autoreset_mode)
{
	TmnfWorld *owners[2][MAX_BATCH_SIZE] = {{0}};
	TmnfPhysicsWorld *worlds[2][MAX_BATCH_SIZE] = {{0}};
	uint32_t player_indices[MAX_BATCH_SIZE] = {0};
	TmnfTrack *track = TmnfTrack_Load(track_path, A01_SHA256);
	for (uint32_t batch = 0; batch < 2; ++batch) {
		for (uint32_t i = 0;
			i < sharding->environment_count; ++i) {
			owners[batch][i] =
				World_Create(track, vehicle_path);
			worlds[batch][i] =
				World_GetPhysicsWorld(owners[batch][i]);
		}
	}

	TmnfRoute *route = TmnfRoute_Load(route_path, A01_SHA256);
	TmnfVecEnvConfig serial_config = TmnfVecEnv_DefaultConfig();
	serial_config.thread_count = 1;
	serial_config.autoreset_mode = autoreset_mode;
	serial_config.max_race_ticks = 1000;
	serial_config.horizon_ticks = 800;
	serial_config.off_track_grace_ticks = 50;
	serial_config.stuck_grace_ticks = 80;
	TmnfVecEnvConfig parallel_config = serial_config;
	parallel_config.thread_count = sharding->thread_count;

	TmnfVecEnv serial;
	TmnfVecEnv parallel;
	TmnfVecEnv_Init(
		&serial, worlds[0], player_indices,
		sharding->environment_count,
		route, &serial_config);
	TmnfVecEnv_Init(
		&parallel, worlds[1], player_indices,
		sharding->environment_count,
		route, &parallel_config);

	TmnfVecEnv_Reset(&serial, NULL, NULL);
	TmnfVecEnv_Reset(&parallel, NULL, NULL);

	uint8_t actions[MAX_BATCH_SIZE];
	TmnfStepResult serial_results[MAX_BATCH_SIZE];
	TmnfStepResult parallel_results[MAX_BATCH_SIZE];
	TmnfGateObservations gates[MAX_BATCH_SIZE], final_gates[MAX_BATCH_SIZE];
	for (uint32_t step = 0; step < ROLLOUT_STEPS; ++step) {
		for (uint32_t i = 0;
			i < sharding->environment_count; ++i) {
			actions[i] = (uint8_t)((step * 5 + i * 7) % 12);
		}
		uint32_t action_repeat = 1 + step % 3;
		TmnfVecEnv_StepDiscrete(
			&serial, actions, action_repeat, serial_results);
		TmnfVecEnv_StepDiscreteWithGates(
			&parallel, actions, action_repeat, parallel_results, gates, final_gates);
		check_gate_sidecars(&parallel, step, parallel_results, gates, final_gates);
		for (uint32_t i = 0;
			i < sharding->environment_count; ++i) {
			compare_result(
				step, i, &serial_results[i],
				&parallel_results[i]);
		}
	}

	TmnfVecEnv_Destroy(&parallel);
	TmnfVecEnv_Destroy(&serial);
	TmnfRoute_Unload(route);
	for (uint32_t batch = 0; batch < 2; ++batch) {
		for (uint32_t i = 0;
			i < sharding->environment_count; ++i) {
			World_Destroy(owners[batch][i]);
		}
	}
	TmnfTrack_Unload(track);
	printf(
		"vec env parallel: %u environments, %u rollout steps, "
		"1 versus %u threads byte-identical\n",
		sharding->environment_count, ROLLOUT_STEPS,
		sharding->thread_count);
}

static void require_analog_quantization(void)
{
	static const int32_t exact_values[] = {
		-65536, -32768, -1, 0, 1, 32768, 65536,
	};
	for (uint32_t i = 0;
		i < sizeof(exact_values) / sizeof(exact_values[0]); ++i) {
		int32_t value = exact_values[i];
		float normalized = (float)value / 65536.0f;
		if (TmnfVecEnv_QuantizeAnalogSteer(normalized) != value) {
			fprintf(stderr,
				"vec env parallel: analog value %d did not "
				"round-trip\n", value);
			exit(1);
		}
	}
	if (TmnfVecEnv_QuantizeAnalogSteer(0x1p-17f) != 1 ||
		TmnfVecEnv_QuantizeAnalogSteer(-0x1p-17f) != -1) {
		fprintf(stderr,
			"vec env parallel: analog half-step symmetry changed\n");
		exit(1);
	}
	for (int32_t value = 1; value <= 65536; value += 257) {
		float normalized = (float)value / 65536.0f;
		if (TmnfVecEnv_QuantizeAnalogSteer(normalized) != value ||
			TmnfVecEnv_QuantizeAnalogSteer(-normalized) != -value) {
			fprintf(stderr,
				"vec env parallel: analog quantization lost "
				"symmetry at %d\n", value);
			exit(1);
		}
	}
	printf(
		"vec env parallel: analog quantization range and symmetry passed\n");
}

static void run_analog_sharding(
	const char *track_path,
	const char *vehicle_path,
	const char *route_path,
	const Sharding *sharding, uint32_t autoreset_mode)
{
	TmnfWorld *owners[2][MAX_BATCH_SIZE] = {{0}};
	TmnfPhysicsWorld *worlds[2][MAX_BATCH_SIZE] = {{0}};
	uint32_t player_indices[MAX_BATCH_SIZE] = {0};
	TmnfTrack *track = TmnfTrack_Load(track_path, A01_SHA256);
	for (uint32_t batch = 0; batch < 2; ++batch) {
		for (uint32_t i = 0;
			i < sharding->environment_count; ++i) {
			owners[batch][i] =
				World_Create(track, vehicle_path);
			worlds[batch][i] =
				World_GetPhysicsWorld(owners[batch][i]);
		}
	}

	TmnfRoute *route = TmnfRoute_Load(route_path, A01_SHA256);
	TmnfVecEnvConfig serial_config = TmnfVecEnv_DefaultConfig();
	serial_config.thread_count = 1;
	serial_config.autoreset_mode = autoreset_mode;
	serial_config.max_race_ticks = 1000;
	serial_config.horizon_ticks = 800;
	serial_config.off_track_grace_ticks = 50;
	serial_config.stuck_grace_ticks = 80;
	serial_config.action_space = TMNF_ACTION_SPACE_ANALOG;
	TmnfVecEnvConfig parallel_config = serial_config;
	parallel_config.thread_count = sharding->thread_count;

	TmnfVecEnv serial;
	TmnfVecEnv parallel;
	TmnfVecEnv_Init(
		&serial, worlds[0], player_indices,
		sharding->environment_count,
		route, &serial_config);
	TmnfVecEnv_Init(
		&parallel, worlds[1], player_indices,
		sharding->environment_count,
		route, &parallel_config);

	TmnfVecEnv_Reset(&serial, NULL, NULL);
	TmnfVecEnv_Reset(&parallel, NULL, NULL);

	TmnfAnalogAction actions[MAX_BATCH_SIZE];
	TmnfStepResult serial_results[MAX_BATCH_SIZE];
	TmnfStepResult parallel_results[MAX_BATCH_SIZE];
	TmnfGateObservations gates[MAX_BATCH_SIZE], final_gates[MAX_BATCH_SIZE];
	for (uint32_t step = 0; step < ROLLOUT_STEPS; ++step) {
		for (uint32_t i = 0;
			i < sharding->environment_count; ++i) {
			int32_t steer =
				(int32_t)((step * 997 + i * 8191) % 131073) -
				65536;
			actions[i] = (TmnfAnalogAction){
				.steer = (float)steer / 65536.0f,
				.gas = (uint8_t)((step + i) % 2),
				.brake = (uint8_t)((step / 3 + i) % 2),
			};
		}
		uint32_t action_repeat = 1 + step % 3;
		TmnfVecEnv_StepAnalog(
			&serial, actions, action_repeat, serial_results);
		TmnfVecEnv_StepAnalogWithGates(
			&parallel, actions, action_repeat, parallel_results, gates, final_gates);
		check_gate_sidecars(&parallel, step, parallel_results, gates, final_gates);
		for (uint32_t i = 0;
			i < sharding->environment_count; ++i) {
			compare_result(
				step, i, &serial_results[i],
				&parallel_results[i]);
		}
	}

	TmnfVecEnv_Destroy(&parallel);
	TmnfVecEnv_Destroy(&serial);
	TmnfRoute_Unload(route);
	for (uint32_t batch = 0; batch < 2; ++batch) {
		for (uint32_t i = 0;
			i < sharding->environment_count; ++i) {
			World_Destroy(owners[batch][i]);
		}
	}
	TmnfTrack_Unload(track);
	printf(
		"vec env parallel: analog %u environments, %u rollout steps, "
		"1 versus %u threads byte-identical\n",
		sharding->environment_count, ROLLOUT_STEPS,
		sharding->thread_count);
}

int main(int argc, char **argv)
{
	if (argc != 4) {
		fprintf(stderr,
			"usage: vec_env_parallel TRACK VEHICLE ROUTE\n");
		return 2;
	}
	require_analog_quantization();
	require_concurrent_step_aborts(argv[1], argv[2], argv[3]);
	for (uint32_t i = 0;
		i < sizeof(SHARDINGS) / sizeof(SHARDINGS[0]); ++i) {
		run_sharding(argv[1], argv[2], argv[3], &SHARDINGS[i], TMNF_AUTORESET_SAME_STEP);
		run_sharding(argv[1], argv[2], argv[3], &SHARDINGS[i], TMNF_AUTORESET_NEXT_STEP);
		run_analog_sharding(
			argv[1], argv[2], argv[3], &SHARDINGS[i], TMNF_AUTORESET_SAME_STEP);
		run_analog_sharding(argv[1], argv[2], argv[3], &SHARDINGS[i], TMNF_AUTORESET_NEXT_STEP);
	}
	return 0;
}
