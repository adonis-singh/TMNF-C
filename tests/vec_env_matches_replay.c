/*
 * Guard: the RL step path (TmnfVecEnv_StepDiscrete / TmnfVecEnv_StepAnalog)
 * must produce exactly the CHmsStateDyna and game-state bytes that the World
 * replay path produces for a committed input schedule, which is itself
 * byte-exact against the game (replay_tick). Digital schedules run through the
 * discrete action space at action repeat 1 tick by tick, and again at repeat 5
 * where the per-decision states must match. Analog schedules run through the
 * analog action space at repeat 1. The env must register the finish at the
 * game's race time. A schedule with respawn presses (TMNFRaceInputs.respawn)
 * runs with the respawn action enabled and skips repeat 5: the press is a
 * single tick, so a repeat window containing it is not constant.
 *
 * Every schedule runs in three regimes of the same env instance, all of which
 * must be identical to the World path: the first episode after Init, a second
 * episode after a warm drive of unrelated inputs and a reset (the training
 * regime, F2b), and an episode after a snapshot restore followed by a reset.
 * A snapshot captured mid-drive and restored into a never-stepped instance
 * must also step in lockstep with its source.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "race.h"
#include "route.h"
#include "vec_env.h"
#include "world.h"

enum {
	TICK_MS = 10,
	CAR_SIZE = 0x878,
	WHEEL_SIZE = 0x2fc,
	GAME_STATE_SIZE = CAR_SIZE + 4 * WHEEL_SIZE,
	WARM_TICKS = 60,
	FRESH_RESTORE_TICKS = 300,
};

typedef enum {
	REGIME_FIRST_EPISODE,
	REGIME_SECOND_EPISODE,
	REGIME_RESET_AFTER_RESTORE,
	REGIME_COUNT,
} EnvRegime;

static const char *REGIME_NAMES[REGIME_COUNT] = {
	"first episode",
	"second episode",
	"reset after restore",
};

static void fail(const char *message)
{
	fprintf(stderr, "vec env matches replay: %s\n", message);
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

/* The game's own input mapper decides which timestamped field is effective. */
static void effective_inputs(
	const TMNFRaceInputs *input, float *gas, float *brake, float *steer)
{
	CSceneVehicleCar scratch;
	memset(&scratch, 0, sizeof(scratch));
	CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
		input, &scratch);
	*gas = scratch.input_gas;
	*brake = scratch.input_brake;
	*steer = scratch.input_steer;
}

static int schedule_is_digital(const TMNFRaceInputs *inputs, uint32_t count)
{
	for (uint32_t tick = 0; tick < count; ++tick) {
		if (inputs[tick].steer_analog_time != 0 ||
			inputs[tick].gas_analog_time != 0)
			return 0;
	}
	return 1;
}

static int schedule_respawns(const TMNFRaceInputs *inputs, uint32_t count)
{
	for (uint32_t tick = 0; tick < count; ++tick) {
		if (inputs[tick].respawn != 0)
			return 1;
	}
	return 0;
}

static uint8_t discrete_action(const TMNFRaceInputs *input)
{
	float gas, brake, steer;
	effective_inputs(input, &gas, &brake, &steer);
	uint32_t steering = steer < 0.0f ? 0 : steer > 0.0f ? 2 : 1;
	uint32_t longitudinal =
		(gas != 0.0f ? 1u : 0u) + (brake != 0.0f ? 2u : 0u);
	uint8_t action = (uint8_t)(longitudinal * 3 + steering);
	if (input->respawn != 0)
		action |= TMNF_DISCRETE_RESPAWN_FLAG;
	return action;
}

/* analog_input() quantizes steer to k/65536 and writes (float)(-k)/65536 into
 * the packet, which the mapper negates back. Every effective steer of a game
 * schedule is k/65536, so action.steer = effective steer round-trips. */
static TmnfAnalogAction analog_action(const TMNFRaceInputs *input)
{
	float gas, brake, steer;
	effective_inputs(input, &gas, &brake, &steer);
	return (TmnfAnalogAction){
		.steer = steer,
		.gas = gas != 0.0f,
		.brake = brake != 0.0f,
		.respawn = input->respawn != 0,
	};
}

typedef struct {
	CHmsStateDyna dyna;
	uint8_t game_state[GAME_STATE_SIZE];
	uint32_t timer_tick;
} PathState;

static void read_state(TmnfWorld *world, PathState *state)
{
	state->dyna = *World_GetPlayerState(world);
	World_WritePlayerGameState(
		world, state->game_state, state->game_state + CAR_SIZE);
	state->timer_tick = World_GetTimerTick(world);
}

/*
 * Digital schedules must match byte for byte. The analog action space writes
 * neutral steer as -0.0 (TMInterface's analog zero, which the game's mapper
 * negates, exactly as the game does with an analog controller), while a
 * schedule's digital neutral ticks map to +0.0; that sign propagates into
 * zero-valued steering fields. For analog schedules the comparison therefore
 * treats +0.0 and -0.0 as equal words. Any non-zero difference, including one
 * a signed zero would later cause through division or atan2, still fails.
 */
static int states_match(
	const PathState *replay, const PathState *env, int analog)
{
	if (memcmp(replay, env, sizeof(*replay)) == 0)
		return 1;
	if (!analog)
		return 0;
	_Static_assert(sizeof(PathState) % 4 == 0, "state is whole words");
	const uint8_t *a = (const uint8_t *)replay;
	const uint8_t *b = (const uint8_t *)env;
	for (uint32_t offset = 0; offset < sizeof(*replay); offset += 4) {
		uint32_t wa, wb;
		memcpy(&wa, a + offset, 4);
		memcpy(&wb, b + offset, 4);
		if (wa != wb && ((wa | wb) & 0x7FFFFFFFu) != 0)
			return 0;
	}
	return 1;
}

static void report_divergence(
	const char *label, uint32_t tick, const PathState *replay,
	const PathState *env)
{
	const uint8_t *a = (const uint8_t *)replay;
	const uint8_t *b = (const uint8_t *)env;
	uint32_t byte = 0;
	for (; byte < sizeof(*replay); byte += 4) {
		uint32_t wa, wb;
		memcpy(&wa, a + byte, 4);
		memcpy(&wb, b + byte, 4);
		if (wa != wb && ((wa | wb) & 0x7FFFFFFFu) != 0)
			break;
	}
	const char *region;
	uint32_t local;
	if (byte < sizeof(replay->dyna)) {
		region = "dyna";
		local = byte;
	} else if (byte < offsetof(PathState, timer_tick)) {
		region = "game_state";
		local = byte - (uint32_t)sizeof(replay->dyna);
	} else {
		region = "timer_tick";
		local = 0;
	}
	uint32_t ea, eb;
	memcpy(&ea, a + (byte & ~3u), 4);
	memcpy(&eb, b + (byte & ~3u), 4);
	fprintf(stderr,
		"vec env matches replay: %s: divergence at tick %u, %s offset 0x%x "
		"(replay 0x%08x, env 0x%08x); replay pos (%.9g %.9g %.9g) env pos "
		"(%.9g %.9g %.9g)\n",
		label, tick, region, local & ~3u, ea, eb,
		replay->dyna.pos.x, replay->dyna.pos.y, replay->dyna.pos.z,
		env->dyna.pos.x, env->dyna.pos.y, env->dyna.pos.z);
	exit(1);
}

/* Replay path: exactly what tests/replay_tick.c does per tick, with the race
 * layer following the run so a respawn press knows its spawn. */
static void run_replay_path(
	TmnfWorld *world, const TmnfRoute *route, const TMNFRaceInputs *inputs,
	uint32_t tick_count, PathState *states)
{
	TmnfPhysicsWorld *physics = World_GetPhysicsWorld(world);
	const TmnfPhysicsCorpus *corpus = &physics->corpora[0];
	physics->route = route;
	TmnfRaceState race;
	TmnfRace_Reset(route, &race, corpus->collision_corpus->live_iso);
	for (uint32_t tick = 0; tick < tick_count; ++tick) {
		if (inputs[tick].respawn != 0) {
			const GmIso4 *spawn = TmnfRace_RespawnLocation(&race);
			if (spawn == NULL)
				fail("schedule respawns before any checkpoint");
			World_Respawn(world, spawn);
		}
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
			&inputs[tick], World_GetPlayerVehicle(world));
		World_AdvanceTimer(world, TICK_MS);
		CHmsZoneDynamic_PhysicsStep2(physics, TICK_MS);
		read_state(world, &states[tick]);
		if (!race.finished) {
			(void)TmnfRace_Step(route, &race, physics->trigger_contacts,
				corpus->collision_corpus->live_iso);
		}
	}
}

typedef struct {
	TmnfTrack *track;
	const char *vehicle_path;
	TmnfRoute *route;
	const TMNFRaceInputs *inputs;
	uint32_t tick_count;
	const PathState *replay_states;
	uint32_t expected_finish_ms;
} Lap;

static void step_env(
	TmnfVecEnv *env, const TMNFRaceInputs *input, int analog,
	uint32_t action_repeat, TmnfStepResult *result)
{
	if (analog) {
		TmnfAnalogAction action = analog_action(input);
		TmnfVecEnv_StepAnalog(env, &action, action_repeat, result);
	} else {
		uint8_t action = discrete_action(input);
		TmnfVecEnv_StepDiscrete(env, &action, action_repeat, result);
	}
}

static TmnfVecEnvConfig env_config(const Lap *lap, int analog)
{
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	config.max_race_ticks = lap->tick_count + 1000;
	config.horizon_ticks = lap->tick_count + 1000;
	config.off_track_grace_ticks = UINT32_MAX;
	config.stuck_grace_ticks = UINT32_MAX;
	config.thread_count = 1;
	config.action_space =
		analog ? TMNF_ACTION_SPACE_ANALOG : TMNF_ACTION_SPACE_DISCRETE;
	config.respawn_action = schedule_respawns(lap->inputs, lap->tick_count);
	return config;
}

/* Inputs unrelated to the schedule: full gas with alternating hard steering
 * (analog: half steer) so the warm drive leaves grounded, sliding, steering
 * history behind that the reset must erase. */
static TMNFRaceInputs warm_input(uint32_t tick, int analog)
{
	uint32_t timestamp = (tick + 1) * TICK_MS;
	int left = (tick / 20) % 2 == 0;
	TMNFRaceInputs input = {
		.accelerate_time = timestamp,
		.accelerate = 1,
	};
	if (analog) {
		input.steer_analog_time = timestamp;
		input.steer_analog = left ? 0.5f : -0.5f;
	} else {
		input.steer_left_time = timestamp;
		input.steer_right_time = timestamp;
		input.steer_left = left;
		input.steer_right = !left;
	}
	return input;
}

static void drive_warm(TmnfVecEnv *env, int analog, uint32_t tick_count)
{
	for (uint32_t tick = 0; tick < tick_count; ++tick) {
		TMNFRaceInputs input = warm_input(tick, analog);
		TmnfStepResult result;
		step_env(env, &input, analog, 1, &result);
		if (result.executed_ticks != 1 || result.terminated ||
			result.truncated) {
			fail("warm drive ended unexpectedly");
		}
	}
}

static void enter_regime(TmnfVecEnv *env, int analog, EnvRegime regime)
{
	switch (regime) {
	case REGIME_FIRST_EPISODE:
		/* A second reset must land on the same state as the first. */
		TmnfVecEnv_Reset(env, NULL, NULL);
		return;
	case REGIME_SECOND_EPISODE:
		drive_warm(env, analog, WARM_TICKS);
		TmnfVecEnv_Reset(env, NULL, NULL);
		return;
	case REGIME_RESET_AFTER_RESTORE: {
		TmnfEnvSnapshot snapshot;
		drive_warm(env, analog, WARM_TICKS / 2);
		TmnfVecEnv_Capture(env, &snapshot);
		drive_warm(env, analog, WARM_TICKS / 2);
		TmnfVecEnv_Restore(env, &snapshot);
		TmnfVecEnv_Reset(env, NULL, NULL);
		return;
	}
	default:
		fail("unknown env regime");
	}
}

static void run_env_path(
	const Lap *lap, int analog, uint32_t action_repeat, EnvRegime regime)
{
	TmnfWorld *owner = World_Create(lap->track, lap->vehicle_path);
	TmnfPhysicsWorld *physics = World_GetPhysicsWorld(owner);
	uint32_t player_index = 0;
	TmnfVecEnvConfig config = env_config(lap, analog);
	TmnfVecEnv env;
	TmnfVecEnv_Init(&env, &physics, &player_index, 1, lap->route, &config);
	enter_regime(&env, analog, regime);
	char label[96];
	snprintf(label, sizeof(label), "%s repeat %u, %s",
		analog ? "analog" : "discrete", action_repeat,
		REGIME_NAMES[regime]);

	uint32_t tick = 0;
	uint32_t finish_ms = 0;
	while (tick < lap->tick_count) {
		TmnfStepResult result;
		step_env(&env, &lap->inputs[tick], analog, action_repeat, &result);
		if (result.executed_ticks == 0 || result.executed_ticks > action_repeat)
			fail("executed tick count is out of range");
		if (tick + result.executed_ticks > lap->tick_count)
			fail("env stepped past the schedule");
		if (analog && result.executed_ticks != 1)
			fail("analog schedules must be replayed one tick per decision");
		for (uint32_t k = 1; k < result.executed_ticks; ++k) {
			if (discrete_action(&lap->inputs[tick + k]) !=
				discrete_action(&lap->inputs[tick]))
				fail("schedule is not constant across the action repeat");
		}
		tick += result.executed_ticks;
		if (result.terminated) {
			if (result.termination_reason != TMNF_TERMINATION_FINISH)
				fail("episode ended before the finish");
			finish_ms = result.race_time_ms;
			/* Autoreset already replaced the world state. Compare the
			 * final observation's position against the replay path. */
			const CHmsStateDyna *dyna = &lap->replay_states[tick - 1].dyna;
			if (memcmp(&result.final_observation.position, &dyna->pos,
					sizeof(dyna->pos)) != 0) {
				fail("final observation position differs from replay");
			}
			break;
		}
		PathState state;
		read_state(owner, &state);
		if (!states_match(&lap->replay_states[tick - 1], &state, analog)) {
			report_divergence(
				label, tick, &lap->replay_states[tick - 1], &state);
		}
	}
	TmnfVecEnv_Destroy(&env);
	World_Destroy(owner);
	if (finish_ms != lap->expected_finish_ms) {
		fprintf(stderr,
			"vec env matches replay: %s finished at %u ms, game %u ms\n",
			label, finish_ms, lap->expected_finish_ms);
		exit(1);
	}
	printf("vec env matches replay: %s, %u ticks %s, finish %u ms\n",
		label, finish_ms / TICK_MS,
		analog ? "identical up to signed zeros" : "byte-identical",
		finish_ms);
}

/*
 * A snapshot captured mid-drive and restored into an instance that has never
 * stepped must continue in lockstep with its source: the restore has to carry
 * every per-tick field, including the ones the spawn tick leaves in their
 * airborne state (CHmsDynaParams.forceFieldScale, dragLinear).
 */
static void run_fresh_restore_lockstep(const Lap *lap, int analog)
{
	TmnfWorld *owners[2];
	TmnfPhysicsWorld *physics[2];
	uint32_t player_indices[2] = { 0, 0 };
	TmnfVecEnv envs[2];
	TmnfVecEnvConfig config = env_config(lap, analog);
	for (uint32_t i = 0; i < 2; ++i) {
		owners[i] = World_Create(lap->track, lap->vehicle_path);
		physics[i] = World_GetPhysicsWorld(owners[i]);
		TmnfVecEnv_Init(
			&envs[i], &physics[i], &player_indices[i], 1, lap->route,
			&config);
	}
	drive_warm(&envs[0], analog, WARM_TICKS);
	TmnfEnvSnapshot snapshot;
	TmnfVecEnv_Capture(&envs[0], &snapshot);
	TmnfVecEnv_Restore(&envs[1], &snapshot);

	uint32_t tick_count = lap->tick_count < FRESH_RESTORE_TICKS
		? lap->tick_count : FRESH_RESTORE_TICKS;
	for (uint32_t tick = 0; tick < tick_count; ++tick) {
		PathState states[2];
		for (uint32_t i = 0; i < 2; ++i) {
			TmnfStepResult result;
			step_env(&envs[i], &lap->inputs[tick], analog, 1, &result);
			if (result.executed_ticks != 1 || result.terminated ||
				result.truncated) {
				fail("fresh-restore lockstep ended unexpectedly");
			}
			read_state(owners[i], &states[i]);
		}
		if (!states_match(&states[0], &states[1], analog)) {
			report_divergence(
				analog ? "analog fresh-instance restore"
				       : "discrete fresh-instance restore",
				tick + 1, &states[0], &states[1]);
		}
	}
	for (uint32_t i = 0; i < 2; ++i) {
		TmnfVecEnv_Destroy(&envs[i]);
		World_Destroy(owners[i]);
	}
	printf("vec env matches replay: %s fresh-instance restore, %u ticks "
		"in lockstep with the source\n",
		analog ? "analog" : "discrete", tick_count);
}

int main(int argc, char **argv)
{
	if (argc != 7) {
		fprintf(stderr,
			"usage: %s TRACK VEHICLE ROUTE INPUTS TRACK_SHA256 FINISH_MS\n",
			argv[0]);
		return 2;
	}
	uint8_t sha256[32];
	parse_sha256(argv[5], sha256);
	char *end;
	unsigned long finish = strtoul(argv[6], &end, 10);
	if (end == argv[6] || *end != '\0' || finish == 0 || finish % TICK_MS != 0)
		fail("expected finish must be a positive multiple of 10 ms");

	Lap lap = {
		.vehicle_path = argv[2],
		.expected_finish_ms = (uint32_t)finish,
	};
	uint32_t tick_count = 0;
	TMNFRaceInputs *inputs = read_inputs(argv[4], &tick_count);
	lap.inputs = inputs;
	lap.tick_count = tick_count;
	lap.track = TmnfTrack_Load(argv[1], sha256);
	lap.route = TmnfRoute_Load(argv[3], sha256);

	PathState *replay_states = calloc(tick_count, sizeof(*replay_states));
	if (replay_states == NULL)
		fail("out of memory");
	TmnfWorld *replay_world = World_Create(lap.track, argv[2]);
	run_replay_path(replay_world, lap.route, inputs, tick_count, replay_states);
	World_Destroy(replay_world);
	lap.replay_states = replay_states;

	int analog = !schedule_is_digital(inputs, tick_count);
	int respawns = schedule_respawns(inputs, tick_count);
	for (EnvRegime regime = 0; regime < REGIME_COUNT; ++regime) {
		run_env_path(&lap, analog, 1, regime);
		if (!analog && !respawns)
			run_env_path(&lap, analog, 5, regime);
	}
	run_fresh_restore_lockstep(&lap, analog);

	free(replay_states);
	TmnfRoute_Unload(lap.route);
	TmnfTrack_Unload(lap.track);
	free(inputs);
	return 0;
}
