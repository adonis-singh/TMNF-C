#include <math.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "race.h"
#include "world.h"
#if defined(TMNF_REPLAY_CUDA)
#include "cuda/tmnf_cuda_env.h"
#endif

enum {
	TICK_MS = 10,
	RECORD_SIZE = 1668,
	CAR_SIZE = 0x878,
	WHEEL_SIZE = 0x2fc,
};

/*
 * Race bookkeeping for schedules that respawn (TMNFRaceInputs.respawn): the
 * spawn a press applies is the last accepted checkpoint's, so the race layer
 * has to follow the run. Enabled by the leading `--route ROUTE` option; a
 * schedule that respawns without one is refused.
 */
typedef struct {
	TmnfRoute *route;
	TmnfRaceState race;
} RaceTracker;

static void fail(const char *message);

static void tracker_open(
	RaceTracker *tracker, const char *path, const uint8_t sha256[32],
	TmnfWorld *world)
{
	tracker->route = TmnfRoute_Load(path, sha256);
	TmnfPhysicsWorld *physics = World_GetPhysicsWorld(world);
	physics->route = tracker->route;
	TmnfRace_Reset(tracker->route, &tracker->race,
		physics->corpora[0].collision_corpus->live_iso);
}

/* 0x0047DCD0 CTrackManiaRace::OnInputEvent -> 0x00472700 SmallRespawn: the
 * press is handled before the tick's control mapping and physics step. */
static void tracker_respawn(
	RaceTracker *tracker, TmnfWorld *world, const TMNFRaceInputs *input)
{
	if (input->respawn == 0)
		return;
	if (tracker->route == NULL)
		fail("input schedule respawns; pass --route ROUTE");
	const GmIso4 *spawn = TmnfRace_RespawnLocation(&tracker->race);
	if (spawn == NULL)
		fail("respawn before any checkpoint restarts the race");
	World_Respawn(world, spawn);
}

static void tracker_step(RaceTracker *tracker, TmnfWorld *world)
{
	if (tracker->route == NULL || tracker->race.finished)
		return;
	const TmnfPhysicsWorld *physics = World_GetPhysicsWorld(world);
	(void)TmnfRace_Step(tracker->route, &tracker->race,
		physics->trigger_contacts,
		physics->corpora[0].collision_corpus->live_iso);
}

static void tracker_close(RaceTracker *tracker)
{
	if (tracker->route != NULL)
		TmnfRoute_Unload(tracker->route);
	tracker->route = NULL;
}

static const uint8_t A01_SHA256[32] = {
	0xf0, 0xa8, 0x70, 0x80, 0x9b, 0xe9, 0x9d, 0xa2,
	0xcb, 0x36, 0xad, 0x5d, 0xf4, 0x3a, 0x2c, 0xf6,
	0x3d, 0x8f, 0x74, 0xfe, 0x4a, 0xc3, 0x47, 0x0e,
	0xca, 0xc6, 0x8b, 0x9e, 0x97, 0x62, 0x5d, 0xc3,
};

static void fail(const char *message);

static uint8_t hex_nibble(char character)
{
	if (character >= '0' && character <= '9')
		return (uint8_t)(character - '0');
	character = (char)tolower((unsigned char)character);
	if (character >= 'a' && character <= 'f')
		return (uint8_t)(character - 'a' + 10);
	fail("track SHA-256 is not hexadecimal");
	return 0;
}

static void parse_sha256(const char *text, uint8_t output[32])
{
	if (strlen(text) != 64)
		fail("track SHA-256 must contain 64 hexadecimal characters");
	for (uint32_t i = 0; i < 32; ++i) {
		output[i] = (uint8_t)(
			hex_nibble(text[i * 2]) << 4 |
			hex_nibble(text[i * 2 + 1]));
	}
}

typedef struct {
	const char *name;
	uint32_t size;
} Field;

static const Field FIELDS[] = {
	{ "race_time", 4 },
	{ "dyna.quat", 16 },
	{ "dyna.rotation", 36 },
	{ "dyna.position", 12 },
	{ "dyna.linear_speed", 12 },
	{ "dyna.add_linear_speed", 12 },
	{ "dyna.angular_speed", 12 },
	{ "dyna.force", 12 },
	{ "dyna.torque", 12 },
	{ "dyna.inverse_inertia_tensor", 36 },
	{ "dyna.not_tweaked_linear_speed", 12 },
	{ "scene_mobil.physics", 180 },
	{ "wheel[0].physics", 328 },
	{ "wheel[1].physics", 328 },
	{ "wheel[2].physics", 328 },
	{ "wheel[3].physics", 328 },
};

typedef struct {
	uint8_t bytes[RECORD_SIZE];
	uint32_t offset;
} Encoder;

static void fail(const char *message)
{
	fprintf(stderr, "replay_tick: %s\n", message);
	exit(2);
}

static void append(Encoder *encoder, const void *data, uint32_t size)
{
	if (size > RECORD_SIZE - encoder->offset)
		fail("record encoder overflow");
	memcpy(encoder->bytes + encoder->offset, data, size);
	encoder->offset += size;
}

static void append_i32(Encoder *encoder, int32_t value)
{
	append(encoder, &value, sizeof(value));
}

static void encode_scene(Encoder *encoder, const uint8_t *car)
{
	append(encoder, car + 0x050, 12);
	append(encoder, car + 0x2e0, 4);
	append(encoder, car + 0x2e4, 8);
	append(encoder, car + 0x59c, 4);
	append(encoder, car + 0x5b0, 16);
	append(encoder, car + 0x5c4, 8);
	append(encoder, car + 0x5dc, 12);
	append(encoder, car + 0x5e8, 4);
	append(encoder, car + 0x5f4, 4);
	append(encoder, car + 0x5f8, 12);
	append(encoder, car + 0x608, 8);
	append(encoder, car + 0x628, 4);
	append(encoder, car + 0x67c, 4);
	append(encoder, car + 0x69c, 4);
	append(encoder, car + 0x70c, 12);
	append(encoder, car + 0x818, 12);
	append(encoder, car + 0x844, 4);
	append(encoder, car + 0x848, 48);
}

static void encode_wheel(Encoder *encoder, const uint8_t *wheel)
{
	append(encoder, wheel + 0x004, 8);
	append(encoder, wheel + 0x010, 48);
	append(encoder, wheel + 0x040, 36);
	append(encoder, wheel + 0x064, 12);
	append(encoder, wheel + 0x070, 48);
	append(encoder, wheel + 0x0a0, 8);
	append(encoder, wheel + 0x0a8, 12);
	append(encoder, wheel + 0x0b4, 12);
	append(encoder, wheel + 0x0c0, 36);
	append(encoder, wheel + 0x0e4, 36);
	append(encoder, wheel + 0x108, 12);
	append(encoder, wheel + 0x120, 4);
	append(encoder, wheel + 0x124, 12);
	append(encoder, wheel + 0x130, 12);
	append(encoder, wheel + 0x140, 4);
	append(encoder, wheel + 0x144, 12);
	append(encoder, wheel + 0x15c, 4);
	append(encoder, wheel + 0x160, 12);
}

static void encode_tick(
	TmnfWorld *world, uint32_t tick, uint8_t output[RECORD_SIZE])
{
	uint8_t car[CAR_SIZE];
	uint8_t wheels[4 * WHEEL_SIZE];
	World_WritePlayerGameState(world, car, wheels);

	Encoder encoder = { { 0 }, 0 };
	append_i32(&encoder, (int32_t)((tick + 1) * TICK_MS));
	const CHmsStateDyna *state = World_GetPlayerState(world);
	append(&encoder, &state->quat, 16);
	append(&encoder, &state->rot, 36);
	append(&encoder, &state->pos, 12);
	append(&encoder, &state->linVel, 12);
	append(&encoder, &state->linVelAdded, 12);
	append(&encoder, &state->angVel, 12);
	append(&encoder, &state->force, 12);
	append(&encoder, &state->torque, 12);
	append(&encoder, &state->invInertiaWorld, 36);
	append(&encoder, &state->tail[1], 12);
	encode_scene(&encoder, car);
	for (uint32_t i = 0; i < 4; ++i)
		encode_wheel(&encoder, wheels + i * WHEEL_SIZE);
	if (encoder.offset != RECORD_SIZE)
		fail("record encoder produced the wrong size");
	memcpy(output, encoder.bytes, RECORD_SIZE);
}

/* Each reference was captured under its own input schedule, so the schedule is
 * part of the reference and must be selected with it. Feeding run1's schedule
 * to the long drive diverges at tick 100 purely because run1 steers there. */
typedef enum {
	SCHEDULE_RUN1 = 0,
	SCHEDULE_LONG_DRIVE = 1,
	SCHEDULE_INPUT_FILE = 2,
} InputSchedule;

typedef struct {
	InputSchedule kind;
	TMNFRaceInputs *file_inputs;
	uint32_t file_tick_count;
} InputSource;

/* oracle/determinism_test.py */
static TMNFRaceInputs schedule_run1(uint32_t tick)
{
	TMNFRaceInputs input = { 0 };
	input.accelerate = tick < 800 || tick >= 900;
	input.brake = (600 <= tick && tick < 650) ||
		(800 <= tick && tick < 900);
	input.steer_left = 100 <= tick && tick < 300;
	input.steer_right = 300 <= tick && tick < 500;
	return input;
}

/* oracle/capture_long_drive.py */
static TMNFRaceInputs schedule_long_drive(uint32_t tick)
{
	TMNFRaceInputs input = { 0 };
	input.accelerate = tick < 250;
	input.brake = 250 <= tick && tick < 300;
	return input;
}

static TMNFRaceInputs scheduled_input(
	const InputSource *source, uint32_t tick)
{
	switch (source->kind) {
	case SCHEDULE_RUN1:
		return schedule_run1(tick);
	case SCHEDULE_LONG_DRIVE:
		return schedule_long_drive(tick);
	case SCHEDULE_INPUT_FILE:
		if (tick >= source->file_tick_count)
			fail("input schedule is shorter than the reference");
		return source->file_inputs[tick];
	}
	fail("unknown input schedule");
	return (TMNFRaceInputs){ 0 };
}

static void report_mismatch(
	uint32_t tick, const uint8_t *expected, const uint8_t *actual)
{
	uint32_t byte = 0;
	while (byte < RECORD_SIZE && expected[byte] == actual[byte])
		++byte;
	uint32_t field_start = 0;
	const Field *field = NULL;
	for (uint32_t i = 0; i < sizeof(FIELDS) / sizeof(FIELDS[0]); ++i) {
		if (byte < field_start + FIELDS[i].size) {
			field = &FIELDS[i];
			break;
		}
		field_start += FIELDS[i].size;
	}
	if (field == NULL)
		fail("mismatch lies outside field table");
	uint32_t local = byte - field_start;
	uint32_t word = local & ~3u;
	uint32_t expected_bits;
	uint32_t actual_bits;
	float expected_float;
	float actual_float;
	memcpy(&expected_bits, expected + field_start + word, 4);
	memcpy(&actual_bits, actual + field_start + word, 4);
	memcpy(&expected_float, &expected_bits, 4);
	memcpy(&actual_float, &actual_bits, 4);
	fprintf(stderr,
		"first divergence: tick %u (race time %u ms), field %s, "
		"byte %u, word %u\n"
		"  expected: 0x%08x %a\n"
		"  actual:   0x%08x %a\n",
		tick, (tick + 1) * TICK_MS, field->name, local, word,
		expected_bits, expected_float, actual_bits, actual_float);
	if (getenv("TMNF_REPLAY_VERBOSE") != NULL) {
		uint32_t start = 0;
		for (uint32_t i = 0;
		     i < sizeof(FIELDS) / sizeof(FIELDS[0]); ++i) {
			if (memcmp(expected + start, actual + start,
					FIELDS[i].size) != 0) {
				uint32_t at = 0;
				while (expected[start + at] == actual[start + at])
					++at;
				uint32_t aligned = at & ~3u;
				memcpy(&expected_bits,
					expected + start + aligned, 4);
				memcpy(&actual_bits, actual + start + aligned, 4);
				fprintf(stderr,
					"  mismatch %-35s +%u "
					"%08x != %08x\n",
					FIELDS[i].name, aligned,
					expected_bits, actual_bits);
			}
			start += FIELDS[i].size;
		}
	}
}

static uint8_t *read_reference(const char *path, uint32_t *tick_count)
{
	FILE *file = fopen(path, "rb");
	if (file == NULL)
		fail("cannot open run reference");
	if (fseek(file, 0, SEEK_END) != 0)
		fail("cannot seek run reference");
	long size = ftell(file);
	if (size <= 0 || size % RECORD_SIZE != 0)
		fail("run reference is not a whole number of records");
	rewind(file);
	uint32_t count = (uint32_t)(size / RECORD_SIZE);
	uint8_t *bytes = malloc((size_t)count * RECORD_SIZE);
	if (bytes == NULL)
		fail("out of memory reading run reference");
	if (fread(bytes, RECORD_SIZE, count, file) != count)
		fail("run reference is truncated");
	if (fclose(file) != 0)
		fail("cannot close run reference");
	*tick_count = count;
	return bytes;
}

static TMNFRaceInputs *read_input_file(
	const char *path, uint32_t *tick_count)
{
	FILE *file = fopen(path, "rb");
	if (file == NULL)
		fail("cannot open input schedule");
	if (fseek(file, 0, SEEK_END) != 0)
		fail("cannot seek input schedule");
	long size = ftell(file);
	if (size <= 0 || size % sizeof(TMNFRaceInputs) != 0)
		fail("input schedule is not a whole number of records");
	uint32_t count = (uint32_t)(size / sizeof(TMNFRaceInputs));
	if (*tick_count != 0 && count != *tick_count)
		fail("input schedule size differs from run reference");
	rewind(file);
	TMNFRaceInputs *inputs =
		malloc((size_t)count * sizeof(*inputs));
	if (inputs == NULL)
		fail("out of memory reading input schedule");
	if (fread(inputs, sizeof(*inputs), count, file) != count) {
		fail("input schedule is truncated");
	}
	if (fclose(file) != 0)
		fail("cannot close input schedule");
	*tick_count = count;
	return inputs;
}

static void report_position(
	uint32_t tick, const uint8_t *expected, const CHmsStateDyna *actual)
{
	GmVec3 game;
	memcpy(&game, expected + 56, sizeof(game));
	double dx = (double)actual->pos.x - game.x;
	double dy = (double)actual->pos.y - game.y;
	double dz = (double)actual->pos.z - game.z;
	double drift = sqrt(dx * dx + dy * dy + dz * dz);
	printf(
		"position tick %u: game=(%.9g, %.9g, %.9g) "
		"native=(%.9g, %.9g, %.9g) drift=%.9g\n",
		tick + 1,
		game.x, game.y, game.z,
		actual->pos.x, actual->pos.y, actual->pos.z,
		drift);
}

static uint32_t float_bits(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return bits;
}

static void report_collisions(
	const TmnfWorld *world, uint32_t tick_number)
{
	const CFastBuffer_SHmsPhysicalCollision *buffer =
		World_GetLastCollisions(world);
	const CHmsReplacementBuf *replacements =
		World_GetPlayerReplacements(world);
	printf("tick %u collisions: %u\n", tick_number, buffer->count);
	for (uint32_t i = 0; i < buffer->count; ++i) {
		const SHmsPhysicalCollision *physical = &buffer->data[i];
		const GmCollision *collision = &physical->collision;
		printf(
			"  %u: tree=%u material=%u/%u flags=%u "
			"separation=%08x,%08x,%08x "
			"normal=%08x,%08x,%08x "
			"position=%08x,%08x,%08x "
			"face_normal=%08x,%08x,%08x\n",
			i, physical->tree1,
			collision->material1, collision->material2,
			collision->flags,
			float_bits(collision->separation.x),
			float_bits(collision->separation.y),
			float_bits(collision->separation.z),
			float_bits(collision->normal.x),
			float_bits(collision->normal.y),
			float_bits(collision->normal.z),
			float_bits(collision->position.x),
			float_bits(collision->position.y),
			float_bits(collision->position.z),
			float_bits(collision->face_normal.x),
			float_bits(collision->face_normal.y),
			float_bits(collision->face_normal.z));
	}
	printf("tick %u replacements: %u\n",
		tick_number, replacements->count);
	for (uint32_t i = 0; i < replacements->count; ++i) {
		printf("  %u: %08x,%08x,%08x\n", i,
			float_bits(replacements->data[i].x),
			float_bits(replacements->data[i].y),
			float_bits(replacements->data[i].z));
	}
}

static int capture_native(int argc, char **argv, const char *route_path)
{
	if (argc != 6 && argc != 7)
		fail("native capture requires TRACK VEHICLE INPUTS OUTPUT [TRACK_SHA256]");
	uint8_t expected_track_sha256[32];
	memcpy(expected_track_sha256, A01_SHA256, sizeof(expected_track_sha256));
	if (argc == 7)
		parse_sha256(argv[6], expected_track_sha256);
	uint32_t tick_count = 0;
	InputSource source = {
		.kind = SCHEDULE_INPUT_FILE,
	};
	source.file_inputs = read_input_file(argv[4], &tick_count);
	source.file_tick_count = tick_count;
	TmnfTrack *track = TmnfTrack_Load(argv[2], expected_track_sha256);
	TmnfWorld *world = World_Create(track, argv[3]);
	RaceTracker tracker = { 0 };
	if (route_path != NULL)
		tracker_open(&tracker, route_path, expected_track_sha256, world);
	FILE *output = fopen(argv[5], "wb");
	if (output == NULL)
		fail("cannot open native capture output");
	uint8_t record[RECORD_SIZE];
	for (uint32_t tick = 0; tick < tick_count; ++tick) {
		TMNFRaceInputs input = scheduled_input(&source, tick);
		tracker_respawn(&tracker, world, &input);
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
			&input, World_GetPlayerVehicle(world));
		encode_tick(world, tick, record);
		if (fwrite(record, RECORD_SIZE, 1, output) != 1)
			fail("cannot write native capture output");
		World_AdvanceTimer(world, TICK_MS);
		CHmsZoneDynamic_PhysicsStep2(
			World_GetPhysicsWorld(world), TICK_MS);
		tracker_step(&tracker, world);
	}
	if (fclose(output) != 0)
		fail("cannot close native capture output");
	printf("native capture: %u ticks, %u bytes\n",
		tick_count, tick_count * RECORD_SIZE);
	tracker_close(&tracker);
	World_Destroy(world);
	TmnfTrack_Unload(track);
	free(source.file_inputs);
	return 0;
}

/* --cuda: the world lives on the GPU. Every tick the harness uploads the
 * input, has the device apply it (respawn included, from the device race
 * state), copies the world back to encode the record the game exposes
 * between input and step, then steps on the device. The host world only
 * lends its allocations to the copy. */
typedef struct {
	int cuda;
	RaceTracker tracker;
#if defined(TMNF_REPLAY_CUDA)
	TmnfCudaVecEnv *env;
#endif
} Stepper;

static void stepper_init(Stepper *stepper, const TmnfTrack *track,
	TmnfWorld *world, const char *route_path, const uint8_t sha256[32])
{
	if (!stepper->cuda) {
		if (route_path != NULL)
			tracker_open(&stepper->tracker, route_path, sha256, world);
		return;
	}
#if defined(TMNF_REPLAY_CUDA)
	TmnfRoute *route = route_path != NULL
		? TmnfRoute_Load(route_path, sha256) : NULL;
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	TmnfCudaVecEnvLimits limits = TmnfCudaVecEnv_DefaultLimits();
	stepper->env = TmnfCudaVecEnv_Create(
		track, world, route, 1, &config, &limits);
	if (stepper->env == NULL)
		fail(TmnfCudaVecEnv_LastError());
	stepper->tracker.route = route;
#else
	(void)track;
	(void)world;
	(void)route_path;
	(void)sha256;
	fail("this harness was built without CUDA support");
#endif
}

/* Applies the input and leaves the world showing the pre-step state; the
 * step itself follows in stepper_step. */
static void stepper_apply(Stepper *stepper, TmnfWorld *world,
	const TMNFRaceInputs *input)
{
	if (!stepper->cuda) {
		tracker_respawn(&stepper->tracker, world, input);
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(
			input, World_GetPlayerVehicle(world));
		return;
	}
#if defined(TMNF_REPLAY_CUDA)
	if (input->respawn != 0 && stepper->tracker.route == NULL)
		fail("input schedule respawns; pass --route ROUTE");
	TmnfCudaVecEnv_StepWithPreStateCopy(
		stepper->env, input, TICK_MS, 0, world);
#endif
}

static void stepper_step(Stepper *stepper, TmnfWorld *world)
{
	if (stepper->cuda)
		return;
	World_AdvanceTimer(world, TICK_MS);
	CHmsZoneDynamic_PhysicsStep2(World_GetPhysicsWorld(world), TICK_MS);
	tracker_step(&stepper->tracker, world);
}

static void stepper_destroy(Stepper *stepper)
{
#if defined(TMNF_REPLAY_CUDA)
	if (stepper->cuda)
		TmnfCudaVecEnv_Destroy(stepper->env);
#endif
	tracker_close(&stepper->tracker);
}

int main(int argc, char **argv)
{
	Stepper stepper = { 0 };
	if (argc >= 2 && strcmp(argv[1], "--cuda") == 0) {
		stepper.cuda = 1;
		--argc;
		++argv;
	}
	const char *route_path = NULL;
	if (argc >= 3 && strcmp(argv[1], "--route") == 0) {
		route_path = argv[2];
		argv[2] = argv[0];
		argv += 2;
		argc -= 2;
	}
	if (argc >= 2 && strcmp(argv[1], "--native-capture") == 0)
		return capture_native(argc, argv, route_path);
	if (argc < 5 || argc > 7) {
		fprintf(stderr,
			"usage: %s [--route ROUTE] A01_TRACK A01_VEHICLE REFERENCE "
			"{run1|long_drive|input_file} [INPUTS] [TRACK_SHA256]\n"
			"       %s [--route ROUTE] --native-capture A01_TRACK "
			"A01_VEHICLE INPUTS OUTPUT [TRACK_SHA256]\n",
			argv[0],
			argv[0]);
		return 2;
	}
	InputSource source = { 0 };
	uint32_t hash_argument = 0;
	if (strcmp(argv[4], "run1") == 0 && (argc == 5 || argc == 6)) {
		source.kind = SCHEDULE_RUN1;
		hash_argument = argc == 6 ? 5 : 0;
	} else if (strcmp(argv[4], "long_drive") == 0 &&
		   (argc == 5 || argc == 6)) {
		source.kind = SCHEDULE_LONG_DRIVE;
		hash_argument = argc == 6 ? 5 : 0;
	} else if (strcmp(argv[4], "input_file") == 0 &&
		   (argc == 6 || argc == 7)) {
		source.kind = SCHEDULE_INPUT_FILE;
		hash_argument = argc == 7 ? 6 : 0;
	} else {
		fail("unknown schedule name");
	}
	uint8_t expected_track_sha256[32];
	memcpy(expected_track_sha256, A01_SHA256, sizeof(expected_track_sha256));
	if (hash_argument != 0)
		parse_sha256(argv[hash_argument], expected_track_sha256);

	uint32_t tick_count = 0;
	uint32_t collision_report_tick = 0;
	uint32_t collision_ticks = 0;
	uint32_t collision_records = 0;
	uint32_t wall_like_records = 0;
	uint32_t maximum_collisions = 0;
	int collision_summary =
		getenv("TMNF_REPLAY_COLLISION_SUMMARY") != NULL;
	const char *collision_report_text =
		getenv("TMNF_REPLAY_COLLISION_TICK");
	if (collision_report_text != NULL) {
		char *end;
		unsigned long value =
			strtoul(collision_report_text, &end, 10);
		if (end == collision_report_text || *end != '\0' ||
			value == 0 || value > UINT32_MAX) {
			fail("invalid TMNF_REPLAY_COLLISION_TICK");
		}
		collision_report_tick = (uint32_t)value;
	}
	uint8_t *reference = read_reference(argv[3], &tick_count);
	if (source.kind == SCHEDULE_INPUT_FILE) {
		source.file_inputs =
			read_input_file(argv[5], &tick_count);
		source.file_tick_count = tick_count;
	}
	TmnfTrack *track =
		TmnfTrack_Load(argv[1], expected_track_sha256);
	TmnfWorld *world = World_Create(track, argv[2]);
	stepper_init(&stepper, track, world, route_path, expected_track_sha256);
	if (stepper.cuda && (collision_summary || collision_report_tick != 0))
		fail("collision reports read host buffers; not available with --cuda");
	uint8_t actual[RECORD_SIZE];
	uint32_t matched = 0;
	int diverged = 0;
	for (uint32_t tick = 0; tick < tick_count; ++tick) {
		TMNFRaceInputs input = scheduled_input(&source, tick);
		stepper_apply(&stepper, world, &input);
		encode_tick(world, tick, actual);
		const uint8_t *expected =
			reference + (size_t)tick * RECORD_SIZE;
		if (memcmp(actual, expected, RECORD_SIZE) != 0) {
			if (!diverged)
				report_mismatch(tick, expected, actual);
			diverged = 1;
		}
		if (!diverged)
			++matched;
		if (source.kind == SCHEDULE_INPUT_FILE &&
			(tick + 1 == 500 || tick + 1 == 1000 ||
			 tick + 1 == 1500 || tick + 1 == 2000 ||
			 tick + 1 == 2500)) {
			report_position(
				tick, expected, World_GetPlayerState(world));
		}
		stepper_step(&stepper, world);
		if (collision_summary) {
			const CFastBuffer_SHmsPhysicalCollision *collisions =
				World_GetLastCollisions(world);
			if (collisions->count != 0)
				++collision_ticks;
			collision_records += collisions->count;
			if (collisions->count > maximum_collisions)
				maximum_collisions = collisions->count;
			for (uint32_t i = 0; i < collisions->count; ++i) {
				if (fabsf(
					    collisions->data[i].collision.normal.y) <
				    0.5f) {
					++wall_like_records;
				}
			}
		}
		if (tick + 1 == collision_report_tick) {
			report_collisions(world, tick + 1);
		}
	}
	printf("%s tick: %u/%u ticks byte-exact\n",
		stepper.cuda ? "cuda" : "full", matched, tick_count);
	if (collision_summary) {
		printf(
			"collision summary: ticks=%u records=%u "
			"wall_like=%u max_per_tick=%u\n",
			collision_ticks, collision_records,
			wall_like_records, maximum_collisions);
	}
	stepper_destroy(&stepper);
	World_Destroy(world);
	TmnfTrack_Unload(track);
	free(source.file_inputs);
	free(reference);
	return matched == tick_count ? 0 : 1;
}
