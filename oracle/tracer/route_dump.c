#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define IMAGE_BASE 0x00400000u
#define PHYSICS_STEP2_VA 0x00549C90u
#define RACE_TIME_VA 0x0047E680u
#define RACE_TRIGGER_VTABLE_VA 0x00B3D0B0u
#define RACE_TRIGGER_ABSORB_VA 0x0047CBA0u
#define BLOCK_GET_SPAWN_LOC_VA 0x0060B410u

#define SECTION_COUNT 5u
#define SEC_METADATA 0u
#define SEC_START 1u
#define SEC_CHECKPOINTS 2u
#define SEC_FINISH 3u
#define SEC_REFERENCE 4u

#define WAYPOINT_START 0u
#define WAYPOINT_FINISH 1u
#define WAYPOINT_CHECKPOINT 2u
#define WAYPOINT_NONE 3u
#define WAYPOINT_START_FINISH 4u

#pragma pack(push, 8)
struct DiskSection {
	uint64_t offset;
	uint32_t count;
	uint32_t stride;
};

struct DiskHeader {
	char magic[8];
	uint32_t version;
	uint32_t endian;
	uint32_t header_size;
	uint32_t section_count;
	uint64_t file_size;
	uint8_t exe_sha256[32];
	uint8_t track_sha256[32];
	struct DiskSection sections[SECTION_COUNT];
	uint8_t payload_sha256[32];
	uint8_t reserved[16];
};

struct DiskMetadata {
	uint32_t lap_count;
	uint32_t checkpoint_count;
	uint32_t finish_count;
	uint32_t reference_count;
	uint32_t total_race_checkpoints;
	uint32_t race_checkpoint_limit;
	uint32_t flags;
	uint32_t reserved;
	uint64_t start_rel;
	uint64_t checkpoints_rel;
	uint64_t finish_rel;
	uint64_t reference_rel;
};

struct DiskStart {
	float transform[12];
	uint8_t initial_dyna_state[0xAC];
	uint32_t block_index;
	uint32_t waypoint_type;
	float spawn[12];
};

struct DiskTrigger {
	uint32_t race_index;
	uint32_t block_index;
	uint32_t waypoint_type;
	uint32_t tree_flags;
	float box[6];
	float transform[12];
	float spawn[12];
	uint32_t no_respawn;
	uint32_t reserved;
};

struct DiskReferencePoint {
	float position[3];
	float arc_length;
};
#pragma pack(pop)

_Static_assert(sizeof(struct DiskSection) == 0x10, "section size");
_Static_assert(sizeof(struct DiskHeader) == 0xE0, "header size");
_Static_assert(sizeof(struct DiskMetadata) == 0x40, "metadata size");
_Static_assert(sizeof(struct DiskStart) == 0x114, "start size");
_Static_assert(sizeof(struct DiskTrigger) == 0x90, "trigger size");
_Static_assert(sizeof(struct DiskReferencePoint) == 0x10, "reference size");

static const uint8_t EXE_SHA256[32] = {
	0x38, 0x47, 0xcf, 0x9f, 0x20, 0xbf, 0xc6, 0x39,
	0x14, 0x45, 0x00, 0x60, 0xed, 0x52, 0x8c, 0x12,
	0x10, 0x4f, 0x74, 0x3d, 0x96, 0xad, 0x23, 0xd6,
	0xe7, 0x6a, 0xbd, 0x17, 0x8d, 0xe8, 0xc8, 0x4f,
};

struct Sha256 {
	uint32_t state[8];
	uint64_t byte_count;
	uint8_t block[64];
	uint32_t block_len;
};

static const uint32_t SHA256_K[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
	0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
	0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
	0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
	0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
	0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

typedef uint32_t (__attribute__((thiscall)) *RaceTimeFn)(void *race);
/* CGameCtnBlock::GetSpawnLoc(GmIso4 &out, ulong, ulong): the respawn
 * isometry of a waypoint block, exactly as CTrackManiaRace::OnCheckpoint
 * (0x0047C330) requests it with both extra arguments zero. */
typedef void (__attribute__((thiscall)) *GetSpawnLocFn)(
	void *block, float *out, uint32_t a, uint32_t b);

static HANDLE g_heap;
static HANDLE g_log = INVALID_HANDLE_VALUE;
static char g_output_path[MAX_PATH];
static uint32_t g_module_base;
static uint8_t *g_trampoline;
static uint8_t g_track_sha256[32];
static volatile LONG g_dump_started;
static volatile LONG g_probe_logged;

static void fatal(const char *message);

static void read_runtime(uint32_t address, void *output, uint32_t size)
{
	SIZE_T read = 0;
	char message[160];
	if (address == 0
	    || !ReadProcessMemory(
		    GetCurrentProcess(), (const void *)(uintptr_t)address,
		    output, size, &read)
	    || read != size) {
		_snprintf(
			message, sizeof(message),
			"cannot read runtime address %08X size=%u", address, size);
		fatal(message);
	}
}

static uint32_t read_u32(uint32_t address)
{
	uint32_t value;
	read_runtime(address, &value, sizeof(value));
	return value;
}

static void write_all(HANDLE file, const void *data, DWORD len)
{
	const uint8_t *cursor = (const uint8_t *)data;
	while (len != 0) {
		DWORD written = 0;
		if (!WriteFile(file, cursor, len, &written, NULL) || written == 0)
			TerminateProcess(GetCurrentProcess(), 140);
		cursor += written;
		len -= written;
	}
}

static void log_line(const char *text)
{
	if (g_log != INVALID_HANDLE_VALUE) {
		write_all(g_log, text, (DWORD)strlen(text));
		write_all(g_log, "\r\n", 2);
		FlushFileBuffers(g_log);
	}
	OutputDebugStringA(text);
}

static void fatal(const char *message)
{
	log_line(message);
	TerminateProcess(GetCurrentProcess(), 141);
}

static uint32_t rotr32(uint32_t value, uint32_t bits)
{
	return (value >> bits) | (value << (32 - bits));
}

static uint32_t load_be32(const uint8_t *p)
{
	return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16
		| (uint32_t)p[2] << 8 | p[3];
}

static void store_be32(uint8_t *p, uint32_t value)
{
	p[0] = (uint8_t)(value >> 24);
	p[1] = (uint8_t)(value >> 16);
	p[2] = (uint8_t)(value >> 8);
	p[3] = (uint8_t)value;
}

static void sha256_transform(struct Sha256 *ctx, const uint8_t block[64])
{
	uint32_t w[64];
	uint32_t a, b, c, d, e, f, g, h;
	uint32_t i;
	for (i = 0; i < 16; ++i)
		w[i] = load_be32(block + i * 4);
	for (i = 16; i < 64; ++i) {
		uint32_t s0 = rotr32(w[i - 15], 7)
			^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
		uint32_t s1 = rotr32(w[i - 2], 17)
			^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];
	for (i = 0; i < 64; ++i) {
		uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
		uint32_t choose = (e & f) ^ (~e & g);
		uint32_t t1 = h + s1 + choose + SHA256_K[i] + w[i];
		uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
		uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
		uint32_t t2 = s0 + majority;
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}
	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
}

static void sha256_init(struct Sha256 *ctx)
{
	memset(ctx, 0, sizeof(*ctx));
	ctx->state[0] = 0x6a09e667;
	ctx->state[1] = 0xbb67ae85;
	ctx->state[2] = 0x3c6ef372;
	ctx->state[3] = 0xa54ff53a;
	ctx->state[4] = 0x510e527f;
	ctx->state[5] = 0x9b05688c;
	ctx->state[6] = 0x1f83d9ab;
	ctx->state[7] = 0x5be0cd19;
}

static void sha256_update(struct Sha256 *ctx, const void *data, uint32_t len)
{
	const uint8_t *bytes = (const uint8_t *)data;
	ctx->byte_count += len;
	while (len != 0) {
		uint32_t space = 64 - ctx->block_len;
		uint32_t take = len < space ? len : space;
		memcpy(ctx->block + ctx->block_len, bytes, take);
		ctx->block_len += take;
		bytes += take;
		len -= take;
		if (ctx->block_len == 64) {
			sha256_transform(ctx, ctx->block);
			ctx->block_len = 0;
		}
	}
}

static void sha256_final(struct Sha256 *ctx, uint8_t digest[32])
{
	uint64_t bit_count = ctx->byte_count * 8;
	uint32_t i;
	ctx->block[ctx->block_len++] = 0x80;
	if (ctx->block_len > 56) {
		memset(ctx->block + ctx->block_len, 0, 64 - ctx->block_len);
		sha256_transform(ctx, ctx->block);
		ctx->block_len = 0;
	}
	memset(ctx->block + ctx->block_len, 0, 56 - ctx->block_len);
	for (i = 0; i < 8; ++i)
		ctx->block[63 - i] = (uint8_t)(bit_count >> (i * 8));
	sha256_transform(ctx, ctx->block);
	for (i = 0; i < 8; ++i)
		store_be32(digest + i * 4, ctx->state[i]);
}

static uint32_t align8(uint32_t value)
{
	if (value > UINT32_MAX - 7)
		fatal("route alignment overflow");
	return (value + 7) & ~7u;
}

static void validate_buffer(uint32_t buffer, uint32_t stride)
{
	uint32_t count = read_u32(buffer);
	uint32_t data = read_u32(buffer + 4);
	uint32_t capacity = read_u32(buffer + 8);
	if (count > capacity || (count != 0 && data == 0)
	    || (count != 0 && count > UINT32_MAX / stride))
		fatal("invalid runtime route buffer");
}

static uint32_t buffer_pointer(uint32_t buffer, uint32_t index)
{
	uint32_t count;
	validate_buffer(buffer, 4);
	count = read_u32(buffer);
	if (index >= count)
		fatal("runtime route buffer index out of range");
	return read_u32(read_u32(buffer + 4) + index * 4);
}

static uint32_t find_race(void)
{
	uint32_t root = read_u32(
		g_module_base + 0x00D68C44u - IMAGE_BASE);
	uint32_t race;
	if (root == 0)
		return 0;
	race = read_u32(root + 0x454);
	if (race == 0)
		return 0;
	if (read_u32(race + 0x18) != root)
		fatal("module root does not point to CTrackManiaRace");
	return race;
}

static uint32_t block_waypoint_type(uint32_t block)
{
	uint32_t info = read_u32(block + 0x24);
	uint32_t type;
	if (info == 0)
		fatal("route block has no block info");
	type = read_u32(info + 0x11C);
	if (type > WAYPOINT_START_FINISH)
		fatal("invalid route waypoint type");
	return type;
}

static uint32_t find_block_index(
	uint32_t challenge, uint32_t scene_mobil, uint32_t expected_type)
{
	uint32_t blocks = challenge + 0x54;
	uint32_t count;
	uint32_t i;
	validate_buffer(blocks, 4);
	count = read_u32(blocks);
	for (i = 0; i < count; ++i) {
		uint32_t block = buffer_pointer(blocks, i);
		if (block != 0 && read_u32(block + 0x34) == scene_mobil) {
			uint32_t actual_type = block_waypoint_type(block);
			if (actual_type != expected_type
			    && !(expected_type == WAYPOINT_FINISH
			         && actual_type == WAYPOINT_START_FINISH))
				fatal("route mobil has unexpected waypoint type");
			return i;
		}
	}
	fatal("route mobil does not map to a challenge block");
	return 0;
}

static uint32_t find_start_block(uint32_t challenge)
{
	uint32_t blocks = challenge + 0x54;
	uint32_t count;
	uint32_t found = UINT32_MAX;
	uint32_t i;
	validate_buffer(blocks, 4);
	count = read_u32(blocks);
	for (i = 0; i < count; ++i) {
		uint32_t block = buffer_pointer(blocks, i);
		uint32_t type =
			block == 0 ? WAYPOINT_NONE : block_waypoint_type(block);
		if (type == WAYPOINT_START || type == WAYPOINT_START_FINISH) {
			if (found != UINT32_MAX)
				fatal("multiple start blocks");
			found = i;
		}
	}
	if (found == UINT32_MAX)
		fatal("start block not found");
	return found;
}

static void block_spawn(uint32_t block, float spawn[12])
{
	GetSpawnLocFn get_spawn_loc = (GetSpawnLocFn)(uintptr_t)(
		g_module_base + BLOCK_GET_SPAWN_LOC_VA - IMAGE_BASE);
	uint32_t i;
	memset(spawn, 0, 12 * sizeof(float));
	get_spawn_loc((void *)(uintptr_t)block, spawn, 0, 0);
	for (i = 0; i < 12; ++i) {
		if (!isfinite(spawn[i]))
			fatal("block spawn location is not finite");
	}
}

/* CGameCtnBlockInfo+0x120: when set, OnCheckpoint keeps the previous spawn
 * location instead of this block's. */
static uint32_t block_no_respawn(uint32_t block)
{
	uint32_t info = read_u32(block + 0x24);
	if (info == 0)
		fatal("route block has no block info");
	return read_u32(info + 0x120) != 0;
}

static uint32_t scene_corpus(uint32_t scene_mobil)
{
	uint32_t item = read_u32(scene_mobil + 0x28);
	uint32_t count;
	uint32_t corpus;
	if (item == 0)
		fatal("route mobil has no collision item");
	validate_buffer(item + 0x34, 4);
	count = read_u32(item + 0x34);
	if (count != 1)
		fatal("route mobil must have exactly one collision corpus");
	corpus = buffer_pointer(item + 0x34, 0);
	if (corpus == 0 || read_u32(corpus + 0x48) != item)
		fatal("route mobil collision corpus mismatch");
	return corpus;
}

static void extract_trigger(
	uint32_t challenge, uint32_t scene_mobil, uint32_t race_index,
	uint32_t waypoint_type, struct DiskTrigger *output)
{
	uint32_t item = read_u32(scene_mobil + 0x28);
	uint32_t corpus = scene_corpus(scene_mobil);
	uint32_t response_model = read_u32(item + 0x14);
	uint32_t tree;
	if (response_model == 0)
		fatal("route mobil has no collision response model");
	tree = read_u32(response_model + 0x64);
	if (tree == 0)
		fatal("route mobil has no collision tree");
	memset(output, 0, sizeof(*output));
	output->race_index = race_index;
	output->block_index = find_block_index(
		challenge, scene_mobil, waypoint_type);
	output->waypoint_type = waypoint_type;
	block_spawn(
		buffer_pointer(challenge + 0x54, output->block_index),
		output->spawn);
	output->no_respawn = block_no_respawn(
		buffer_pointer(challenge + 0x54, output->block_index));
	output->tree_flags = read_u32(tree + 0x9C);
	/* A CPlugTree's box (+0x34) lives in its parent's frame: the collision
	 * manager tests it under the parent's world isometry and applies the
	 * tree's own Location (+0x5C, flag 4) only to its surface and children
	 * (src/race.c car_tree_contact mirrors that). The trigger the port
	 * builds from this box is therefore placed with the corpus isometry
	 * alone. Nonidentity locations occur in Desert and Rally gates: stale
	 * composed RallyA1 records displace its finish by 16 m on both horizontal
	 * axes. The box must never be transformed by the root's own Location. */
	read_runtime(tree + 0x34, output->box, sizeof(output->box));
	read_runtime(corpus + 0x18, output->transform, sizeof(output->transform));
}

static void trigger_center(
	const struct DiskTrigger *trigger, float output[3])
{
	const float *m = trigger->transform;
	const float *c = trigger->box;
	output[0] = c[0] * m[0] + c[1] * m[1] + c[2] * m[2] + m[9];
	output[1] = c[0] * m[3] + c[1] * m[4] + c[2] * m[5] + m[10];
	output[2] = c[0] * m[6] + c[1] * m[7] + c[2] * m[8] + m[11];
}

static uint32_t find_initial_dyna(uint32_t zone_dynamic)
{
	uint32_t buffer = zone_dynamic + 0x140;
	uint32_t count;
	uint32_t found = 0;
	uint32_t i;
	validate_buffer(buffer, 4);
	count = read_u32(buffer);
	for (i = 0; i < count; ++i) {
		uint32_t corpus = buffer_pointer(buffer, i);
		uint32_t dyna;
		if (corpus == 0)
			fatal("null dynamic corpus");
		dyna = read_u32(corpus + 0x58);
		if (dyna == 0)
			continue;
		if (read_u32(dyna + 0x32C) == 0)
			fatal("dynamic corpus has no live state");
		if (found != 0 && found != dyna)
			fatal("multiple dynamic rigid bodies at race start");
		found = dyna;
	}
	return found;
}

static void write_snapshot(
	uint32_t race, uint32_t challenge, uint32_t dyna)
{
	static const uint32_t strides[SECTION_COUNT] = {
		sizeof(struct DiskMetadata),
		sizeof(struct DiskStart),
		sizeof(struct DiskTrigger),
		sizeof(struct DiskTrigger),
		sizeof(struct DiskReferencePoint),
	};
	struct DiskHeader header;
	struct DiskMetadata metadata;
	struct DiskStart start;
	struct DiskTrigger *checkpoints;
	struct DiskTrigger *finishes;
	struct DiskReferencePoint *reference;
	uint8_t *file;
	struct Sha256 sha;
	HANDLE handle;
	uint32_t checkpoint_count;
	uint32_t finish_count;
	uint32_t start_block;
	uint32_t state;
	uint32_t offset;
	uint32_t i;
	char line[320];

	validate_buffer(race + 0xD4, 4);
	validate_buffer(race + 0xE0, 4);
	checkpoint_count = read_u32(race + 0xD4);
	finish_count = read_u32(race + 0xE0);
	/* Several finish blocks are legal (A12-Speed has four); every one is
	 * recorded and tools/generate_route_centerline.py keeps the one the
	 * official ghost crosses. */
	if (finish_count == 0)
		fatal("race has no finish");
	if (finish_count > 64)
		fatal("race has an implausible finish count");
	if (checkpoint_count > (UINT32_MAX - 2) / sizeof(*reference))
		fatal("route reference count overflow");

	checkpoints = NULL;
	if (checkpoint_count != 0) {
		checkpoints = (struct DiskTrigger *)HeapAlloc(
			g_heap, HEAP_ZERO_MEMORY,
			checkpoint_count * sizeof(*checkpoints));
	}
	finishes = (struct DiskTrigger *)HeapAlloc(
		g_heap, HEAP_ZERO_MEMORY, finish_count * sizeof(*finishes));
	reference = (struct DiskReferencePoint *)HeapAlloc(
		g_heap, HEAP_ZERO_MEMORY,
		(checkpoint_count + 2) * sizeof(*reference));
	if ((checkpoint_count != 0 && checkpoints == NULL) ||
	    finishes == NULL || reference == NULL)
		fatal("route snapshot allocation failed");

	start_block = find_start_block(challenge);
	memset(&start, 0, sizeof(start));
	state = read_u32(dyna + 0x32C);
	read_runtime(state, start.initial_dyna_state, sizeof(start.initial_dyna_state));
	memcpy(start.transform, start.initial_dyna_state + 0x10, 0x24);
	memcpy(start.transform + 9, start.initial_dyna_state + 0x34, 0x0C);
	start.block_index = start_block;
	start.waypoint_type = block_waypoint_type(
		buffer_pointer(challenge + 0x54, start_block));
	block_spawn(buffer_pointer(challenge + 0x54, start_block), start.spawn);

	for (i = 0; i < checkpoint_count; ++i) {
		uint32_t scene = buffer_pointer(race + 0xD4, i);
		extract_trigger(
			challenge, scene, i, WAYPOINT_CHECKPOINT, &checkpoints[i]);
	}
	for (i = 0; i < finish_count; ++i) {
		uint32_t scene = buffer_pointer(race + 0xE0, i);
		extract_trigger(
			challenge, scene, i,
			block_waypoint_type(buffer_pointer(
				challenge + 0x54,
				find_block_index(challenge, scene, WAYPOINT_FINISH))),
			&finishes[i]);
		if (finishes[i].waypoint_type != WAYPOINT_FINISH
		    && finishes[i].waypoint_type != WAYPOINT_START_FINISH)
			fatal("route finish has an invalid waypoint type");
		if (finishes[i].waypoint_type == WAYPOINT_START_FINISH
		    && (finish_count != 1
		        || start.waypoint_type != WAYPOINT_START_FINISH
		        || start.block_index != finishes[i].block_index
		        || read_u32(race + 0xEC) <= 1))
			fatal("shared start/finish route metadata mismatch");
	}

	memset(&metadata, 0, sizeof(metadata));
	metadata.lap_count = read_u32(race + 0xEC);
	metadata.checkpoint_count = checkpoint_count;
	metadata.finish_count = finish_count;
	metadata.reference_count = checkpoint_count + 2;
	metadata.total_race_checkpoints = read_u32(race + 0x2C0);
	metadata.race_checkpoint_limit = read_u32(race + 0x2C4);
	metadata.flags = metadata.lap_count > 1 ? 1u : 0u;
	if (metadata.lap_count == 0)
		fatal("race lap count is zero");

	memcpy(reference[0].position, start.transform + 9, 0x0C);
	for (i = 0; i < checkpoint_count; ++i)
		trigger_center(&checkpoints[i], reference[i + 1].position);
	trigger_center(&finishes[0], reference[checkpoint_count + 1].position);
	for (i = 1; i < metadata.reference_count; ++i) {
		float dx = reference[i].position[0] - reference[i - 1].position[0];
		float dy = reference[i].position[1] - reference[i - 1].position[1];
		float dz = reference[i].position[2] - reference[i - 1].position[2];
		float distance = sqrtf(dx * dx + dy * dy + dz * dz);
		if (!(distance > 0.0f) || !isfinite(distance))
			fatal("degenerate route reference segment");
		reference[i].arc_length =
			reference[i - 1].arc_length + distance;
	}

	memset(&header, 0, sizeof(header));
	memcpy(header.magic, "TMNFROU1", 8);
	header.version = 1;
	header.endian = 0x12345678;
	header.header_size = sizeof(header);
	header.section_count = SECTION_COUNT;
	memcpy(header.exe_sha256, EXE_SHA256, sizeof(EXE_SHA256));
	memcpy(header.track_sha256, g_track_sha256, sizeof(g_track_sha256));
	offset = sizeof(header);
	for (i = 0; i < SECTION_COUNT; ++i) {
		uint32_t count = 1;
		if (i == SEC_CHECKPOINTS)
			count = checkpoint_count;
		else if (i == SEC_FINISH)
			count = finish_count;
		else if (i == SEC_REFERENCE)
			count = metadata.reference_count;
		offset = align8(offset);
		header.sections[i].offset = offset;
		header.sections[i].count = count;
		header.sections[i].stride = strides[i];
		if (count > (UINT32_MAX - offset) / strides[i])
			fatal("route snapshot size overflow");
		offset += count * strides[i];
	}
	header.file_size = offset;
	metadata.start_rel = header.sections[SEC_START].offset;
	metadata.checkpoints_rel = header.sections[SEC_CHECKPOINTS].offset;
	metadata.finish_rel = header.sections[SEC_FINISH].offset;
	metadata.reference_rel = header.sections[SEC_REFERENCE].offset;

	file = (uint8_t *)HeapAlloc(g_heap, HEAP_ZERO_MEMORY, offset);
	if (file == NULL)
		fatal("route file allocation failed");
	memcpy(file, &header, sizeof(header));
	memcpy(
		file + (uint32_t)header.sections[SEC_METADATA].offset,
		&metadata, sizeof(metadata));
	memcpy(
		file + (uint32_t)header.sections[SEC_START].offset,
		&start, sizeof(start));
	memcpy(
		file + (uint32_t)header.sections[SEC_CHECKPOINTS].offset,
		checkpoints, checkpoint_count * sizeof(*checkpoints));
	memcpy(
		file + (uint32_t)header.sections[SEC_FINISH].offset,
		finishes, finish_count * sizeof(*finishes));
	memcpy(
		file + (uint32_t)header.sections[SEC_REFERENCE].offset,
		reference, metadata.reference_count * sizeof(*reference));
	sha256_init(&sha);
	sha256_update(&sha, file + sizeof(header), offset - sizeof(header));
	sha256_final(&sha, ((struct DiskHeader *)file)->payload_sha256);

	handle = CreateFileA(
		g_output_path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL, NULL);
	if (handle == INVALID_HANDLE_VALUE)
		fatal("cannot create route snapshot");
	write_all(handle, file, offset);
	FlushFileBuffers(handle);
	CloseHandle(handle);

	_snprintf(
		line, sizeof(line),
		"route checkpoints=%u finish=%u laps=%u total=%u limit=%u "
		"reference=%u length=%.9g bytes=%u",
		checkpoint_count, finish_count, metadata.lap_count,
		metadata.total_race_checkpoints, metadata.race_checkpoint_limit,
		metadata.reference_count,
		reference[metadata.reference_count - 1].arc_length, offset);
	log_line(line);
	_snprintf(
		line, sizeof(line),
		"start block=%u position=%.9g,%.9g,%.9g spawn=%.9g,%.9g,%.9g",
		start.block_index, start.transform[9], start.transform[10],
		start.transform[11], start.spawn[9], start.spawn[10],
		start.spawn[11]);
	log_line(line);
	for (i = 0; i < checkpoint_count; ++i) {
		float center[3];
		trigger_center(&checkpoints[i], center);
		_snprintf(
			line, sizeof(line),
			"checkpoint index=%u block=%u flags=%08X center=%.9g,%.9g,%.9g "
			"extent=%.9g,%.9g,%.9g spawn=%.9g,%.9g,%.9g no_respawn=%u",
			i, checkpoints[i].block_index, checkpoints[i].tree_flags,
			center[0], center[1], center[2],
			checkpoints[i].box[3], checkpoints[i].box[4],
			checkpoints[i].box[5],
			checkpoints[i].spawn[9], checkpoints[i].spawn[10],
			checkpoints[i].spawn[11], checkpoints[i].no_respawn);
		log_line(line);
	}
	for (i = 0; i < finish_count; ++i) {
		float center[3];
		trigger_center(&finishes[i], center);
		_snprintf(
			line, sizeof(line),
			"finish index=%u block=%u flags=%08X center=%.9g,%.9g,%.9g "
			"extent=%.9g,%.9g,%.9g",
			i, finishes[i].block_index, finishes[i].tree_flags,
			center[0], center[1], center[2],
			finishes[i].box[3], finishes[i].box[4], finishes[i].box[5]);
		log_line(line);
	}
	HeapFree(g_heap, 0, file);
	HeapFree(g_heap, 0, reference);
	HeapFree(g_heap, 0, finishes);
	HeapFree(g_heap, 0, checkpoints);
}

static void __cdecl dump_if_ready(uint32_t zone_dynamic)
{
	uint32_t zone;
	uint32_t race;
	uint32_t challenge;
	uint32_t dyna;
	char line[192];
	RaceTimeFn race_time;
	if (g_dump_started != 0)
		return;
	zone = read_u32(zone_dynamic + 0x168);
	if (zone == 0)
		return;
	race = find_race();
	if (race == 0)
		return;
	challenge = read_u32(race + 0xC4);
	if (challenge == 0 || read_u32(race + 0xE0) == 0)
		return;
	race_time = (RaceTimeFn)(uintptr_t)(
		g_module_base + RACE_TIME_VA - IMAGE_BASE);
	if (InterlockedCompareExchange(&g_probe_logged, 1, 0) == 0) {
		_snprintf(
			line, sizeof(line),
			"route ready race=%08X challenge=%08X time=%u checkpoints=%u "
			"finish=%u dynamic=%u",
			race, challenge, race_time((void *)(uintptr_t)race),
			read_u32(race + 0xD4), read_u32(race + 0xE0),
			read_u32(zone_dynamic + 0x140));
		log_line(line);
	}
	if (race_time((void *)(uintptr_t)race) != 0)
		return;
	dyna = find_initial_dyna(zone_dynamic);
	if (dyna == 0)
		return;
	if (InterlockedCompareExchange(&g_dump_started, 1, 0) != 0)
		return;
	write_snapshot(race, challenge, dyna);
}

static void emit_u8(uint8_t **cursor, uint8_t value)
{
	*(*cursor)++ = value;
}

static void emit_u32(uint8_t **cursor, uint32_t value)
{
	memcpy(*cursor, &value, 4);
	*cursor += 4;
}

static void emit_rel32(uint8_t **cursor, uint8_t opcode, const void *destination)
{
	uintptr_t next;
	int32_t displacement;
	emit_u8(cursor, opcode);
	next = (uintptr_t)*cursor + 4;
	displacement = (int32_t)((uintptr_t)destination - next);
	emit_u32(cursor, (uint32_t)displacement);
}

static uint8_t *allocate_code(void)
{
	uint8_t *memory = (uint8_t *)VirtualAlloc(
		NULL, 128, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
	if (memory == NULL)
		fatal("cannot allocate route hook code");
	return memory;
}

static void install_hook(void)
{
	uint8_t *target = (uint8_t *)(uintptr_t)(
		g_module_base + PHYSICS_STEP2_VA - IMAGE_BASE);
	uint8_t *stub;
	uint8_t *cursor;
	int32_t existing_displacement;
	uint8_t *existing_destination;
	DWORD old_protect;
	DWORD ignored;
	if (target[0] != 0xE9)
		fatal("PhysicsStep2 is not detoured by CoreMod");
	memcpy(&existing_displacement, target + 1, 4);
	existing_destination = target + 5 + existing_displacement;
	g_trampoline = allocate_code();
	cursor = g_trampoline;
	emit_rel32(&cursor, 0xE9, existing_destination);
	stub = allocate_code();
	cursor = stub;
	emit_u8(&cursor, 0x9C);
	emit_u8(&cursor, 0x60);
	emit_u8(&cursor, 0xFF);
	emit_u8(&cursor, 0x74);
	emit_u8(&cursor, 0x24);
	emit_u8(&cursor, 0x18);
	emit_rel32(&cursor, 0xE8, dump_if_ready);
	emit_u8(&cursor, 0x83);
	emit_u8(&cursor, 0xC4);
	emit_u8(&cursor, 0x04);
	emit_u8(&cursor, 0x61);
	emit_u8(&cursor, 0x9D);
	emit_rel32(&cursor, 0xE9, g_trampoline);
	if (!VirtualProtect(target, 5, PAGE_EXECUTE_READWRITE, &old_protect))
		fatal("cannot make PhysicsStep2 writable");
	cursor = target;
	emit_rel32(&cursor, 0xE9, stub);
	if (!VirtualProtect(target, 5, old_protect, &ignored))
		fatal("cannot restore PhysicsStep2 protection");
	FlushInstructionCache(GetCurrentProcess(), target, 5);
	log_line("route dump hook installed");
}

static DWORD WINAPI initialize_dumper(void *unused)
{
	HMODULE executable;
	char log_path[MAX_PATH];
	char track_sha256[65];
	uint32_t i;
	(void)unused;
	if (GetEnvironmentVariableA(
		    "TMNF_ROUTE_PATH", g_output_path, sizeof(g_output_path)) == 0)
		fatal("TMNF_ROUTE_PATH is required");
	if (GetEnvironmentVariableA(
		    "TMNF_TRACK_SHA256", track_sha256,
		    sizeof(track_sha256)) != 64)
		fatal("TMNF_TRACK_SHA256 must contain 64 hexadecimal characters");
	for (i = 0; i < 32; ++i) {
		unsigned int byte;
		if (sscanf(track_sha256 + i * 2, "%2x", &byte) != 1)
			fatal("TMNF_TRACK_SHA256 is not hexadecimal");
		g_track_sha256[i] = (uint8_t)byte;
	}
	if (_snprintf(
		    log_path, sizeof(log_path), "%s.log", g_output_path) < 0)
		fatal("route log path is too long");
	g_log = CreateFileA(
		log_path, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL, NULL);
	if (g_log == INVALID_HANDLE_VALUE)
		fatal("cannot create route dumper log");
	executable = GetModuleHandleA(NULL);
	if (executable == NULL)
		fatal("cannot locate TmForever module");
	g_module_base = (uint32_t)(uintptr_t)executable;
	g_heap = GetProcessHeap();
	install_hook();
	return 0;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, void *reserved)
{
	(void)reserved;
	if (reason == DLL_PROCESS_ATTACH) {
		HANDLE thread;
		DisableThreadLibraryCalls(instance);
		thread = CreateThread(NULL, 0, initialize_dumper, NULL, 0, NULL);
		if (thread == NULL)
			return FALSE;
		CloseHandle(thread);
	} else if (reason == DLL_PROCESS_DETACH) {
		if (g_log != INVALID_HANDLE_VALUE)
			CloseHandle(g_log);
	}
	return TRUE;
}
