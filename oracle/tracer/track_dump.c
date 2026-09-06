#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define IMAGE_BASE 0x00400000u
#define PHYSICS_STEP2_VA 0x00549C90u
#define MATERIAL_TABLE_VA 0x00D6EEC0u

#define SECTION_COUNT 12u
#define MATERIAL_COUNT 31u

enum SectionId {
	SEC_ENTRIES,
	SEC_SURFACES,
	SEC_MESHES,
	SEC_VERTICES,
	SEC_FACES,
	SEC_NODES,
	SEC_MATERIAL_IDS,
	SEC_MATERIAL_DATA,
	SEC_COLLISION_PAIRS,
	SEC_CORPUS_ISOS,
	SEC_WATER,
	SEC_WATER_CELLS
};

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

struct DiskEntry {
	uint32_t skip_count;
	uint8_t box[0x18];
	uint8_t iso[0x30];
	uint32_t tree_flags;
	uint64_t surface_rel;
	uint32_t tree_id;
	uint32_t corpus_id;
};

struct DiskSurface {
	uint64_t mesh_rel;
	uint64_t material_ids_rel;
	uint32_t material_count;
	uint32_t reserved;
};

/* CHmsZone+0x154 GmMap2<unsigned char> water map plus the collection's
 * surface level (+0x178) and floor (+0x17c), read by 0x007C2910. */
struct DiskWater {
	float cell_x;
	float cell_z;
	float origin_x;
	float origin_z;
	uint32_t width;
	uint32_t height;
	uint32_t default_cell;
	float level;
	float floor;
	uint32_t reserved;
};

struct DiskMesh {
	uint32_t vtable;
	uint16_t material_index;
	uint8_t type;
	uint8_t reserved;
	uint32_t vertex_count;
	uint32_t pad0c;
	uint64_t vertices_rel;
	uint32_t face_count;
	uint32_t pad1c;
	uint64_t faces_rel;
	uint32_t node_count;
	uint32_t pad2c;
	uint64_t nodes_rel;
};
#pragma pack(pop)

_Static_assert(sizeof(struct DiskSection) == 0x10, "section size");
_Static_assert(sizeof(struct DiskHeader) == 0x150, "header size");
_Static_assert(sizeof(struct DiskWater) == 0x28, "water size");
_Static_assert(sizeof(struct DiskEntry) == 0x60, "entry size");
_Static_assert(sizeof(struct DiskSurface) == 0x18, "surface size");
_Static_assert(sizeof(struct DiskMesh) == 0x38, "mesh size");

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

struct ByteVec {
	uint8_t *data;
	uint32_t size;
	uint32_t capacity;
};

struct AddressMapEntry {
	uint32_t address;
	uint32_t index;
	uint32_t count;
};

struct AddressMap {
	struct AddressMapEntry *entries;
	uint32_t count;
	uint32_t capacity;
};

struct Snapshot {
	struct ByteVec sections[SECTION_COUNT];
	struct AddressMap surfaces;
	struct AddressMap meshes;
	struct AddressMap vertices;
	struct AddressMap faces;
	struct AddressMap nodes;
	struct AddressMap material_ids;
	struct AddressMap trees;
	struct AddressMap corpora;
	struct AddressMap pairs;
};

static HANDLE g_heap;
static HANDLE g_log = INVALID_HANDLE_VALUE;
static char g_output_path[MAX_PATH];
static uint32_t g_module_base;
static uint8_t *g_trampoline;
static uint8_t g_track_sha256[32];
static volatile LONG g_dump_started;

static void fatal(const char *message);

static void read_runtime(uint32_t address, void *output, uint32_t size)
{
	SIZE_T read = 0;
	char message[128];
	if (!ReadProcessMemory(
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

static uint16_t read_u16(uint32_t address)
{
	uint16_t value;
	read_runtime(address, &value, sizeof(value));
	return value;
}

static uint8_t read_u8(uint32_t address)
{
	uint8_t value;
	read_runtime(address, &value, sizeof(value));
	return value;
}

static void write_all(HANDLE file, const void *data, DWORD len)
{
	const uint8_t *cursor = (const uint8_t *)data;
	while (len != 0) {
		DWORD written = 0;
		if (!WriteFile(file, cursor, len, &written, NULL) || written == 0)
			TerminateProcess(GetCurrentProcess(), 130);
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
	TerminateProcess(GetCurrentProcess(), 131);
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

static void vec_reserve(struct ByteVec *vec, uint32_t extra)
{
	uint32_t required;
	uint32_t capacity;
	uint8_t *data;
	if (extra > UINT32_MAX - vec->size)
		fatal("snapshot vector overflow");
	required = vec->size + extra;
	if (required <= vec->capacity)
		return;
	capacity = vec->capacity == 0 ? 256 : vec->capacity;
	while (capacity < required) {
		if (capacity > UINT32_MAX / 2)
			fatal("snapshot vector capacity overflow");
		capacity *= 2;
	}
	if (vec->data == NULL)
		data = (uint8_t *)HeapAlloc(g_heap, 0, capacity);
	else
		data = (uint8_t *)HeapReAlloc(g_heap, 0, vec->data, capacity);
	if (data == NULL)
		fatal("snapshot vector allocation failed");
	vec->data = data;
	vec->capacity = capacity;
}

static uint32_t vec_append(
	struct ByteVec *vec, const void *data, uint32_t size)
{
	uint32_t offset = vec->size;
	vec_reserve(vec, size);
	memcpy(vec->data + vec->size, data, size);
	vec->size += size;
	return offset;
}

static uint32_t align8(uint32_t value)
{
	if (value > UINT32_MAX - 7)
		fatal("snapshot alignment overflow");
	return (value + 7) & ~7u;
}

static void map_reserve(struct AddressMap *map)
{
	uint32_t capacity;
	struct AddressMapEntry *entries;
	if (map->count < map->capacity)
		return;
	capacity = map->capacity == 0 ? 64 : map->capacity * 2;
	if (capacity < map->capacity)
		fatal("address map capacity overflow");
	if (map->entries == NULL)
		entries = (struct AddressMapEntry *)HeapAlloc(
			g_heap, 0, capacity * sizeof(*entries));
	else
		entries = (struct AddressMapEntry *)HeapReAlloc(
			g_heap, 0, map->entries, capacity * sizeof(*entries));
	if (entries == NULL)
		fatal("address map allocation failed");
	map->entries = entries;
	map->capacity = capacity;
}

static int map_find(
	const struct AddressMap *map, uint32_t address, uint32_t *index,
	uint32_t *count)
{
	uint32_t i;
	for (i = 0; i < map->count; ++i) {
		if (map->entries[i].address == address) {
			*index = map->entries[i].index;
			*count = map->entries[i].count;
			return 1;
		}
	}
	return 0;
}

static void map_add(
	struct AddressMap *map, uint32_t address, uint32_t index,
	uint32_t count)
{
	map_reserve(map);
	map->entries[map->count].address = address;
	map->entries[map->count].index = index;
	map->entries[map->count].count = count;
	++map->count;
}

static uint32_t stable_id(struct AddressMap *map, uint32_t address)
{
	uint32_t index;
	uint32_t count;
	if (address == 0)
		return 0;
	if (map_find(map, address, &index, &count))
		return index;
	index = map->count + 1;
	map_add(map, address, index, 1);
	return index;
}

static uint32_t append_raw_array(
	struct Snapshot *snapshot, struct AddressMap *map,
	enum SectionId section_id, uint32_t address, uint32_t count,
	uint32_t stride)
{
	struct ByteVec *section = &snapshot->sections[section_id];
	uint32_t index;
	uint32_t old_count;
	uint32_t bytes;
	if (address == 0 || count == 0)
		fatal("empty runtime array");
	if (map_find(map, address, &index, &old_count)) {
		if (old_count != count)
			fatal("deduplicated array count mismatch");
		return index;
	}
	if (count > UINT32_MAX / stride)
		fatal("runtime array size overflow");
	bytes = count * stride;
	if (section->size % stride != 0)
		fatal("section stride alignment mismatch");
	index = section->size / stride;
	vec_append(section, (const void *)(uintptr_t)address, bytes);
	map_add(map, address, index, count);
	return index;
}

static uint32_t append_material_ids(
	struct Snapshot *snapshot, uint32_t data, uint32_t count)
{
	struct ByteVec *section = &snapshot->sections[SEC_MATERIAL_IDS];
	uint32_t index;
	uint32_t old_count;
	uint32_t i;
	if (data == 0 || count == 0)
		fatal("surface has no material remap");
	if (map_find(&snapshot->material_ids, data, &index, &old_count)) {
		if (old_count != count)
			fatal("deduplicated material remap count mismatch");
		return index;
	}
	index = section->size;
	vec_reserve(section, count);
	for (i = 0; i < count; ++i) {
		uint32_t material = read_u32(data + i * 4);
		if (material == 0)
			fatal("null surface material");
		section->data[section->size++] = read_u8(material + 0x18);
	}
	map_add(&snapshot->material_ids, data, index, count);
	return index;
}

static uint32_t process_mesh(struct Snapshot *snapshot, uint32_t mesh_address)
{
	struct DiskMesh mesh;
	uint32_t existing;
	uint32_t ignored;
	uint32_t index;
	uint32_t vertex_data;
	uint32_t face_data;
	uint32_t node_data;
	uint32_t vertex_index;
	uint32_t face_index;
	uint32_t node_index;
	if (map_find(&snapshot->meshes, mesh_address, &existing, &ignored))
		return existing;
	if (mesh_address == 0 || read_u8(mesh_address + 6) != 7)
		fatal("A01 contains a non-mesh static surface");
	memset(&mesh, 0, sizeof(mesh));
	mesh.vtable = read_u32(mesh_address) - g_module_base + IMAGE_BASE;
	mesh.material_index = read_u16(mesh_address + 4);
	mesh.type = read_u8(mesh_address + 6);
	mesh.reserved = read_u8(mesh_address + 7);
	mesh.vertex_count = read_u32(mesh_address + 0x08);
	vertex_data = read_u32(mesh_address + 0x0C);
	mesh.face_count = read_u32(mesh_address + 0x10);
	face_data = read_u32(mesh_address + 0x14);
	mesh.node_count = read_u32(mesh_address + 0x20);
	node_data = read_u32(mesh_address + 0x24);
	if (mesh.node_count > read_u32(mesh_address + 0x28))
		fatal("mesh node count exceeds capacity");
	vertex_index = append_raw_array(
		snapshot, &snapshot->vertices, SEC_VERTICES,
		vertex_data, mesh.vertex_count, 0x0C);
	face_index = append_raw_array(
		snapshot, &snapshot->faces, SEC_FACES,
		face_data, mesh.face_count, 0x20);
	node_index = append_raw_array(
		snapshot, &snapshot->nodes, SEC_NODES,
		node_data, mesh.node_count, 0x20);
	mesh.vertices_rel = vertex_index;
	mesh.faces_rel = face_index;
	mesh.nodes_rel = node_index;
	index = snapshot->sections[SEC_MESHES].size / sizeof(mesh);
	vec_append(&snapshot->sections[SEC_MESHES], &mesh, sizeof(mesh));
	map_add(&snapshot->meshes, mesh_address, index, 1);
	return index;
}

static uint32_t process_surface(
	struct Snapshot *snapshot, uint32_t surface_address)
{
	struct DiskSurface surface;
	uint32_t existing;
	uint32_t ignored;
	uint32_t geom_wrapper;
	uint32_t mesh_address;
	uint32_t material_data;
	uint32_t material_capacity;
	uint32_t index;
	if (map_find(&snapshot->surfaces, surface_address, &existing, &ignored))
		return existing;
	if (surface_address == 0)
		fatal("null static surface");
	geom_wrapper = read_u32(surface_address + 0x14);
	if (geom_wrapper == 0)
		fatal("null surface geometry wrapper");
	mesh_address = read_u32(geom_wrapper + 0x34);
	memset(&surface, 0, sizeof(surface));
	surface.material_count = read_u32(surface_address + 0x18);
	material_data = read_u32(surface_address + 0x1C);
	material_capacity = read_u32(surface_address + 0x20);
	if (surface.material_count > material_capacity)
		fatal("surface material count exceeds capacity");
	surface.mesh_rel = process_mesh(snapshot, mesh_address);
	surface.material_ids_rel = append_material_ids(
		snapshot, material_data, surface.material_count);
	index = snapshot->sections[SEC_SURFACES].size / sizeof(surface);
	vec_append(&snapshot->sections[SEC_SURFACES], &surface, sizeof(surface));
	map_add(&snapshot->surfaces, surface_address, index, 1);
	return index;
}

static uint32_t locate_static_group(uint32_t zone, uint32_t *entry_count)
{
	uint32_t found = 0;
	uint32_t i;
	for (i = 0; i < 5; ++i) {
		uint32_t group = zone + i * 0x44;
		uint32_t count = read_u32(group + 0x30);
		if (read_u32(group + 0x40) != 0 && count > 1) {
			if (found != 0)
				fatal("multiple populated static groups");
			if (count > read_u32(group + 0x38)
			    || read_u32(group + 0x34) == 0)
				fatal("invalid static entry buffer");
			found = group;
			*entry_count = count;
		}
	}
	return found;
}

static uint32_t locate_car_group(uint32_t zone, uint32_t static_group)
{
	uint32_t candidate = 0;
	uint32_t i;
	char message[192];
	for (i = 0; i < 5; ++i) {
		uint32_t group = zone + i * 0x44;
		uint32_t against_count;
		uint32_t against_data;
		uint32_t j;
		int references_static = 0;
		if (read_u32(group + 0x40) != 0
		    || read_u32(group + 0x0C) == 0)
			continue;
		against_count = read_u32(group + 0x24);
		against_data = read_u32(group + 0x28);
		_snprintf(
			message, sizeof(message),
			"group=%u address=%08X corpora=%u against=%u@%08X cap=%u",
			i, group, read_u32(group + 0x0C), against_count,
			against_data, read_u32(group + 0x2C));
		log_line(message);
		if (against_count > read_u32(group + 0x2C))
			fatal("against-group count exceeds capacity");
		for (j = 0; j < against_count; ++j) {
			if (read_u32(against_data + j * 0x1C) == static_group)
				references_static = 1;
		}
		if (references_static) {
			if (candidate != 0)
				fatal("multiple car-group candidates");
			candidate = group;
		}
	}
	if (candidate == 0)
		fatal("car collision group not found");
	return candidate;
}

static void append_collision_pairs(
	struct Snapshot *snapshot, uint32_t car_group,
	uint32_t expected_static_group)
{
	uint32_t count = read_u32(car_group + 0x24);
	uint32_t data = read_u32(car_group + 0x28);
	uint32_t static_target = 0;
	uint32_t i;
	for (i = 0; i < count; ++i) {
		uint32_t element = data + i * 0x1C;
		uint32_t target = read_u32(element);
		uint32_t pair = read_u32(element + 4);
		uint32_t index;
		uint32_t ignored;
		if (target != 0 && read_u32(target + 0x40) != 0
		    && read_u32(target + 0x30) > 1) {
			if (static_target != 0 && static_target != target)
				fatal("multiple static against-group targets");
			static_target = target;
		}
		if (pair == 0)
			fatal("null collision pair");
		if (!map_find(&snapshot->pairs, pair, &index, &ignored)) {
			index = snapshot->sections[SEC_COLLISION_PAIRS].size / 0x14;
			vec_append(
				&snapshot->sections[SEC_COLLISION_PAIRS],
				(const void *)(uintptr_t)pair, 0x14);
			map_add(&snapshot->pairs, pair, index, 1);
		}
	}
	if (static_target != expected_static_group)
		fatal("static against-group target mismatch");
	if (snapshot->pairs.count == 0)
		fatal("no collision pairs found");
}

static void append_water(struct Snapshot *snapshot, uint32_t zone)
{
	struct DiskWater water;
	uint32_t map = zone + 0x154;
	uint32_t cell_count;
	uint32_t cells;
	char message[192];
	memset(&water, 0, sizeof(water));
	read_runtime(map, &water, 0x18);
	water.default_cell = read_u32(map + 0x18) & 0xFFu;
	cell_count = read_u32(map + 0x1C);
	cells = read_u32(map + 0x20);
	read_runtime(zone + 0x178, &water.level, 4);
	read_runtime(zone + 0x17C, &water.floor, 4);
	if (water.cell_x != water.cell_z || !(water.cell_x > 0.0f)
	    || water.origin_x != 0.0f || water.origin_z != 0.0f
	    || water.width == 0 || water.width > 256
	    || water.height == 0 || water.height > 256
	    || water.default_cell > 1
	    || cell_count != water.width * water.height || cells == 0
	    || !(water.level > -1000.0f && water.level < 1000.0f)
	    || !(water.floor <= water.level)) {
		_snprintf(
			message, sizeof(message),
			"unexpected water map: cell=%g/%g origin=%g/%g size=%ux%u "
			"default=%u count=%u cells=%08X level=%g floor=%g",
			(double)water.cell_x, (double)water.cell_z,
			(double)water.origin_x, (double)water.origin_z,
			water.width, water.height, water.default_cell, cell_count,
			cells, (double)water.level, (double)water.floor);
		fatal(message);
	}
	_snprintf(
		message, sizeof(message),
		"water map cell=%g size=%ux%u default=%u level=%g floor=%g",
		(double)water.cell_x, water.width, water.height,
		water.default_cell, (double)water.level, (double)water.floor);
	log_line(message);
	vec_append(&snapshot->sections[SEC_WATER], &water, sizeof(water));
	vec_append(
		&snapshot->sections[SEC_WATER_CELLS],
		(const void *)(uintptr_t)cells, cell_count);
}

static void gather_snapshot(
	struct Snapshot *snapshot, uint32_t zone, uint32_t static_group,
	uint32_t entry_count)
{
	uint32_t entries = read_u32(static_group + 0x34);
	uint32_t car_group;
	uint32_t inactive_surface_index = UINT32_MAX;
	uint32_t i;
	for (i = 0; i < entry_count; ++i) {
		uint32_t source = entries + i * 0x58;
		uint32_t surface = read_u32(source + 0x4C);
		uint32_t tree = read_u32(source + 0x50);
		if (surface != 0 && tree != 0) {
			inactive_surface_index = process_surface(snapshot, surface);
			break;
		}
	}
	if (inactive_surface_index == UINT32_MAX)
		fatal("static buffer contains no active surface");
	for (i = 0; i < entry_count; ++i) {
		uint32_t source = entries + i * 0x58;
		struct DiskEntry entry;
		uint32_t tree = read_u32(source + 0x50);
		uint32_t corpus = read_u32(source + 0x54);
		uint32_t surface = read_u32(source + 0x4C);
		memset(&entry, 0, sizeof(entry));
		entry.skip_count = read_u32(source);
		memcpy(entry.box, (const void *)(uintptr_t)(source + 4), 0x18);
		memcpy(entry.iso, (const void *)(uintptr_t)(source + 0x1C), 0x30);
		if (surface == 0 || tree == 0) {
			entry.tree_flags = 0;
			entry.surface_rel = inactive_surface_index;
		} else {
			entry.tree_flags = read_u32(tree + 0x9C);
			entry.surface_rel = process_surface(snapshot, surface);
		}
		entry.tree_id = stable_id(&snapshot->trees, tree);
		entry.corpus_id = stable_id(&snapshot->corpora, corpus);
		vec_append(&snapshot->sections[SEC_ENTRIES], &entry, sizeof(entry));
	}
	log_line("static entries gathered");
	/* CHmsCorpus::GetLocation (0x005474A0) returns corpus+0x18 for static
	 * corpora; WheelAbsorbContact reads its rotation for the contact axis.
	 * Stable corpus id k is the k-th map entry, so the table is indexed by
	 * corpus_id - 1. Inactive entries (null surface or tree) carry stale
	 * corpus pointers that must not be dereferenced; their rows stay zero. */
	for (i = 0; i < snapshot->corpora.count; ++i) {
		uint32_t corpus = snapshot->corpora.entries[i].address;
		uint32_t j;
		int active = 0;
		uint8_t zero[0x30];
		if (snapshot->corpora.entries[i].index != i + 1)
			fatal("corpus ids are not dense");
		for (j = 0; j < entry_count && !active; ++j) {
			uint32_t source = entries + j * 0x58;
			if (read_u32(source + 0x54) == corpus
			    && read_u32(source + 0x4C) != 0
			    && read_u32(source + 0x50) != 0)
				active = 1;
		}
		if (active) {
			vec_append(
				&snapshot->sections[SEC_CORPUS_ISOS],
				(const void *)(uintptr_t)(corpus + 0x18), 0x30);
		} else {
			memset(zero, 0, sizeof(zero));
			vec_append(&snapshot->sections[SEC_CORPUS_ISOS], zero, 0x30);
		}
	}
	log_line("static corpus locations gathered");
	vec_append(
		&snapshot->sections[SEC_MATERIAL_DATA],
		(const void *)(uintptr_t)(
			g_module_base + MATERIAL_TABLE_VA - IMAGE_BASE),
		MATERIAL_COUNT * 8);
	log_line("physical materials gathered");
	car_group = locate_car_group(zone, static_group);
	log_line("car group located");
	append_collision_pairs(snapshot, car_group, static_group);
	log_line("collision pairs gathered");
	append_water(snapshot, zone);
	log_line("water map gathered");
}

static void patch_relative_offsets(
	struct Snapshot *snapshot, const struct DiskHeader *header)
{
	uint32_t i;
	uint32_t entry_count =
		snapshot->sections[SEC_ENTRIES].size / sizeof(struct DiskEntry);
	uint32_t surface_count =
		snapshot->sections[SEC_SURFACES].size / sizeof(struct DiskSurface);
	uint32_t mesh_count =
		snapshot->sections[SEC_MESHES].size / sizeof(struct DiskMesh);
	struct DiskEntry *entries =
		(struct DiskEntry *)snapshot->sections[SEC_ENTRIES].data;
	struct DiskSurface *surfaces =
		(struct DiskSurface *)snapshot->sections[SEC_SURFACES].data;
	struct DiskMesh *meshes =
		(struct DiskMesh *)snapshot->sections[SEC_MESHES].data;
	for (i = 0; i < entry_count; ++i)
		entries[i].surface_rel =
			header->sections[SEC_SURFACES].offset
			+ entries[i].surface_rel * sizeof(struct DiskSurface);
	for (i = 0; i < surface_count; ++i) {
		surfaces[i].mesh_rel =
			header->sections[SEC_MESHES].offset
			+ surfaces[i].mesh_rel * sizeof(struct DiskMesh);
		surfaces[i].material_ids_rel =
			header->sections[SEC_MATERIAL_IDS].offset
			+ surfaces[i].material_ids_rel;
	}
	for (i = 0; i < mesh_count; ++i) {
		meshes[i].vertices_rel =
			header->sections[SEC_VERTICES].offset
			+ meshes[i].vertices_rel * 0x0C;
		meshes[i].faces_rel =
			header->sections[SEC_FACES].offset
			+ meshes[i].faces_rel * 0x20;
		meshes[i].nodes_rel =
			header->sections[SEC_NODES].offset
			+ meshes[i].nodes_rel * 0x20;
	}
}

static void write_snapshot(struct Snapshot *snapshot)
{
	static const uint32_t strides[SECTION_COUNT] = {
		0x60, 0x18, 0x38, 0x0C, 0x20, 0x20, 1, 8, 0x14, 0x30, 0x28, 1
	};
	struct DiskHeader header;
	struct ByteVec file;
	struct Sha256 sha;
	HANDLE handle;
	uint32_t offset;
	uint32_t i;
	char summary[320];
	char temp_path[MAX_PATH + 8];
	memset(&header, 0, sizeof(header));
	memset(&file, 0, sizeof(file));
	memcpy(header.magic, "TMNFTRK1", 8);
	header.version = 3;
	header.endian = 0x12345678;
	header.header_size = sizeof(header);
	header.section_count = SECTION_COUNT;
	memcpy(header.exe_sha256, EXE_SHA256, 32);
	memcpy(header.track_sha256, g_track_sha256, 32);
	offset = sizeof(header);
	for (i = 0; i < SECTION_COUNT; ++i) {
		offset = align8(offset);
		header.sections[i].offset = offset;
		header.sections[i].stride = strides[i];
		if (snapshot->sections[i].size == 0
		    || snapshot->sections[i].size % strides[i] != 0)
			fatal("empty or malformed snapshot section");
		header.sections[i].count =
			snapshot->sections[i].size / strides[i];
		if (snapshot->sections[i].size > UINT32_MAX - offset)
			fatal("snapshot file size overflow");
		offset += snapshot->sections[i].size;
	}
	header.file_size = offset;
	patch_relative_offsets(snapshot, &header);
	vec_reserve(&file, offset);
	memset(file.data, 0, offset);
	file.size = offset;
	memcpy(file.data, &header, sizeof(header));
	for (i = 0; i < SECTION_COUNT; ++i)
		memcpy(
			file.data + (uint32_t)header.sections[i].offset,
			snapshot->sections[i].data,
			snapshot->sections[i].size);
	sha256_init(&sha);
	sha256_update(&sha, file.data + sizeof(header), offset - sizeof(header));
	sha256_final(&sha, ((struct DiskHeader *)file.data)->payload_sha256);
	/*
	 * Write beside the target and rename over it. Truncating the target in
	 * place tears down the pages under any process that has it mapped
	 * (src/track.c maps snapshots MAP_PRIVATE); a rename leaves the old
	 * inode intact for existing mappings.
	 */
	_snprintf(temp_path, sizeof(temp_path), "%s.tmp", g_output_path);
	handle = CreateFileA(
		temp_path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL, NULL);
	if (handle == INVALID_HANDLE_VALUE)
		fatal("cannot create track snapshot");
	write_all(handle, file.data, file.size);
	FlushFileBuffers(handle);
	CloseHandle(handle);
	if (!MoveFileExA(temp_path, g_output_path, MOVEFILE_REPLACE_EXISTING))
		fatal("cannot rename track snapshot into place");
	_snprintf(
		summary, sizeof(summary),
		"snapshot entries=%u surfaces=%u meshes=%u vertices=%u "
		"faces=%u nodes=%u material_ids=%u pairs=%u corpora=%u bytes=%u",
		header.sections[SEC_ENTRIES].count,
		header.sections[SEC_SURFACES].count,
		header.sections[SEC_MESHES].count,
		header.sections[SEC_VERTICES].count,
		header.sections[SEC_FACES].count,
		header.sections[SEC_NODES].count,
		header.sections[SEC_MATERIAL_IDS].count,
		header.sections[SEC_COLLISION_PAIRS].count,
		header.sections[SEC_CORPUS_ISOS].count,
		offset);
	log_line(summary);
}

static void __cdecl dump_if_ready(uint32_t zone_dynamic)
{
	struct Snapshot snapshot;
	uint32_t zone;
	uint32_t static_group;
	uint32_t entry_count = 0;
	if (g_dump_started != 0)
		return;
	zone = read_u32(zone_dynamic + 0x168);
	if (zone == 0)
		return;
	static_group = locate_static_group(zone, &entry_count);
	if (static_group == 0)
		return;
	if (InterlockedCompareExchange(&g_dump_started, 1, 0) != 0)
		return;
	memset(&snapshot, 0, sizeof(snapshot));
	gather_snapshot(&snapshot, zone, static_group, entry_count);
	write_snapshot(&snapshot);
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
		fatal("cannot allocate hook code");
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
	log_line("track dump hook installed");
}

static DWORD WINAPI initialize_dumper(void *unused)
{
	HMODULE executable;
	char log_path[MAX_PATH];
	char track_sha256[65];
	uint32_t i;
	(void)unused;
	if (GetEnvironmentVariableA(
		    "TMNF_TRACK_PATH", g_output_path, sizeof(g_output_path)) == 0)
		fatal("TMNF_TRACK_PATH is required");
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
		fatal("track log path is too long");
	g_log = CreateFileA(
		log_path, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL, NULL);
	if (g_log == INVALID_HANDLE_VALUE)
		fatal("cannot create track dumper log");
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
