#include "track.h"

#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const uint8_t TMNF_21126_EXE_SHA256[32] = {
	0x38, 0x47, 0xcf, 0x9f, 0x20, 0xbf, 0xc6, 0x39,
	0x14, 0x45, 0x00, 0x60, 0xed, 0x52, 0x8c, 0x12,
	0x10, 0x4f, 0x74, 0x3d, 0x96, 0xad, 0x23, 0xd6,
	0xe7, 0x6a, 0xbd, 0x17, 0x8d, 0xe8, 0xc8, 0x4f,
};

static void fail(const char *message) {
	fprintf(stderr, "tmnf track: %s\n", message);
	abort();
}

/* Compact SHA-256, used to authenticate the immutable mmap payload before
 * relative pointers are fixed up in the private mapping. */
typedef struct {
	uint32_t state[8];
	uint64_t byte_count;
	uint8_t block[64];
	uint32_t block_len;
} Sha256;

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

static uint32_t rotr32(uint32_t value, uint32_t bits) {
	return (value >> bits) | (value << (32 - bits));
}

static uint32_t load_be32(const uint8_t *p) {
	return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 |
		(uint32_t)p[2] << 8 | p[3];
}

static void store_be32(uint8_t *p, uint32_t value) {
	p[0] = (uint8_t)(value >> 24);
	p[1] = (uint8_t)(value >> 16);
	p[2] = (uint8_t)(value >> 8);
	p[3] = (uint8_t)value;
}

static void sha256_transform(Sha256 *ctx, const uint8_t block[64]) {
	uint32_t w[64];
	for (uint32_t i = 0; i < 16; ++i) {
		w[i] = load_be32(block + i * 4);
	}
	for (uint32_t i = 16; i < 64; ++i) {
		uint32_t s0 = rotr32(w[i - 15], 7) ^
			rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
		uint32_t s1 = rotr32(w[i - 2], 17) ^
			rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}

	uint32_t a = ctx->state[0];
	uint32_t b = ctx->state[1];
	uint32_t c = ctx->state[2];
	uint32_t d = ctx->state[3];
	uint32_t e = ctx->state[4];
	uint32_t f = ctx->state[5];
	uint32_t g = ctx->state[6];
	uint32_t h = ctx->state[7];
	for (uint32_t i = 0; i < 64; ++i) {
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

static void sha256_init(Sha256 *ctx) {
	*ctx = (Sha256){
		.state = {
			0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
			0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
		},
	};
}

static void sha256_update(Sha256 *ctx, const void *data, size_t len) {
	const uint8_t *bytes = data;
	ctx->byte_count += len;
	while (len != 0) {
		size_t space = 64 - ctx->block_len;
		size_t take = len < space ? len : space;
		memcpy(ctx->block + ctx->block_len, bytes, take);
		ctx->block_len += (uint32_t)take;
		bytes += take;
		len -= take;
		if (ctx->block_len == 64) {
			sha256_transform(ctx, ctx->block);
			ctx->block_len = 0;
		}
	}
}

static void sha256_final(Sha256 *ctx, uint8_t digest[32]) {
	uint64_t bit_count = ctx->byte_count * 8;
	ctx->block[ctx->block_len++] = 0x80;
	if (ctx->block_len > 56) {
		memset(ctx->block + ctx->block_len, 0, 64 - ctx->block_len);
		sha256_transform(ctx, ctx->block);
		ctx->block_len = 0;
	}
	memset(ctx->block + ctx->block_len, 0, 56 - ctx->block_len);
	for (uint32_t i = 0; i < 8; ++i) {
		ctx->block[63 - i] = (uint8_t)(bit_count >> (i * 8));
	}
	sha256_transform(ctx, ctx->block);
	for (uint32_t i = 0; i < 8; ++i) {
		store_be32(digest + i * 4, ctx->state[i]);
	}
}

static uint64_t section_bytes(const TmnfTrackSection *section) {
	uint64_t bytes = (uint64_t)section->count * section->stride;
	if (section->stride != 0 &&
		bytes / section->stride != section->count) {
		fail("section size overflow");
	}
	return bytes;
}

static void validate_sections(
	const TmnfTrackHeader *header, size_t mapping_size) {
	static const uint32_t expected_stride[TMNF_TRACK_SECTION_COUNT] = {
		sizeof(TmnfTrackStaticEntry),
		sizeof(TmnfTrackSurface),
		sizeof(TmnfTrackMesh),
		sizeof(GmVec3),
		sizeof(GmSurfMeshFace),
		sizeof(GmSurfMeshNode),
		1,
		sizeof(TmnfTrackMaterialData),
		sizeof(TmnfTrackCollisionPair),
		sizeof(GmIso4),
		sizeof(TmnfTrackWaterHeader),
		1,
	};

	for (uint32_t i = 0; i < TMNF_TRACK_SECTION_COUNT; ++i) {
		const TmnfTrackSection *section = &header->sections[i];
		if (section->stride != expected_stride[i] ||
			section->count == 0 ||
			(section->offset & 7) != 0 ||
			section->offset < sizeof(TmnfTrackHeader)) {
			fail("invalid section descriptor");
		}
		uint64_t bytes = section_bytes(section);
		if (section->offset > mapping_size ||
			bytes > mapping_size - section->offset) {
			fail("section outside file");
		}
	}
	for (uint32_t i = 0; i < TMNF_TRACK_SECTION_COUNT; ++i) {
		uint64_t a0 = header->sections[i].offset;
		uint64_t a1 = a0 + section_bytes(&header->sections[i]);
		for (uint32_t j = i + 1; j < TMNF_TRACK_SECTION_COUNT; ++j) {
			uint64_t b0 = header->sections[j].offset;
			uint64_t b1 = b0 + section_bytes(&header->sections[j]);
			if (a0 < b1 && b0 < a1) {
				fail("overlapping sections");
			}
		}
	}
}

static void *resolve_array(
	uint8_t *base, const TmnfTrackSection *section,
	uint64_t relative, uint32_t count) {
	uint64_t bytes = (uint64_t)count * section->stride;
	uint64_t section_end = section->offset + section_bytes(section);
	if (relative < section->offset ||
		relative > section_end ||
		bytes > section_end - relative ||
		(relative - section->offset) % section->stride != 0) {
		fail("relative pointer outside target section");
	}
	return base + relative;
}

static void fix_meshes(uint8_t *base, TmnfTrackHeader *header) {
	const TmnfTrackSection *mesh_section =
		&header->sections[TMNF_TRACK_MESHES];
	TmnfTrackMesh *meshes = (TmnfTrackMesh *)(base + mesh_section->offset);
	for (uint32_t i = 0; i < mesh_section->count; ++i) {
		TmnfTrackMesh *raw = &meshes[i];
		if (raw->base.type != GM_SURF_MESH ||
			raw->vertex_count == 0 || raw->face_count == 0 ||
			raw->node_count == 0) {
			fail("invalid mesh header");
		}
		uint64_t vertices_rel = raw->vertices_rel;
		uint64_t faces_rel = raw->faces_rel;
		uint64_t nodes_rel = raw->nodes_rel;
		raw->vertices_rel = (uintptr_t)resolve_array(
			base, &header->sections[TMNF_TRACK_VERTICES],
			vertices_rel, raw->vertex_count);
		raw->faces_rel = (uintptr_t)resolve_array(
			base, &header->sections[TMNF_TRACK_FACES],
			faces_rel, raw->face_count);
		raw->nodes_rel = (uintptr_t)resolve_array(
			base, &header->sections[TMNF_TRACK_NODES],
			nodes_rel, raw->node_count);

		GmSurfMesh *mesh = (GmSurfMesh *)raw;
		for (uint32_t f = 0; f < mesh->face_count; ++f) {
			for (uint32_t v = 0; v < 3; ++v) {
				if (mesh->faces[f].vertex[v] >= mesh->vertex_count) {
					fail("mesh face vertex outside vertex array");
				}
			}
		}
		for (uint32_t n = 0; n < mesh->node_count; ++n) {
			const GmSurfMeshNode *node = &mesh->nodes[n];
			if (node->skip_count == 0 ||
				n + node->skip_count > mesh->node_count ||
				(node->face_index != UINT32_MAX &&
				 node->face_index >= mesh->face_count)) {
				fail("invalid mesh octree node");
			}
		}
	}
}

static void fix_surfaces(uint8_t *base, TmnfTrackHeader *header) {
	const TmnfTrackSection *section =
		&header->sections[TMNF_TRACK_SURFACES];
	TmnfTrackSurface *surfaces =
		(TmnfTrackSurface *)(base + section->offset);
	for (uint32_t i = 0; i < section->count; ++i) {
		TmnfTrackSurface *raw = &surfaces[i];
		uint64_t mesh_rel = raw->mesh_rel;
		uint64_t material_ids_rel = raw->material_ids_rel;
		if (raw->material_count == 0) {
			fail("surface has no material remap");
		}
		raw->mesh_rel = (uintptr_t)resolve_array(
			base, &header->sections[TMNF_TRACK_MESHES], mesh_rel, 1);
		raw->material_ids_rel = (uintptr_t)resolve_array(
			base, &header->sections[TMNF_TRACK_MATERIAL_IDS],
			material_ids_rel, raw->material_count);

		const CPlugSurface *surface = (const CPlugSurface *)raw;
		const GmSurfMesh *mesh = (const GmSurfMesh *)surface->geom;
		for (uint32_t f = 0; f < mesh->face_count; ++f) {
			if (mesh->faces[f].material_index >= surface->material_count) {
				fail("face material outside surface remap");
			}
		}
	}
}

static void fix_entries(uint8_t *base, TmnfTrackHeader *header) {
	const TmnfTrackSection *section =
		&header->sections[TMNF_TRACK_ENTRIES];
	TmnfTrackStaticEntry *entries =
		(TmnfTrackStaticEntry *)(base + section->offset);
	for (uint32_t i = 0; i < section->count; ++i) {
		TmnfTrackStaticEntry *raw = &entries[i];
		if (raw->skip_count == 0 ||
			i + raw->skip_count > section->count) {
			fail("invalid flattened static entry");
		}
		uint64_t surface_rel = raw->surface_rel;
		raw->surface_rel = (uintptr_t)resolve_array(
			base, &header->sections[TMNF_TRACK_SURFACES],
			surface_rel, 1);
	}
}

/* The water map comes from the snapshot as the game holds it (CHmsZone
 * +0x154 map, +0x178 level, +0x17c floor). It is independent of the
 * material-13 collision faces: Stadium's set cells are exactly its water
 * blocks, Desert has set cells without faces, Snow has no water (default
 * cell 1, level == floor), and Island has water faces at 19 and 59 m beside
 * its sea at 35 m, some in unset cells. */

static void derive_water(TmnfTrack *track, const uint8_t *base,
	const TmnfTrackHeader *header) {
	TmnfTrackWater *water = &track->water;
	const TmnfTrackWaterHeader *raw = (const TmnfTrackWaterHeader *)(
		base + header->sections[TMNF_TRACK_WATER].offset);
	const uint8_t *cells =
		base + header->sections[TMNF_TRACK_WATER_CELLS].offset;
	if (header->sections[TMNF_TRACK_WATER].count != 1 ||
		raw->cell_x != raw->cell_z || !(raw->cell_x > 0.0f) ||
		raw->origin_x != 0.0f || raw->origin_z != 0.0f ||
		raw->width == 0 || raw->width > 256 ||
		raw->height == 0 || raw->height > 256 ||
		raw->default_cell > 1 || raw->reserved != 0 ||
		!(raw->floor <= raw->level) ||
		header->sections[TMNF_TRACK_WATER_CELLS].count !=
			raw->width * raw->height) {
		fail("invalid water map");
	}
	memset(water, 0, sizeof(*water));
	water->cell_x = raw->cell_x;
	water->cell_z = raw->cell_z;
	water->origin_x = raw->origin_x;
	water->origin_z = raw->origin_z;
	water->width = raw->width;
	water->height = raw->height;
	water->default_cell = (uint8_t)raw->default_cell;
	water->cells = cells;
	water->level = raw->level;
	water->floor = raw->floor;
	for (uint32_t i = 0; i < raw->width * raw->height; ++i) {
		if (cells[i] > 1) {
			fail("water map cell is not a flag");
		}
		water->cell_count += cells[i];
	}
}

/* Static collision grid.
 *
 * The game tests a query box against the flattened static tree in DFS order
 * with skip counts, then against every passing leaf's mesh tree the same way.
 * Both scans are order-sensitive (contacts are appended in visit order and the
 * response sort is not a total order), so the acceleration below never changes
 * which nodes are tested or in what order for a given query. It only removes
 * nodes whose test outcome is already decided for every query box that lies in
 * a grid cell's region:
 *
 *   - a node whose box does not overlap the region (in exact arithmetic)
 *     fails the float test for every such query, and so does its subtree;
 *   - a non-leaf node whose box contains the region with margin passes for
 *     every such query and is dropped from the list, its children stay.
 *
 * The float AABB test |c1 - c2| <= h1 + h2 has relative error below 2^-22, so
 * a margin of 1e-6 times the largest coordinate plus 5 cm covers it. The cell
 * region is the cell expanded by GRID_EXPAND, which must exceed the half
 * extent of any accelerated query box; the runtime check in collision.c falls
 * back to the full scan otherwise.
 *
 * The same idea is applied a second time inside each mesh, in mesh space,
 * where the game tests the query box (the ellipsoid's box mapped through the
 * relative transform): a uniform grid per mesh with MESH_GRID_CELL-sized
 * cells, whose lists are built for the cell expanded by MESH_GRID_EXPAND. Two
 * levels serve wheel-sized and body-sized boxes. collision.c checks the
 * mesh-space box against the list's stored region before using it, so the
 * budgets only decide how often the restricted lists apply, never the result.
 *
 * A leaf of the static tree whose mesh has no reachable node for the cell's
 * region (mapped into mesh space with slack for the float transforms, since
 * the runtime mesh-space box derives from the world box through float
 * arithmetic) is marked empty and the pair is skipped outright. That needs
 * the mesh-space box to lie inside the mapped region: the ellipsoid's world
 * box lies inside the cell region (checked at runtime), its mesh-space box is
 * the AABB of the ellipsoid's box rotated by the relative transform, and that
 * rotated box lies inside the rotated world box, whose AABB lies inside the
 * mapped region. GRID_MESH_EXPAND adds one metre of headroom on top.
 *
 * Cells with identical list content share storage (most cells above or
 * beside the track see only the same few unbounded planes). */
#define GRID_CELL 16.0
#define GRID_EXPAND 2.5
#define GRID_MESH_EXPAND (GRID_EXPAND + 1.0)
#define GRID_MESH_REGION_MARGIN 0.5
#define GRID_BOUNDS_LEAF_HALF 64.0
#define GRID_MAX_CELLS (1u << 20)
#define GRID_ORTHONORMAL_TOLERANCE 1e-5
static const double MESH_GRID_CELL[TMNF_MESH_GRID_LEVELS] = { 1.0, 2.0 };
static const double MESH_GRID_EXPAND[TMNF_MESH_GRID_LEVELS] = { 0.75, 2.25 };
#define MESH_GRID_MAX_CELLS (1u << 16)

typedef struct {
	double lo[3];
	double hi[3];
} DBox;

typedef struct {
	void *data;
	size_t count;
	size_t capacity;
	size_t stride;
} Vec;

static void vec_reserve(Vec *vec, size_t count) {
	if (count <= vec->capacity) {
		return;
	}
	size_t capacity = vec->capacity ? vec->capacity : 256;
	while (capacity < count) {
		capacity *= 2;
	}
	void *data = realloc(vec->data, capacity * vec->stride);
	if (data == NULL) {
		fail("grid allocation failed");
	}
	vec->data = data;
	vec->capacity = capacity;
}

static void *vec_push(Vec *vec) {
	vec_reserve(vec, vec->count + 1);
	return (uint8_t *)vec->data + vec->count++ * vec->stride;
}

static void *vec_at(const Vec *vec, size_t index) {
	return (uint8_t *)vec->data + index * vec->stride;
}

/* Content-addressed store of variable-length lists inside a Vec. */
typedef struct {
	uint32_t *slots;        /* offsets into the owning Vec, UINT32_MAX empty */
	uint32_t *lengths;
	size_t capacity;
	size_t count;
} ListIndex;

static uint64_t hash_bytes(const void *data, size_t size) {
	const uint8_t *bytes = data;
	uint64_t hash = UINT64_C(0xcbf29ce484222325);
	for (size_t i = 0; i < size; ++i) {
		hash ^= bytes[i];
		hash *= UINT64_C(0x100000001b3);
	}
	return hash;
}

static void list_index_grow(ListIndex *index, const Vec *pool) {
	size_t capacity = index->capacity ? index->capacity * 2 : 4096;
	uint32_t *slots = malloc(capacity * sizeof(*slots));
	uint32_t *lengths = malloc(capacity * sizeof(*lengths));
	if (slots == NULL || lengths == NULL) {
		fail("grid allocation failed");
	}
	memset(slots, 0xff, capacity * sizeof(*slots));
	for (size_t i = 0; i < index->capacity; ++i) {
		if (index->slots[i] == UINT32_MAX) {
			continue;
		}
		uint64_t hash = hash_bytes(
			vec_at(pool, index->slots[i]),
			(size_t)index->lengths[i] * pool->stride);
		size_t at = hash & (capacity - 1);
		while (slots[at] != UINT32_MAX) {
			at = (at + 1) & (capacity - 1);
		}
		slots[at] = index->slots[i];
		lengths[at] = index->lengths[i];
	}
	free(index->slots);
	free(index->lengths);
	index->slots = slots;
	index->lengths = lengths;
	index->capacity = capacity;
}

/* The candidate list occupies pool[start .. count). Returns the offset of an
 * identical stored list, truncating the pool back to start, or keeps the new
 * list and returns start. */
static uint32_t list_index_intern(
	ListIndex *index, Vec *pool, size_t start) {
	size_t length = pool->count - start;
	size_t bytes = length * pool->stride;
	if (index->count * 2 >= index->capacity) {
		list_index_grow(index, pool);
	}
	uint64_t hash = hash_bytes(vec_at(pool, start), bytes);
	size_t at = hash & (index->capacity - 1);
	while (index->slots[at] != UINT32_MAX) {
		if (index->lengths[at] == length &&
			memcmp(vec_at(pool, index->slots[at]), vec_at(pool, start),
				bytes) == 0) {
			pool->count = start;
			return index->slots[at];
		}
		at = (at + 1) & (index->capacity - 1);
	}
	if (start > UINT32_MAX || length > UINT32_MAX) {
		fail("static grid too large");
	}
	index->slots[at] = (uint32_t)start;
	index->lengths[at] = (uint32_t)length;
	index->count++;
	return (uint32_t)start;
}

static void list_index_free(ListIndex *index) {
	free(index->slots);
	free(index->lengths);
}

static int box_overlaps(const GmBoxAligned *box, const DBox *region) {
	const float *c = &box->center.x;
	const float *h = &box->half_extent.x;
	for (int a = 0; a < 3; ++a) {
		if (!((double)c[a] - (double)h[a] <= region->hi[a] &&
			region->lo[a] <= (double)c[a] + (double)h[a])) {
			return 0;
		}
	}
	return 1;
}

static int box_contains(
	const GmBoxAligned *box, const DBox *region, double margin) {
	const float *c = &box->center.x;
	const float *h = &box->half_extent.x;
	for (int a = 0; a < 3; ++a) {
		if (!((double)c[a] - (double)h[a] + margin <= region->lo[a] &&
			region->hi[a] <= (double)c[a] + (double)h[a] - margin)) {
			return 0;
		}
	}
	return 1;
}

static double box_reach(const GmBoxAligned *box) {
	const float *c = &box->center.x;
	const float *h = &box->half_extent.x;
	double reach = 0.0;
	for (int a = 0; a < 3; ++a) {
		double value = fabs((double)c[a]) + (double)h[a];
		if (value > reach) {
			reach = value;
		}
	}
	return reach;
}

static double rounding_margin(double reach) {
	return 0.05 + 1e-6 * reach;
}

/* Fixes list-local skip counts: a frame is (source subtree end, list index
 * of the node, or SIZE_MAX for a node that was dropped). */
typedef struct {
	uint32_t end;
	size_t list_index;
} SkipFrame;

static void close_frames(
	SkipFrame *frames, uint32_t *depth, uint32_t source_index,
	Vec *list, size_t skip_offset) {
	while (*depth > 0 && source_index >= frames[*depth - 1].end) {
		SkipFrame frame = frames[--*depth];
		if (frame.list_index != SIZE_MAX) {
			uint32_t *skip = (uint32_t *)(
				(uint8_t *)vec_at(list, frame.list_index) + skip_offset);
			*skip = (uint32_t)(list->count - frame.list_index);
		}
	}
}

static void push_frame(
	SkipFrame *frames, uint32_t *depth, uint32_t end, size_t list_index) {
	if (*depth >= 256) {
		fail("collision tree deeper than 256");
	}
	frames[*depth].end = end;
	frames[*depth].list_index = list_index;
	(*depth)++;
}

static int iso_is_orthonormal(const GmIso4 *iso) {
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			double dot = 0.0;
			for (int k = 0; k < 3; ++k) {
				dot += (double)iso->m[k * 3 + i] *
					(double)iso->m[k * 3 + j];
			}
			if (fabs(dot - (i == j ? 1.0 : 0.0)) >
				GRID_ORTHONORMAL_TOLERANCE) {
				return 0;
			}
		}
	}
	return 1;
}

typedef struct {
	Vec nodes;              /* TmnfStaticCellNode */
	ListIndex node_lists;
	Vec mesh_pool;          /* TmnfMeshPoolSlot */
	ListIndex mesh_lists;
	double *mesh_margins;   /* per track mesh */
	TmnfMeshGrid *mesh_grids;  /* per track mesh, built on first use */
	uint8_t *mesh_grid_built;
} GridBuilder;

/* Restricts a mesh's node array to a mesh-space region and interns the list.
 * Returns the header slot. */
static uint32_t build_mesh_list(
	GridBuilder *builder, const GmSurfMesh *mesh, const DBox *region,
	double margin) {
	Vec *pool = &builder->mesh_pool;
	size_t start = pool->count;
	TmnfMeshPoolSlot *header = vec_push(pool);
	memset(header, 0, sizeof(*header));

	SkipFrame frames[256];
	uint32_t depth = 0;
	uint32_t i = 0;
	while (i < mesh->node_count) {
		close_frames(
			frames, &depth, i, pool, offsetof(TmnfMeshListEntry, skip));
		const GmSurfMeshNode *source = &mesh->nodes[i];
		if (!box_overlaps(&source->box, region)) {
			i += source->skip_count;
			continue;
		}
		int is_leaf = source->face_index != UINT32_MAX;
		size_t list_index = SIZE_MAX;
		if (is_leaf || !box_contains(&source->box, region, margin)) {
			TmnfMeshPoolSlot *slot = vec_push(pool);
			slot->entry.node = i;
			slot->entry.skip = 1;
			list_index = pool->count - 1;
		}
		if (source->skip_count > 1) {
			push_frame(
				frames, &depth, i + source->skip_count, list_index);
		}
		i++;
	}
	close_frames(
		frames, &depth, UINT32_MAX, pool,
		offsetof(TmnfMeshListEntry, skip));

	header = vec_at(pool, start);
	header->node_count = (uint32_t)(pool->count - start - 1);
	return list_index_intern(&builder->mesh_lists, pool, start);
}

/* True when some leaf of the mesh tree is reached by the game's scan for a
 * query box inside the region: every ancestor and the leaf overlap it. */
static int mesh_reaches_region(const GmSurfMesh *mesh, const DBox *region) {
	uint32_t i = 0;
	while (i < mesh->node_count) {
		const GmSurfMeshNode *node = &mesh->nodes[i];
		if (!box_overlaps(&node->box, region)) {
			i += node->skip_count;
			continue;
		}
		if (node->face_index != UINT32_MAX) {
			return 1;
		}
		i++;
	}
	return 0;
}

/* Builds the per-level uniform grids over one mesh. Each level's cell list
 * region is the cell expanded by the level's query half-extent budget. */
static void build_mesh_grid(
	GridBuilder *builder, const GmSurfMesh *mesh, double margin,
	TmnfMeshGrid *grid) {
	DBox bounds;
	for (int a = 0; a < 3; ++a) {
		bounds.lo[a] = INFINITY;
		bounds.hi[a] = -INFINITY;
	}
	for (uint32_t i = 0; i < mesh->node_count; ++i) {
		const float *c = &mesh->nodes[i].box.center.x;
		const float *h = &mesh->nodes[i].box.half_extent.x;
		for (int a = 0; a < 3; ++a) {
			if ((double)c[a] - (double)h[a] < bounds.lo[a]) {
				bounds.lo[a] = (double)c[a] - (double)h[a];
			}
			if ((double)c[a] + (double)h[a] > bounds.hi[a]) {
				bounds.hi[a] = (double)c[a] + (double)h[a];
			}
		}
	}
	memset(grid, 0, sizeof(*grid));
	if (mesh->face_count != 0) {
		TmnfSphereFaceEdges *edges = malloc((size_t)mesh->face_count * sizeof(*edges));
		if (edges == NULL)
			fail("sphere edge cache allocation failed");
		for (uint32_t i = 0; i < mesh->face_count; ++i) {
			const GmSurfMeshFace *face = &mesh->faces[i];
			for (uint32_t v = 0; v < 4; ++v) {
				const GmVec3 *vertex = &mesh->vertices[face->vertex[v % 3]];
				edges[i].xyz[0][v] = vertex->x;
				edges[i].xyz[1][v] = vertex->y;
				edges[i].xyz[2][v] = vertex->z;
			}
			edges[i].padding[0] = edges[i].padding[1] = 0.0f;
			for (uint32_t e = 0; e < 3; ++e) {
				edges[i].edges[e] = TmnfSphereMeshEdge_Prepare(
					&mesh->vertices[face->vertex[e]],
					&mesh->vertices[face->vertex[(e + 1) % 3]], &face->normal);
			}
		}
		grid->sphere_edges = edges;
	}
	for (int l = 0; l < TMNF_MESH_GRID_LEVELS; ++l) {
		TmnfMeshGridLevel *level = &grid->levels[l];
		double expand = MESH_GRID_EXPAND[l];
		double cell_size = MESH_GRID_CELL[l];
		uint32_t dims[3];
		for (;;) {
			uint64_t total = 1;
			for (int a = 0; a < 3; ++a) {
				double extent = bounds.hi[a] - bounds.lo[a] +
					2.0 * (expand + cell_size);
				dims[a] = (uint32_t)floor(extent / cell_size) + 1;
				total *= dims[a];
			}
			if (total <= MESH_GRID_MAX_CELLS) {
				break;
			}
			cell_size *= 2.0;
		}
		level->cell_size = cell_size;
		level->inv_cell_size = 1.0 / cell_size;
		level->max_half = expand - margin;
		uint32_t cell_count = 1;
		for (int a = 0; a < 3; ++a) {
			level->origin[a] = bounds.lo[a] - expand - cell_size;
			level->dims[a] = dims[a];
			cell_count *= dims[a];
		}
		level->cell_count = cell_count;
		uint32_t *cells = malloc((size_t)cell_count * sizeof(*cells));
		if (cells == NULL) {
			fail("grid allocation failed");
		}
		for (uint32_t cell = 0; cell < cell_count; ++cell) {
			uint32_t index[3] = {
				cell % dims[0],
				(cell / dims[0]) % dims[1],
				cell / (dims[0] * dims[1]),
			};
			DBox region;
			for (int a = 0; a < 3; ++a) {
				region.lo[a] = level->origin[a] + index[a] * cell_size -
					expand;
				region.hi[a] = region.lo[a] + cell_size + 2.0 * expand;
			}
			cells[cell] = build_mesh_list(builder, mesh, &region, margin);
		}
		level->cells = cells;
	}
}

/* The mesh grid for a leaf entry, or NULL when its surface is not a mesh or
 * its placement is not a rotation (mesh-space boxes then need no rescaling
 * argument; see the header comment). */
static const TmnfMeshGrid *entry_mesh_grid(
	GridBuilder *builder, const TmnfTrack *track,
	const HmsStaticCollisionEntry *entry) {
	const CPlugSurface *surface = entry->surface;
	if (surface->geom->type != GM_SURF_MESH ||
		!iso_is_orthonormal(&entry->iso)) {
		return NULL;
	}
	const GmSurfMesh *mesh = (const GmSurfMesh *)surface->geom;
	size_t m = (size_t)(mesh - track->meshes);
	if (!builder->mesh_grid_built[m]) {
		build_mesh_grid(
			builder, mesh, builder->mesh_margins[m],
			&builder->mesh_grids[m]);
		builder->mesh_grid_built[m] = 1;
	}
	return &builder->mesh_grids[m];
}

/* True when a query box inside world_region can reach a leaf of the entry's
 * mesh tree: the region is mapped into mesh space (an AABB of its corners
 * with slack for the float transforms at runtime) and scanned. */
static int entry_reaches_region(
	const TmnfTrack *track, const HmsStaticCollisionEntry *entry,
	const DBox *world_region) {
	const CPlugSurface *surface = entry->surface;
	if (surface->geom->type != GM_SURF_MESH ||
		!iso_is_orthonormal(&entry->iso)) {
		return 1;
	}
	(void)track;
	const GmSurfMesh *mesh = (const GmSurfMesh *)surface->geom;
	const GmIso4 *iso = &entry->iso;
	DBox region;
	for (int a = 0; a < 3; ++a) {
		region.lo[a] = INFINITY;
		region.hi[a] = -INFINITY;
	}
	for (int corner = 0; corner < 8; ++corner) {
		double w[3];
		for (int a = 0; a < 3; ++a) {
			w[a] = ((corner >> a) & 1) ? world_region->hi[a] :
				world_region->lo[a];
			w[a] -= (double)iso->t[a];
		}
		for (int a = 0; a < 3; ++a) {
			double m = (double)iso->m[0 * 3 + a] * w[0] +
				(double)iso->m[1 * 3 + a] * w[1] +
				(double)iso->m[2 * 3 + a] * w[2];
			if (m < region.lo[a]) {
				region.lo[a] = m;
			}
			if (m > region.hi[a]) {
				region.hi[a] = m;
			}
		}
	}
	for (int a = 0; a < 3; ++a) {
		region.lo[a] -= GRID_MESH_REGION_MARGIN;
		region.hi[a] += GRID_MESH_REGION_MARGIN;
	}
	return mesh_reaches_region(mesh, &region);
}

/* Drops the leaves no query in the cell can hit, and the internal nodes left
 * without a leaf below them, from the list built from `start`. A dropped
 * node's box test could only ever have led to a skipped pair, so the visit
 * sequence of the remaining nodes is unchanged. */
static void prune_empty_cell_nodes(Vec *nodes, size_t start) {
	uint32_t count = (uint32_t)(nodes->count - start);
	TmnfStaticCellNode *list = vec_at(nodes, start);
	uint32_t *kept_below = malloc((size_t)(count + 1) * sizeof(*kept_below));
	if (kept_below == NULL) {
		fail("grid allocation failed");
	}
	/* kept_below[j] = kept nodes in list[j..count): walking backwards, a
	 * subtree keeps an internal node iff a leaf inside it is kept. */
	kept_below[count] = 0;
	for (uint32_t j = count; j-- > 0;) {
		int keep;
		if (list[j].entry_index == TMNF_CELL_NODE_INTERNAL) {
			keep = kept_below[j + 1] - kept_below[j + list[j].skip] > 0;
		} else {
			keep = list[j].entry_index != TMNF_CELL_NODE_EMPTY;
		}
		kept_below[j] = kept_below[j + 1] + (keep ? 1u : 0u);
	}
	uint32_t out = 0;
	for (uint32_t j = 0; j < count; ++j) {
		if (kept_below[j] == kept_below[j + 1]) {
			continue;
		}
		TmnfStaticCellNode node = list[j];
		node.skip = kept_below[j] - kept_below[j + node.skip];
		list[out++] = node;
	}
	free(kept_below);
	nodes->count = start + out;
}

static void build_cell_list(
	GridBuilder *builder, const TmnfTrack *track, const DBox *cell,
	double margin, uint32_t *offset, uint32_t *count) {
	DBox region;
	DBox mesh_region;
	for (int a = 0; a < 3; ++a) {
		region.lo[a] = cell->lo[a] - GRID_EXPAND;
		region.hi[a] = cell->hi[a] + GRID_EXPAND;
		mesh_region.lo[a] = cell->lo[a] - GRID_MESH_EXPAND;
		mesh_region.hi[a] = cell->hi[a] + GRID_MESH_EXPAND;
	}

	Vec *nodes = &builder->nodes;
	size_t start = nodes->count;
	SkipFrame frames[256];
	uint32_t depth = 0;
	uint32_t i = 0;
	while (i < track->entry_count) {
		close_frames(
			frames, &depth, i, nodes, offsetof(TmnfStaticCellNode, skip));
		const HmsStaticCollisionEntry *entry = &track->entries[i];
		if (!box_overlaps(&entry->box, &region)) {
			i += entry->skip_count;
			continue;
		}
		int is_leaf = entry->surface != NULL &&
			(entry->tree_flags & 0x80u) != 0;
		size_t list_index = SIZE_MAX;
		if (is_leaf || !box_contains(&entry->box, &region, margin)) {
			uint32_t entry_index = TMNF_CELL_NODE_INTERNAL;
			if (is_leaf) {
				entry_index = entry_reaches_region(
					track, entry, &mesh_region) ? i :
					TMNF_CELL_NODE_EMPTY;
			}
			TmnfStaticCellNode *node = vec_push(nodes);
			memset(node, 0, sizeof(*node));
			node->box = entry->box;
			node->skip = 1;
			node->entry_index = entry_index;
			list_index = nodes->count - 1;
		}
		if (entry->skip_count > 1) {
			push_frame(frames, &depth, i + entry->skip_count, list_index);
		}
		i++;
	}
	close_frames(
		frames, &depth, UINT32_MAX, nodes,
		offsetof(TmnfStaticCellNode, skip));
	prune_empty_cell_nodes(nodes, start);
	*count = (uint32_t)(nodes->count - start);
	*offset = list_index_intern(&builder->node_lists, nodes, start);
}

static void build_static_grid(TmnfTrack *track) {
	TmnfStaticGrid *grid = &track->grid;
	memset(grid, 0, sizeof(*grid));

	DBox bounds;
	for (int a = 0; a < 3; ++a) {
		bounds.lo[a] = INFINITY;
		bounds.hi[a] = -INFINITY;
	}
	double reach = 0.0;
	uint32_t bounded_leaves = 0;
	for (uint32_t i = 0; i < track->entry_count; ++i) {
		const HmsStaticCollisionEntry *entry = &track->entries[i];
		double value = box_reach(&entry->box);
		if (value > reach) {
			reach = value;
		}
		if (entry->surface == NULL || (entry->tree_flags & 0x80u) == 0) {
			continue;
		}
		const float *c = &entry->box.center.x;
		const float *h = &entry->box.half_extent.x;
		if (h[0] > GRID_BOUNDS_LEAF_HALF || h[1] > GRID_BOUNDS_LEAF_HALF ||
			h[2] > GRID_BOUNDS_LEAF_HALF) {
			continue;
		}
		bounded_leaves++;
		for (int a = 0; a < 3; ++a) {
			if ((double)c[a] - (double)h[a] < bounds.lo[a]) {
				bounds.lo[a] = (double)c[a] - (double)h[a];
			}
			if ((double)c[a] + (double)h[a] > bounds.hi[a]) {
				bounds.hi[a] = (double)c[a] + (double)h[a];
			}
		}
	}
	if (bounded_leaves == 0) {
		return;
	}
	if (!isfinite(reach)) {
		fail("static collision tree has non-finite boxes");
	}

	double cell_size = GRID_CELL;
	uint32_t dims[3];
	for (;;) {
		uint64_t total = 1;
		for (int a = 0; a < 3; ++a) {
			double extent = bounds.hi[a] - bounds.lo[a];
			dims[a] = (uint32_t)floor(extent / cell_size) + 1;
			total *= dims[a];
		}
		if (total <= GRID_MAX_CELLS) {
			break;
		}
		cell_size *= 2.0;
	}

	grid->cell_size = cell_size;
	grid->inv_cell_size = 1.0 / cell_size;
	grid->expand = GRID_EXPAND;
	grid->margin = rounding_margin(reach);
	for (int a = 0; a < 3; ++a) {
		grid->origin[a] = bounds.lo[a];
		grid->dims[a] = dims[a];
	}
	grid->cell_count = dims[0] * dims[1] * dims[2];

	GmIso4 *inverse_isos = malloc(
		(size_t)track->entry_count * sizeof(*inverse_isos));
	const TmnfMeshGrid **entry_mesh_grids = malloc(
		(size_t)track->entry_count * sizeof(*entry_mesh_grids));
	uint32_t *offsets = malloc(
		(size_t)grid->cell_count * sizeof(*offsets));
	uint32_t *counts = malloc(
		(size_t)grid->cell_count * sizeof(*counts));
	GridBuilder builder = {
		.nodes = { .stride = sizeof(TmnfStaticCellNode) },
		.mesh_pool = { .stride = sizeof(TmnfMeshPoolSlot) },
		.mesh_margins = malloc(
			(size_t)track->mesh_count * sizeof(*builder.mesh_margins)),
		.mesh_grids = calloc(
			track->mesh_count, sizeof(*builder.mesh_grids)),
		.mesh_grid_built = calloc(track->mesh_count, 1),
	};
	if (inverse_isos == NULL || entry_mesh_grids == NULL ||
		offsets == NULL || counts == NULL ||
		builder.mesh_margins == NULL || builder.mesh_grids == NULL ||
		builder.mesh_grid_built == NULL) {
		fail("grid allocation failed");
	}
	for (uint32_t m = 0; m < track->mesh_count; ++m) {
		const GmSurfMesh *mesh = &track->meshes[m];
		double mesh_reach = 0.0;
		for (uint32_t i = 0; i < mesh->node_count; ++i) {
			double value = box_reach(&mesh->nodes[i].box);
			if (value > mesh_reach) {
				mesh_reach = value;
			}
		}
		if (!isfinite(mesh_reach)) {
			fail("mesh collision tree has non-finite boxes");
		}
		builder.mesh_margins[m] = rounding_margin(mesh_reach);
	}
	for (uint32_t i = 0; i < track->entry_count; ++i) {
		const HmsStaticCollisionEntry *entry = &track->entries[i];
		GmIso4_SetInverse(&inverse_isos[i], &entry->iso);
		entry_mesh_grids[i] = NULL;
		if (entry->surface != NULL && (entry->tree_flags & 0x80u) != 0) {
			entry_mesh_grids[i] = entry_mesh_grid(&builder, track, entry);
		}
	}

	for (uint32_t cell = 0; cell < grid->cell_count; ++cell) {
		uint32_t index[3] = {
			cell % dims[0],
			(cell / dims[0]) % dims[1],
			cell / (dims[0] * dims[1]),
		};
		DBox box;
		for (int a = 0; a < 3; ++a) {
			box.lo[a] = grid->origin[a] + index[a] * cell_size;
			box.hi[a] = box.lo[a] + cell_size;
		}
		build_cell_list(
			&builder, track, &box, grid->margin,
			&offsets[cell], &counts[cell]);
	}

	for (uint32_t m = 0; m < track->mesh_count; ++m) {
		builder.mesh_grids[m].pool = builder.mesh_pool.data;
	}
	grid->cell_offsets = offsets;
	grid->cell_counts = counts;
	grid->nodes = builder.nodes.data;
	grid->mesh_pool = builder.mesh_pool.data;
	grid->mesh_grids = builder.mesh_grids;
	grid->entry_mesh_grids = entry_mesh_grids;
	grid->entry_inverse_isos = inverse_isos;
	track->grid_node_count = builder.nodes.count;
	track->grid_mesh_slot_count = builder.mesh_pool.count;
	list_index_free(&builder.node_lists);
	list_index_free(&builder.mesh_lists);
	free(builder.mesh_margins);
	free(builder.mesh_grid_built);
}

static void free_static_grid(TmnfTrack *track) {
	const TmnfStaticGrid *grid = &track->grid;
	if (grid->cell_count == 0) {
		return;
	}
	for (uint32_t m = 0; m < track->mesh_count; ++m) {
		free((void *)grid->mesh_grids[m].sphere_edges);
		for (int l = 0; l < TMNF_MESH_GRID_LEVELS; ++l) {
			free((void *)grid->mesh_grids[m].levels[l].cells);
		}
	}
	free((void *)grid->mesh_grids);
	free((void *)grid->entry_mesh_grids);
	free((void *)grid->cell_offsets);
	free((void *)grid->cell_counts);
	free((void *)grid->nodes);
	free((void *)grid->mesh_pool);
	free((void *)grid->entry_inverse_isos);
}

/* CHmsCorpus::GetLocation (0x005474A0) is corpus+0x18, not the flattened tree
 * iso of the entry, so the body iso is the corpus iso. */
static void build_static_response_bodies(TmnfTrack *track) {
	uint32_t maximum = 0;
	for (uint32_t i = 0; i < track->entry_count; ++i) {
		if (maximum < track->entries[i].corpus_ref) {
			maximum = track->entries[i].corpus_ref;
		}
	}
	track->static_response_count = maximum + 1;
	track->static_response_bodies = calloc(
		track->static_response_count,
		sizeof(*track->static_response_bodies));
	track->static_response_present = calloc(track->static_response_count, 1);
	if (track->static_response_bodies == NULL ||
		track->static_response_present == NULL) {
		fail("response body allocation failed");
	}
	for (uint32_t i = 0; i < track->entry_count; ++i) {
		uint32_t id = track->entries[i].corpus_ref;
		if (id == 0 || track->static_response_present[id] != 0) {
			continue;
		}
		CHmsResponseBody *body = &track->static_response_bodies[id];
		track->static_response_present[id] = 1;
		body->corpus_ref = id;
		body->classification_flags = 0x11888101u;
		body->response_flags = 0xfff18002u;
		body->response_weight = 1.0f;
		body->iso = track->corpus_isos[id - 1];
	}
}

TmnfTrack *TmnfTrack_Load(
	const char *path, const uint8_t expected_track_sha256[32]) {
	int fd = open(path, O_RDONLY);
	if (fd < 0) {
		fail("cannot open snapshot");
	}
	struct stat st;
	if (fstat(fd, &st) != 0 ||
		st.st_size < (off_t)sizeof(TmnfTrackHeader)) {
		fail("invalid snapshot size");
	}
	size_t size = (size_t)st.st_size;
	uint8_t *base = mmap(
		NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
	close(fd);
	if (base == MAP_FAILED) {
		fail("mmap failed");
	}

	TmnfTrackHeader *header = (TmnfTrackHeader *)base;
	if (memcmp(header->magic, "TMNFTRK1", 8) != 0 ||
		header->version != TMNF_TRACK_VERSION ||
		header->endian != 0x12345678 ||
		header->header_size != sizeof(TmnfTrackHeader) ||
		header->section_count != TMNF_TRACK_SECTION_COUNT ||
		header->file_size != size) {
		fail("invalid snapshot header");
	}
	if (memcmp(header->exe_sha256, TMNF_21126_EXE_SHA256, 32) != 0) {
		fail("snapshot targets a different executable");
	}
	if (memcmp(header->track_sha256, expected_track_sha256, 32) != 0) {
		fail("snapshot is for a different track");
	}
	validate_sections(header, size);
	if (header->sections[TMNF_TRACK_MATERIAL_DATA].count !=
		TMNF_TRACK_MATERIAL_COUNT) {
		fail("physical material table must contain 31 entries");
	}

	uint8_t payload_digest[32];
	Sha256 sha;
	sha256_init(&sha);
	sha256_update(
		&sha, base + header->header_size,
		size - header->header_size);
	sha256_final(&sha, payload_digest);
	if (memcmp(payload_digest, header->payload_sha256, 32) != 0) {
		fail("payload SHA-256 mismatch");
	}

	fix_meshes(base, header);
	fix_surfaces(base, header);
	fix_entries(base, header);

	TmnfTrack *track = calloc(1, sizeof(*track));
	if (track == NULL) {
		fail("track allocation failed");
	}
	track->mapping = base;
	track->mapping_size = size;
	track->header = header;
	track->entries = (const HmsStaticCollisionEntry *)(
		base + header->sections[TMNF_TRACK_ENTRIES].offset);
	track->entry_count = header->sections[TMNF_TRACK_ENTRIES].count;
	track->surfaces = (const CPlugSurface *)(
		base + header->sections[TMNF_TRACK_SURFACES].offset);
	track->surface_count = header->sections[TMNF_TRACK_SURFACES].count;
	track->meshes = (const GmSurfMesh *)(
		base + header->sections[TMNF_TRACK_MESHES].offset);
	track->mesh_count = header->sections[TMNF_TRACK_MESHES].count;
	track->materials = (const TmnfTrackMaterialData *)(
		base + header->sections[TMNF_TRACK_MATERIAL_DATA].offset);
	track->collision_pairs = (const TmnfTrackCollisionPair *)(
		base + header->sections[TMNF_TRACK_COLLISION_PAIRS].offset);
	track->collision_pair_count =
		header->sections[TMNF_TRACK_COLLISION_PAIRS].count;
	track->corpus_isos = (const GmIso4 *)(
		base + header->sections[TMNF_TRACK_CORPUS_ISOS].offset);
	track->corpus_count = header->sections[TMNF_TRACK_CORPUS_ISOS].count;
	for (uint32_t i = 0; i < track->entry_count; ++i) {
		if (track->entries[i].corpus_ref > track->corpus_count) {
			fail("static entry references a missing corpus location");
		}
	}
	derive_water(track, base, header);
	build_static_grid(track);
	build_static_response_bodies(track);

	if (mprotect(base, size, PROT_READ) != 0) {
		fail("mprotect failed");
	}
	return track;
}

void TmnfTrack_BindStaticGroup(
	const TmnfTrack *track, CHmsCollisionManager_SGroup *group) {
	group->is_static = 1;
	group->static_entry_count = track->entry_count;
	group->static_entries =
		(HmsStaticCollisionEntry *)(uintptr_t)track->entries;
	group->static_grid = track->grid.cell_count != 0 ? &track->grid : NULL;
}

void TmnfTrack_Unload(TmnfTrack *track) {
	if (munmap(track->mapping, track->mapping_size) != 0) {
		fail("munmap failed");
	}
	free_static_grid(track);
	free(track->static_response_present);
	free(track->static_response_bodies);
	free(track);
}
