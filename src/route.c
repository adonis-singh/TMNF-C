#include "route.h"

#include <fcntl.h>
#include <math.h>
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

TMNF_HD static void fail(const char *message) {
	tmnf_fail(message);
}

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
	const uint8_t *bytes = (const uint8_t *)data;
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

static uint64_t section_bytes(const TmnfRouteSection *section) {
	uint64_t bytes = (uint64_t)section->count * section->stride;
	if (section->stride != 0 &&
		bytes / section->stride != section->count) {
		fail("section size overflow");
	}
	return bytes;
}

static void validate_sections(
	const TmnfRouteHeader *header, size_t mapping_size) {
	static const uint32_t expected_stride[TMNF_ROUTE_SECTION_COUNT] = {
		sizeof(TmnfRouteMetadata),
		sizeof(TmnfRouteStart),
		sizeof(TmnfRouteTrigger),
		sizeof(TmnfRouteTrigger),
		sizeof(TmnfRouteReferencePoint),
	};

	for (uint32_t i = 0; i < TMNF_ROUTE_SECTION_COUNT; ++i) {
		const TmnfRouteSection *section = &header->sections[i];
		if (section->stride != expected_stride[i] ||
			(section->count == 0 &&
			 i != TMNF_ROUTE_CHECKPOINTS) ||
			(section->offset & 7) != 0 ||
			section->offset < sizeof(TmnfRouteHeader)) {
			fail("invalid section descriptor");
		}
		uint64_t bytes = section_bytes(section);
		if (section->offset > mapping_size ||
			bytes > mapping_size - section->offset) {
			fail("section outside file");
		}
	}
	for (uint32_t i = 0; i < TMNF_ROUTE_SECTION_COUNT; ++i) {
		uint64_t a0 = header->sections[i].offset;
		uint64_t a1 = a0 + section_bytes(&header->sections[i]);
		for (uint32_t j = i + 1; j < TMNF_ROUTE_SECTION_COUNT; ++j) {
			uint64_t b0 = header->sections[j].offset;
			uint64_t b1 = b0 + section_bytes(&header->sections[j]);
			if (a0 < b1 && b0 < a1) {
				fail("overlapping sections");
			}
		}
	}
}

static void *resolve_array(
	uint8_t *base, const TmnfRouteSection *section,
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

TMNF_HD static int finite_floats(const float *values, uint32_t count) {
	for (uint32_t i = 0; i < count; ++i) {
		if (!isfinite(values[i])) {
			return 0;
		}
	}
	return 1;
}

/* A spawn isometry is a proper rotation (the game builds it from a block
 * placement and a block-info isometry) with a finite translation. */
static int valid_spawn(const GmIso4 *spawn) {
	if (!finite_floats((const float *)spawn, 12)) {
		return 0;
	}
	for (uint32_t row = 0; row < 3; ++row) {
		const float *r = &spawn->m[row * 3];
		float length = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
		if (fabsf(length - 1.0f) > 1e-3f) {
			return 0;
		}
	}
	return 1;
}

static void validate_trigger(
	const TmnfRouteTrigger *trigger, uint32_t index, uint32_t waypoint_type) {
	if (trigger->race_index != index ||
		trigger->waypoint_type != waypoint_type ||
		trigger->tree_flags == 0 ||
		!finite_floats((const float *)&trigger->box, 6) ||
		!finite_floats((const float *)&trigger->transform, 12) ||
		!(trigger->box.half_extent.x > 0.0f) ||
		!(trigger->box.half_extent.y > 0.0f) ||
		!(trigger->box.half_extent.z > 0.0f) ||
		!valid_spawn(&trigger->spawn) ||
		trigger->no_respawn > 1 ||
		trigger->reserved != 0) {
		fail("invalid route trigger");
	}
}

static void fix_and_validate(uint8_t *base, TmnfRouteHeader *header) {
	TmnfRouteMetadata *metadata = (TmnfRouteMetadata *)(
		base + header->sections[TMNF_ROUTE_METADATA].offset);
	if (header->sections[TMNF_ROUTE_METADATA].count != 1 ||
		header->sections[TMNF_ROUTE_START].count != 1 ||
		header->sections[TMNF_ROUTE_FINISH].count != metadata->finish_count ||
		metadata->lap_count == 0 ||
		metadata->finish_count == 0 ||
		(uint64_t)metadata->finish_count + metadata->checkpoint_count > 64 ||
		metadata->checkpoint_count !=
			header->sections[TMNF_ROUTE_CHECKPOINTS].count ||
		metadata->centerline_count !=
			header->sections[TMNF_ROUTE_REFERENCE].count ||
		metadata->reference_count != metadata->checkpoint_count + 2 ||
		metadata->centerline_count < metadata->reference_count ||
		(metadata->flags & ~TMNF_ROUTE_MULTILAP) != 0 ||
		((metadata->flags & TMNF_ROUTE_MULTILAP) != 0) !=
			(metadata->lap_count > 1)) {
		fail("invalid route metadata");
	}

	uint64_t start_rel = metadata->start_rel;
	uint64_t checkpoints_rel = metadata->checkpoints_rel;
	uint64_t finish_rel = metadata->finish_rel;
	uint64_t reference_rel = metadata->reference_rel;
	metadata->start_rel = (uintptr_t)resolve_array(
		base, &header->sections[TMNF_ROUTE_START], start_rel, 1);
	metadata->checkpoints_rel = (uintptr_t)resolve_array(
		base, &header->sections[TMNF_ROUTE_CHECKPOINTS],
		checkpoints_rel, metadata->checkpoint_count);
	metadata->finish_rel = (uintptr_t)resolve_array(
		base, &header->sections[TMNF_ROUTE_FINISH], finish_rel, metadata->finish_count);
	metadata->reference_rel = (uintptr_t)resolve_array(
		base, &header->sections[TMNF_ROUTE_REFERENCE],
		reference_rel, metadata->centerline_count);

	const TmnfRouteStart *start =
		(const TmnfRouteStart *)(uintptr_t)metadata->start_rel;
	const TmnfRouteTrigger *checkpoints =
		(const TmnfRouteTrigger *)(uintptr_t)metadata->checkpoints_rel;
	const TmnfRouteTrigger *finish =
		(const TmnfRouteTrigger *)(uintptr_t)metadata->finish_rel;
	const TmnfRouteReferencePoint *centerline =
		(const TmnfRouteReferencePoint *)(uintptr_t)metadata->reference_rel;
	if ((start->waypoint_type != TMNF_ROUTE_WAYPOINT_START &&
		 start->waypoint_type != TMNF_ROUTE_WAYPOINT_START_FINISH) ||
		!finite_floats((const float *)&start->transform, 12) ||
		!finite_floats((const float *)&start->initial_state, 43) ||
		memcmp(start->transform.m, start->initial_state.rot.m, 0x24) != 0 ||
		memcmp(start->transform.t, &start->initial_state.pos, 0x0C) != 0 ||
		!valid_spawn(&start->spawn)) {
		fail("invalid route start");
	}
	for (uint32_t i = 0; i < metadata->checkpoint_count; ++i) {
		validate_trigger(
			&checkpoints[i], i, TMNF_ROUTE_WAYPOINT_CHECKPOINT);
	}
	for (uint32_t i = 0; i < metadata->finish_count; ++i) {
		if (finish[i].waypoint_type == TMNF_ROUTE_WAYPOINT_FINISH) {
			validate_trigger(&finish[i], i, TMNF_ROUTE_WAYPOINT_FINISH);
			if (start->waypoint_type != TMNF_ROUTE_WAYPOINT_START)
				fail("dedicated finish has a shared start");
		} else {
			validate_trigger(&finish[i], i, TMNF_ROUTE_WAYPOINT_START_FINISH);
			if (metadata->finish_count != 1 ||
				start->waypoint_type != TMNF_ROUTE_WAYPOINT_START_FINISH ||
				start->block_index != finish[i].block_index ||
				metadata->lap_count <= 1)
				fail("invalid shared start/finish");
		}
	}
	if (centerline[0].arc_length != 0.0f ||
		centerline[0].leg_index != 0 ||
		memcmp(&centerline[0].position, start->transform.t, 0x0C) != 0) {
		fail("invalid reference-line origin");
	}
	for (uint32_t i = 0; i < metadata->centerline_count; ++i) {
		if (!finite_floats((const float *)&centerline[i], 5) ||
			!(centerline[i].half_width > 0.0f) ||
			centerline[i].leg_index > metadata->checkpoint_count ||
			(i != 0 &&
			 (centerline[i].leg_index <
				centerline[i - 1].leg_index ||
			  centerline[i].leg_index >
				centerline[i - 1].leg_index + 1)) ||
			(i != 0 &&
			 !(centerline[i].arc_length >
				centerline[i - 1].arc_length))) {
			fail("invalid reference line");
		}
		if (i != 0) {
			float dx = centerline[i].position.x -
				centerline[i - 1].position.x;
			float dy = centerline[i].position.y -
				centerline[i - 1].position.y;
			float dz = centerline[i].position.z -
				centerline[i - 1].position.z;
			float length = sqrtf(dx * dx + dy * dy + dz * dz);
			float arc_delta = centerline[i].arc_length -
				centerline[i - 1].arc_length;
			float tolerance = fmaxf(0.001f, length * 0.0001f);
			if (!(length > 0.0f) ||
				fabsf(arc_delta - length) > tolerance) {
				fail("reference arc length does not match geometry");
			}
		}
	}
	if (centerline[metadata->centerline_count - 1].leg_index !=
		metadata->checkpoint_count) {
		fail("reference line does not contain every route leg");
	}
}

static uint32_t build_projection_bvh(
	TmnfRoute *route, uint32_t first, uint32_t count, uint32_t depth) {
	if (count == 0 || depth >= 64) {
		fail("reference line exceeds BVH depth bound");
	}
	uint32_t node_index = route->projection_bvh_count++;
	TmnfRouteBvhNode *node = &route->projection_bvh[node_index];
	node->minimum[0] = node->minimum[1] = node->minimum[2] = INFINITY;
	node->maximum[0] = node->maximum[1] = node->maximum[2] = -INFINITY;
	for (uint32_t segment = first; segment < first + count; ++segment) {
		const GmVec3 *a = &route->centerline[segment].position;
		const GmVec3 *b = &route->centerline[segment + 1].position;
		const float *pa = (const float *)a;
		const float *pb = (const float *)b;
		for (uint32_t axis = 0; axis < 3; ++axis) {
			node->minimum[axis] = fminf(
				node->minimum[axis], fminf(pa[axis], pb[axis]));
			node->maximum[axis] = fmaxf(
				node->maximum[axis], fmaxf(pa[axis], pb[axis]));
		}
	}
	if (depth + 1 > route->projection_bvh_height) {
		route->projection_bvh_height = depth + 1;
	}
	if (count == 1) {
		node->left = UINT32_MAX;
		node->right = UINT32_MAX;
		node->segment = first;
		return node_index;
	}
	uint32_t left_count = count / 2;
	node->segment = UINT32_MAX;
	node->left = build_projection_bvh(
		route, first, left_count, depth + 1);
	node->right = build_projection_bvh(
		route, first + left_count, count - left_count, depth + 1);
	return node_index;
}

static void initialize_projection_bvh(TmnfRoute *route) {
	uint32_t segment_count = route->metadata->centerline_count - 1;
	if (segment_count > (UINT32_MAX - 1) / 2) {
		fail("reference line is too large for its BVH");
	}
	uint32_t node_count = segment_count * 2 - 1;
	route->projection_bvh = (TmnfRouteBvhNode *)calloc(
		node_count, sizeof(*route->projection_bvh));
	if (route->projection_bvh == NULL) {
		fail("projection BVH allocation failed");
	}
	route->projection_bvh_count = 0;
	route->projection_bvh_height = 0;
	if (build_projection_bvh(route, 0, segment_count, 0) != 0 ||
		route->projection_bvh_count != node_count) {
		fail("projection BVH construction failed");
	}
}

TmnfRoute *TmnfRoute_Load(
	const char *path, const uint8_t expected_track_sha256[32]) {
	int fd = open(path, O_RDONLY);
	if (fd < 0) {
		fail("cannot open snapshot");
	}
	struct stat st;
	if (fstat(fd, &st) != 0 ||
		st.st_size < (off_t)sizeof(TmnfRouteHeader)) {
		fail("invalid snapshot size");
	}
	size_t size = (size_t)st.st_size;
	uint8_t *base = (uint8_t *)mmap(
		NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
	close(fd);
	if (base == MAP_FAILED) {
		fail("mmap failed");
	}

	TmnfRouteHeader *header = (TmnfRouteHeader *)base;
	if (memcmp(header->magic, "TMNFROU1", 8) != 0 ||
		header->version != TMNF_ROUTE_VERSION ||
		header->endian != 0x12345678 ||
		header->header_size != sizeof(TmnfRouteHeader) ||
		header->section_count != TMNF_ROUTE_SECTION_COUNT ||
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
	fix_and_validate(base, header);

	TmnfRoute *route = (TmnfRoute *)calloc(1, sizeof(*route));
	if (route == NULL) {
		fail("route allocation failed");
	}
	route->mapping = base;
	route->mapping_size = size;
	route->header = header;
	route->metadata = (const TmnfRouteMetadata *)(
		base + header->sections[TMNF_ROUTE_METADATA].offset);
	route->start =
		(const TmnfRouteStart *)(uintptr_t)route->metadata->start_rel;
	route->checkpoints =
		(const TmnfRouteTrigger *)(uintptr_t)route->metadata->checkpoints_rel;
	route->finish =
		(const TmnfRouteTrigger *)(uintptr_t)route->metadata->finish_rel;
	route->centerline =
		(const TmnfRouteReferencePoint *)(uintptr_t)route->metadata->reference_rel;
	initialize_projection_bvh(route);
	if (mprotect(base, size, PROT_READ) != 0) {
		fail("mprotect failed");
	}
	return route;
}

void TmnfRoute_Unload(TmnfRoute *route) {
	free(route->projection_bvh);
	if (munmap(route->mapping, route->mapping_size) != 0) {
		fail("munmap failed");
	}
	free(route);
}

TMNF_HD const TmnfRouteStart *TmnfRoute_GetStart(const TmnfRoute *route) {
	return route->start;
}

TMNF_HD uint32_t TmnfRoute_GetCheckpointCount(const TmnfRoute *route) {
	return route->metadata->checkpoint_count;
}

TMNF_HD const TmnfRouteTrigger *TmnfRoute_GetCheckpoint(
	const TmnfRoute *route, uint32_t index) {
	if (index >= route->metadata->checkpoint_count) {
		fail("checkpoint index out of range");
	}
	return &route->checkpoints[index];
}

TMNF_HD const TmnfRouteTrigger *TmnfRoute_GetFinish(const TmnfRoute *route) {
	return route->finish;
}

TMNF_HD uint32_t TmnfRoute_GetReferencePointCount(const TmnfRoute *route) {
	if (route == NULL || route->metadata == NULL ||
		route->centerline == NULL ||
		route->metadata->centerline_count < 2) {
		fail("route has no centerline");
	}
	return route->metadata->centerline_count;
}

TMNF_HD const TmnfRouteReferencePoint *TmnfRoute_GetReferencePoints(
	const TmnfRoute *route) {
	if (route == NULL || route->metadata == NULL ||
		route->centerline == NULL ||
		route->metadata->centerline_count < 2) {
		fail("route has no centerline");
	}
	return route->centerline;
}

TMNF_HD float TmnfRoute_GetReferenceLength(const TmnfRoute *route) {
	if (route == NULL || route->metadata == NULL ||
		route->centerline == NULL ||
		route->metadata->centerline_count < 2) {
		fail("route has no centerline");
	}
	return route->centerline[
		route->metadata->centerline_count - 1].arc_length;
}

TMNF_HD static TmnfRouteProjection project_in_memory_route(
	const TmnfRoute *route, const GmVec3 *world_position) {
	if (route->mapping != NULL || route->centerline == NULL ||
		route->metadata == NULL ||
		route->metadata->centerline_count < 2) {
		fail("route projection index is missing");
	}
	float best_distance_sq = INFINITY;
	TmnfRouteProjection result = {
		.segment_index = UINT32_MAX,
		.centerline_segment_index = UINT32_MAX,
	};
	for (uint32_t i = 0; i + 1 < route->metadata->centerline_count; ++i) {
		const TmnfRouteReferencePoint *a = &route->centerline[i];
		const TmnfRouteReferencePoint *b = &route->centerline[i + 1];
		float dx = b->position.x - a->position.x;
		float dy = b->position.y - a->position.y;
		float dz = b->position.z - a->position.z;
		float px = world_position->x - a->position.x;
		float py = world_position->y - a->position.y;
		float pz = world_position->z - a->position.z;
		float length_sq = dx * dx + dy * dy + dz * dz;
		if (!(length_sq > 0.0f)) {
			fail("in-memory route has a degenerate segment");
		}
		float t = (px * dx + py * dy + pz * dz) / length_sq;
		if (t < 0.0f) {
			t = 0.0f;
		} else if (t > 1.0f) {
			t = 1.0f;
		}
		float ox = px - t * dx;
		float oy = py - t * dy;
		float oz = pz - t * dz;
		float distance_sq = ox * ox + oy * oy + oz * oz;
		result.segments_tested++;
		if (distance_sq < best_distance_sq ||
			(distance_sq == best_distance_sq &&
			 i < result.centerline_segment_index)) {
			best_distance_sq = distance_sq;
			result.arc_length = a->arc_length +
				t * (b->arc_length - a->arc_length);
			result.half_width = a->half_width +
				t * (b->half_width - a->half_width);
			result.segment_index = a->leg_index;
			result.centerline_segment_index = i;
		}
	}
	result.lateral_offset = sqrtf(best_distance_sq);
	return result;
}

TMNF_HD TmnfRouteProjection TmnfRoute_Project(
	const TmnfRoute *route, const GmVec3 *world_position) {
	if (!finite_floats((const float *)world_position, 3)) {
		fail("cannot project a non-finite position");
	}
	if (route->projection_bvh == NULL) {
		return project_in_memory_route(route, world_position);
	}
	float best_distance_sq = INFINITY;
	TmnfRouteProjection result = {
		.segment_index = UINT32_MAX,
		.centerline_segment_index = UINT32_MAX,
	};
	uint32_t stack[64];
	uint32_t stack_size = 1;
	stack[0] = 0;
	while (stack_size != 0) {
		const TmnfRouteBvhNode *node =
			&route->projection_bvh[stack[--stack_size]];
		float node_distance_sq = 0.0f;
		const float *position = (const float *)world_position;
		for (uint32_t axis = 0; axis < 3; ++axis) {
			float offset = 0.0f;
			if (position[axis] < node->minimum[axis]) {
				offset = node->minimum[axis] - position[axis];
			} else if (position[axis] > node->maximum[axis]) {
				offset = position[axis] - node->maximum[axis];
			}
			node_distance_sq += offset * offset;
		}
		if (node_distance_sq > best_distance_sq) {
			continue;
		}
		if (node->segment == UINT32_MAX) {
			const TmnfRouteBvhNode *left =
				&route->projection_bvh[node->left];
			const TmnfRouteBvhNode *right =
				&route->projection_bvh[node->right];
			float left_distance_sq = 0.0f;
			float right_distance_sq = 0.0f;
			for (uint32_t axis = 0; axis < 3; ++axis) {
				float left_offset = 0.0f;
				float right_offset = 0.0f;
				if (position[axis] < left->minimum[axis]) {
					left_offset = left->minimum[axis] - position[axis];
				} else if (position[axis] > left->maximum[axis]) {
					left_offset = position[axis] - left->maximum[axis];
				}
				if (position[axis] < right->minimum[axis]) {
					right_offset = right->minimum[axis] - position[axis];
				} else if (position[axis] > right->maximum[axis]) {
					right_offset = position[axis] - right->maximum[axis];
				}
				left_distance_sq += left_offset * left_offset;
				right_distance_sq += right_offset * right_offset;
			}
			if (stack_size + 2 > 64) {
				fail("projection BVH traversal stack overflow");
			}
			if (left_distance_sq <= right_distance_sq) {
				stack[stack_size++] = node->right;
				stack[stack_size++] = node->left;
			} else {
				stack[stack_size++] = node->left;
				stack[stack_size++] = node->right;
			}
			continue;
		}
		uint32_t i = node->segment;
		const TmnfRouteReferencePoint *a = &route->centerline[i];
		const TmnfRouteReferencePoint *b = &route->centerline[i + 1];
		float dx = b->position.x - a->position.x;
		float dy = b->position.y - a->position.y;
		float dz = b->position.z - a->position.z;
		float px = world_position->x - a->position.x;
		float py = world_position->y - a->position.y;
		float pz = world_position->z - a->position.z;
		float length_sq = dx * dx + dy * dy + dz * dz;
		float t = (px * dx + py * dy + pz * dz) / length_sq;
		if (t < 0.0f) {
			t = 0.0f;
		} else if (t > 1.0f) {
			t = 1.0f;
		}
		float ox = px - t * dx;
		float oy = py - t * dy;
		float oz = pz - t * dz;
		float distance_sq = ox * ox + oy * oy + oz * oz;
		result.segments_tested++;
		if (distance_sq < best_distance_sq ||
			(distance_sq == best_distance_sq &&
			 i < result.centerline_segment_index)) {
			best_distance_sq = distance_sq;
			result.arc_length = a->arc_length +
				t * (b->arc_length - a->arc_length);
			result.half_width = a->half_width +
				t * (b->half_width - a->half_width);
			result.segment_index = a->leg_index;
			result.centerline_segment_index = i;
		}
	}
	if (result.centerline_segment_index == UINT32_MAX) {
		fail("projection BVH returned no segment");
	}
	result.lateral_offset = sqrtf(best_distance_sq);
	return result;
}
