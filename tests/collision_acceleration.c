/* Compare immutable mesh acceleration with the original ordered full scan.
 * Probe face interiors, edges, vertices, grid boundaries and oversized queries
 * on real geometry. Both contact bytes and contact order must match. */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "track.h"

static uint32_t seed = 0x98bc70d1;
static uint32_t random_u32(void)
{
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}

static void fail(const char *message)
{
	fprintf(stderr, "collision_acceleration: %s\n", message);
	exit(1);
}

int main(int argc, char **argv)
{
	if (argc != 2 && argc != 3) fail("usage: TRACK [scalar|avx2|avx512]");
	if (argc == 3) {
		int supported = strcmp(argv[2], "scalar") == 0;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
		if (strcmp(argv[2], "avx2") == 0)
			supported = __builtin_cpu_supports("avx2");
		if (strcmp(argv[2], "avx512") == 0)
			supported = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
#endif
		if (!supported) {
			printf("collision ISA %s unavailable on this host\n", argv[2]);
			return 77;
		}
		if (setenv("TMNF_COLLISION_PACKET", argv[2], 1)) fail("setenv");
	}
	TmnfTrackHeader header;
	FILE *file = fopen(argv[1], "rb");
	if (!file || fread(&header, sizeof(header), 1, file) != 1) fail("track header");
	fclose(file);
	TmnfTrack *track = TmnfTrack_Load(argv[1], header.track_sha256);
	if (!track->grid.mesh_grids) fail("fixture has no mesh acceleration");
	CHmsCollisionBuffer buffers[2];
	for (int k = 0; k < 2; ++k) CHmsCollisionBuffer_Init(&buffers[k]);
	uint64_t queries = 0, contacts[2] = {0, 0}, cached_faces = 0;
	for (uint32_t m = 0; m < track->mesh_count; ++m) {
		const GmSurfMesh *mesh = &track->meshes[m];
		if (!mesh->face_count) continue;
		const TmnfMeshGrid *grid = &track->grid.mesh_grids[m];
		if (grid->sphere_edges) cached_faces += mesh->face_count;
		TmnfMeshQueryAccel accel = { .mesh_grid = grid };
		GmIso4 identity = { .m = {1, 0, 0, 0, 1, 0, 0, 0, 1} };
		LocatedGmSurf mesh_located = { .surf = (GmSurf *)&mesh->base, .iso = &identity };
		for (int k = 0; k < 2; ++k)
			CFastBuffer_SHmsPhysicalCollision_SetSizeAtLeast(
				&buffers[k].collisions, mesh->node_count + 1);
		for (uint32_t sample = 0; sample < 128; ++sample) {
			const GmSurfMeshFace *face = &mesh->faces[random_u32() % mesh->face_count];
			const GmVec3 *a = &mesh->vertices[face->vertex[0]];
			const GmVec3 *b = &mesh->vertices[face->vertex[1]];
			const GmVec3 *c = &mesh->vertices[face->vertex[2]];
			GmIso4 pose = { .m = {1, 0, 0, 0, 1, 0, 0, 0, 1} };
			float u = sample % 4 == 0 ? 0.0f : sample % 4 == 1 ? 0.5f : 1.0f / 3.0f;
			float v = sample % 4 < 2 ? 0.0f : 1.0f / 3.0f;
			float radius = (float[]){0.000001f, 0.25f, 0.5f, 1.0f, 3.0f}[sample % 5];
			float offset = ((int)(random_u32() % 401) - 200) * radius / 100.0f;
			for (int axis = 0; axis < 3; ++axis) {
				float av = ((const float *)a)[axis];
				pose.t[axis] = av + u * (((const float *)b)[axis] - av)
					+ v * (((const float *)c)[axis] - av)
					+ offset * ((const float *)&face->normal)[axis];
			}
			if (sample % 8 == 0 && grid->levels[0].cell_count) {
				const TmnfMeshGridLevel *level = &grid->levels[0];
				int axis = (int)(sample % 3);
				double cell = floor((pose.t[axis] - level->origin[axis]) * level->inv_cell_size);
				float boundary = (float)(level->origin[axis] + cell * level->cell_size);
				pose.t[axis] = nextafterf(boundary, sample % 16 ? INFINITY : -INFINITY);
			}
			/* Exact axis rotation, including nonuniform ellipsoid boxes. */
			if (sample % 3 == 0) {
				pose.m[0] = pose.m[8] = 0;
				pose.m[2] = 1;
				pose.m[6] = -1;
			}
			GmSurfSphere sphere = { .base = {.type = GM_SURF_SPHERE}, .radius = radius };
			GmSurfEllipsoid ellipsoid = { .base = {.type = GM_SURF_ELLIPSOID},
				.radii = {radius, radius * 0.7f, radius * 1.4f} };
			for (int shape = 0; shape < 2; ++shape) {
				LocatedGmSurf moving = { .surf = shape ? &ellipsoid.base : &sphere.base,
					.iso = &pose, .is_located = 1 };
				int result[2];
				for (int k = 0; k < 2; ++k) {
					buffers[k].collisions.count = 0;
					memset(buffers[k].collisions.data, 0,
						buffers[k].collisions.capacity * sizeof(SHmsPhysicalCollision));
					mesh_located.accel = k ? &accel : NULL;
					result[k] = shape ? GmCollision_Ellipsoid_Mesh(&moving, &mesh_located, &buffers[k])
						: GmCollision_Sphere_Mesh(&moving, &mesh_located, &buffers[k]);
				}
				if (result[0] != result[1] || buffers[0].collisions.count != buffers[1].collisions.count)
					fail("result/count differs from full scan");
				for (uint32_t i = 0; i < buffers[0].collisions.count; ++i)
					if (memcmp(CHmsCollisionBuffer_GetCollision(&buffers[0], i),
						CHmsCollisionBuffer_GetCollision(&buffers[1], i), sizeof(GmCollision))) {
						fprintf(stderr, "mesh %u sample %u shape %d contact %u\n", m, sample, shape, i);
						fail("contact bytes/order differ from full scan");
					}
				contacts[shape] += buffers[0].collisions.count;
				++queries;
			}
		}
	}
	if (!queries || !contacts[0] || !contacts[1] || !cached_faces) fail("vacuous fixture");
	printf("collision_acceleration: %" PRIu64 " queries; %" PRIu64 " sphere and %" PRIu64
		" ellipsoid contacts; %" PRIu64 " cached faces; all bytes/order match\n",
		queries, contacts[0], contacts[1], cached_faces);
	for (int k = 0; k < 2; ++k) CHmsCollisionBuffer_Destroy(&buffers[k]);
	TmnfTrack_Unload(track);
	return 0;
}
