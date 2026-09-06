#include "collision.h"
#include "collision_packet.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#if TMNF_SSE
#include <emmintrin.h>
#endif

#include "tmnf_fp.h"

TMNF_HD static void tree_world_iso(
	GmIso4 *out, const CPlugTree *tree, const GmIso4 *parent) {
	if ((tree->flags & 4u) == 0) {
		memcpy(out, parent, sizeof(*out));
		return;
	}
	memcpy(out, &tree->local_iso, sizeof(*out));
	GmIso4_Mult(out, parent);
}


/* 0x00538090  Initializes a collision buffer with capacity 50. */
/* UNVALIDATED */
TMNF_HD CHmsCollisionBuffer *CHmsCollisionBuffer_Init(CHmsCollisionBuffer *self) {
	CFastBuffer_SHmsPhysicalCollision_Init(&self->collisions);
	CFastBuffer_SHmsPhysicalCollision_SetSizeAtLeast(&self->collisions, 0x32);
	return self;
}

TMNF_HD void CHmsCollisionBuffer_Destroy(CHmsCollisionBuffer *self) {
	CFastBuffer_SHmsPhysicalCollision_Destroy(&self->collisions);
}

TMNF_HD GmCollision *CHmsCollisionBuffer_AddCollision(CHmsCollisionBuffer *self) {
	SHmsPhysicalCollision *physical =
		CFastBuffer_SHmsPhysicalCollision_AddNewElem(&self->collisions);
	return &physical->collision;
}

TMNF_HD GmCollision *CHmsCollisionBuffer_GetCollision(
	CHmsCollisionBuffer *self, uint32_t index) {
	return &CFastBuffer_SHmsPhysicalCollision_At(
		&self->collisions, index)->collision;
}

TMNF_HD uint32_t CHmsCollisionBuffer_GetCount(const CHmsCollisionBuffer *self) {
	return self->collisions.count;
}

/* 0x00539880  Initializes a mergeable per-sphere collision buffer. */
/* UNVALIDATED */
TMNF_HD SHmsSphereBufferContact *SHmsSphereBufferContact_Init(
	SHmsSphereBufferContact *self) {
	CHmsCollisionBuffer_Init(&self->base);
	self->active = 0;
	return self;
}

TMNF_HD static int GmVec3_IsNearlyEqual(
	const GmVec3 *self, const GmVec3 *other) {
	float absolute = (float)fabs(F(other->x));
	float tolerance = x87_r24(0x1.4f8b588e368f1p-17 * F(absolute));
	float lower = x87_sub(other->x, tolerance);
	float upper = x87_add(other->x, tolerance);
	if (!(lower <= F(self->x)) || !(F(self->x) <= upper)) {
		return 0;
	}

	absolute = (float)fabs(F(other->y));
	tolerance = x87_r24(0x1.4f8b588e368f1p-17 * F(absolute));
	lower = x87_sub(other->y, tolerance);
	upper = x87_add(other->y, tolerance);
	if (!(lower <= F(self->y)) || !(F(self->y) <= upper)) {
		return 0;
	}

	absolute = (float)fabs(F(other->z));
	tolerance = x87_r24(0x1.4f8b588e368f1p-17 * F(absolute));
	lower = x87_sub(other->z, tolerance);
	upper = x87_add(other->z, tolerance);
	return lower <= F(self->z) && F(self->z) <= upper;
}

/* Global 0x00D67530, cos(pi/6). The initialiser at 0x00AF4A00 loads the double
 * at 0x00B55E38 (0.5235987901687622, pi/6 widened from float), calls cos, and
 * stores the result as float. Reading 0x00D67530 out of the file image instead
 * yields its pre-init value 0xff336d1b, about -2.4e38, which makes the merge
 * test below always break and silently drops contacts. */
TMNF_HD static float sphere_merge_threshold(void) {
	uint32_t bits = UINT32_C(0x3f5db3d7);
	float value;
	memcpy(&value, &bits, sizeof(value));
	return value;
}

/* 0x00538100  Merges a per-sphere buffer into the destination. */
/* UNVALIDATED */
TMNF_HD void SHmsSphereBufferContact_MergeAndAddToCollisions(
	SHmsSphereBufferContact *self, CHmsCollisionBuffer *destination) {
	uint32_t source_count = self->base.collisions.count;
	uint32_t destination_start = destination->collisions.count;

	for (uint32_t i = 0; i < source_count; i++) {
		SHmsPhysicalCollision *source =
			CFastBuffer_SHmsPhysicalCollision_At(&self->base.collisions, i);
		if (source->collision.flags != 0) {
			SHmsPhysicalCollision *added =
				CFastBuffer_SHmsPhysicalCollision_AddNewElem(
					&destination->collisions);
			memcpy(added, source, sizeof(*added));
		}
	}

	uint32_t destination_end = destination->collisions.count;
	for (uint32_t i = 0; i < source_count; i++) {
		SHmsPhysicalCollision *source =
			CFastBuffer_SHmsPhysicalCollision_At(&self->base.collisions, i);
		if (source->collision.flags != 0) {
			continue;
		}

		uint32_t j = destination_start;
		for (; j < destination_end; j++) {
			SHmsPhysicalCollision *existing =
				CFastBuffer_SHmsPhysicalCollision_At(
					&destination->collisions, j);
			if (GmVec3_IsNearlyEqual(
					&source->collision.face_normal,
					&existing->collision.face_normal)) {
				break;
			}
			float dot = x87_add(
				x87_add(
					x87_mul(
						source->collision.normal.z,
						existing->collision.normal.z),
					x87_mul(
						source->collision.normal.x,
						existing->collision.normal.x)),
				x87_mul(
					source->collision.normal.y,
					existing->collision.normal.y));
			if (sphere_merge_threshold() < dot) {
				break;
			}
		}
		if (j == destination_end) {
			SHmsPhysicalCollision *added =
				CFastBuffer_SHmsPhysicalCollision_AddNewElem(
					&destination->collisions);
			memcpy(added, source, sizeof(*added));
		}
	}

	self->active = 0;
	self->base.collisions.count = 0;
}

/* 0x00547C80  Orders physical collisions for response processing. */
/* UNVALIDATED */
TMNF_HD int SHmsPhysicalCollision_Compare(
	const SHmsPhysicalCollision *a, const SHmsPhysicalCollision *b) {
#define COMPARE_FIELD(field) \
	do { \
		if (!(b->collision.field <= a->collision.field)) { \
			return 1; \
		} \
		if (b->collision.field < a->collision.field) { \
			return -1; \
		} \
	} while (0)

	COMPARE_FIELD(position.x);
	COMPARE_FIELD(position.y);
	COMPARE_FIELD(position.z);
	COMPARE_FIELD(normal.x);
	COMPARE_FIELD(normal.y);
	COMPARE_FIELD(normal.z);
	COMPARE_FIELD(separation.x);
	COMPARE_FIELD(separation.y);
	COMPARE_FIELD(separation.z);
#undef COMPARE_FIELD

	if (a->collision.flags == 0 && b->collision.flags != 0) {
		return -1;
	}
	return 1;
}

/* 0x008F49D0  Computes sphere/sphere overlap and contact data. */
/* UNVALIDATED */
TMNF_HD int GmCollision_Sphere_Sphere(
	const LocatedGmSurf *sphere1, const LocatedGmSurf *sphere2,
	CHmsCollisionBuffer *buffer) {
	const GmSurfSphere *shape1 = (const GmSurfSphere *)sphere1->surf;
	const GmSurfSphere *shape2 = (const GmSurfSphere *)sphere2->surf;

	float dx = x87_sub(sphere2->iso->t[0], sphere1->iso->t[0]);
	float dy = x87_sub(sphere2->iso->t[1], sphere1->iso->t[1]);
	float dz = x87_sub(sphere2->iso->t[2], sphere1->iso->t[2]);
	float distance_squared = x87_add(
		x87_add(x87_mul(dx, dx), x87_mul(dy, dy)),
		x87_mul(dz, dz));
	double radius_sum_x87 = F(shape2->radius) + F(shape1->radius);
	double radius_squared = radius_sum_x87 * radius_sum_x87;
	float radius_sum = (float)radius_sum_x87;
	if (!(F(distance_squared) < radius_squared)) {
		return 0;
	}

	float distance = x87_sqrt(distance_squared);
	GmCollision *collision = CHmsCollisionBuffer_AddCollision(buffer);
	if (distance <= 1.0e-5f) {
		collision->normal.x = 0.0f;
		collision->normal.y = -1.0f;
		collision->normal.z = 0.0f;
		collision->separation.x = 0.0f;
		collision->separation.y = shape2->radius;
		collision->separation.z = 0.0f;
		collision->position.x = sphere1->iso->t[0];
		collision->position.y = sphere1->iso->t[1];
		collision->position.z = sphere1->iso->t[2];
	} else {
		float inverse_distance = x87_rcp(distance);
		dx = (float)(F(inverse_distance) * F(dx));
		dy = (float)(F(dy) * F(inverse_distance));
		dz = (float)(F(inverse_distance) * F(dz));

		collision->normal.x = -dx;
		collision->normal.y = -dy;
		collision->normal.z = -dz;

		float penetration = x87_sub(
			x87_add(shape2->radius, shape1->radius), distance);
		collision->separation.x = F(penetration) * F(dx);
		collision->separation.y = F(dy) * F(penetration);
		collision->separation.z = F(penetration) * F(dz);

		float radius1 = shape1->radius;
		collision->position.x = F(radius1) * F(dx);
		collision->position.y = F(dy) * F(radius1);
		collision->position.z = F(radius1) * F(dz);
		collision->position.x =
			F(sphere1->iso->t[0]) + F(collision->position.x);
		collision->position.y =
			F(sphere1->iso->t[1]) + F(collision->position.y);
		collision->position.z =
			F(sphere1->iso->t[2]) + F(collision->position.z);
	}
	collision->material1 = shape1->base.material_index;
	collision->material2 = shape2->base.material_index;
	(void)radius_sum;
	return 1;
}

TMNF_HD static void GmIso4_SetIdentity(GmIso4 *iso) {
	memset(iso, 0, sizeof(*iso));
	iso->m[0] = 1.0f;
	iso->m[4] = 1.0f;
	iso->m[8] = 1.0f;
}

TMNF_HD static void GmIso4_MultInverse(GmIso4 *self, const GmIso4 *other) {
	GmIso4 inverse;
	GmIso4_SetInverse(&inverse, other);
	GmIso4_Mult(self, &inverse);
}

TMNF_HD static void relative_iso(
	GmIso4 *relative,
	const LocatedGmSurf *first, const LocatedGmSurf *second) {
	if (first->is_located == 0) {
		GmIso4_SetIdentity(relative);
	} else {
		memcpy(relative, first->iso, sizeof(*relative));
	}
	if (second->is_located != 0) {
		if (second->accel != NULL && second->accel->inverse_iso != NULL) {
			GmIso4_Mult(relative, second->accel->inverse_iso);
		} else {
			GmIso4_MultInverse(relative, second->iso);
		}
	}
}

/* Selects the region-restricted node list of the finest grid level whose
 * budget covers the query box, from the cell holding the box center (see
 * track.c). Returns NULL for the full node array. NaN coordinates fail every
 * comparison and take the full array, which rejects them too. */
TMNF_HD static const TmnfMeshListEntry *select_mesh_list(
	const GmSurfMesh *mesh, const TmnfMeshQueryAccel *accel,
	const GmBoxAligned *mesh_box, uint32_t *count) {
	*count = mesh->node_count;
	if (accel == NULL || accel->mesh_grid == NULL) {
		return NULL;
	}
	const TmnfMeshGrid *grid = accel->mesh_grid;
	const float *c = &mesh_box->center.x;
	const float *h = &mesh_box->half_extent.x;
	for (int l = 0; l < TMNF_MESH_GRID_LEVELS; ++l) {
		const TmnfMeshGridLevel *level = &grid->levels[l];
		uint32_t cell = 0;
		int a;
		for (a = 2; a >= 0; --a) {
			if (!(F(h[a]) <= level->max_half)) {
				break;
			}
			double offset = F(c[a]) - level->origin[a];
			if (!(offset >= 0.0 &&
				offset < level->dims[a] * level->cell_size)) {
				break;
			}
			uint32_t index = (uint32_t)(offset * level->inv_cell_size);
			if (index >= level->dims[a]) {
				break;
			}
			cell = cell * level->dims[a] + index;
		}
		if (a >= 0) {
			continue;
		}
		const TmnfMeshPoolSlot *list = grid->pool + level->cells[cell];
		*count = list->node_count;
		return &list[1].entry;
	}
	return NULL;
}

TMNF_HD static void transform_direction(GmVec3 *vector, const GmIso4 *iso) {
	GmVec3 source = *vector;
	GmVec3_SetMult_Mat3(vector, &source, (const GmMat3 *)iso);
}

TMNF_HD static void finish_mesh_collision(
	GmCollision *collision, const GmSurf *first,
	const GmSurfMeshFace *face, const GmVec3 *position,
	const GmVec3 *normal, const GmVec3 *separation, uint32_t flags) {
	collision->normal = *normal;
	collision->separation = *separation;
	collision->position = *position;
	collision->material1 = first->material_index;
	collision->material2 = face->material_index;
	collision->flags = flags;
	collision->face_normal = face->normal;
}

/* Sphere feature contact. dx/dy/dz is center - feature and distance is the
 * value the game uses as |center - feature| for this branch; the end-vertex
 * branch passes sqrt(sqrt(feature_squared)) (see GmCollision_Sphere_Mesh). */
TMNF_HD static void add_sphere_feature_collision(
	const GmSurfSphere *sphere, const GmSurfMeshFace *face,
	const GmVec3 *position, float dx, float dy, float dz, float distance,
	CHmsCollisionBuffer *buffer) {
	float inverse_distance = x87_rcp(distance);
	GmVec3 normal = {
		x87_mul(inverse_distance, dx),
		x87_mul(dy, inverse_distance),
		x87_mul(inverse_distance, dz),
	};
	float scale = x87_mul(
		x87_sub(distance, sphere->radius), inverse_distance);
	GmVec3 radial_separation = {
		x87_mul(scale, dx),
		x87_mul(dy, scale),
		x87_mul(scale, dz),
	};
	float plane_separation = x87_add(
		x87_add(
			x87_mul(face->normal.x, radial_separation.x),
			x87_mul(radial_separation.y, face->normal.y)),
		x87_mul(radial_separation.z, face->normal.z));
	GmVec3 separation = {
		x87_mul(plane_separation, face->normal.x),
		x87_mul(plane_separation, face->normal.y),
		x87_mul(plane_separation, face->normal.z),
	};
	GmCollision *collision = CHmsCollisionBuffer_AddCollision(buffer);
	finish_mesh_collision(
		collision, &sphere->base, face, position, &normal, &separation, 0);
}

/* 0x008EA2D0  Computes sphere contacts against a triangle mesh.
 *
 * Validated against the DesertA1 sphere/mesh trace (Desert wheels are
 * spheres; Stadium never reaches this path). The three feature branches
 * differ in the game and are reproduced as-is:
 *  - start vertex: reject when feature_squared > r*r or <= 1e-10,
 *    distance = sqrt(feature_squared);
 *  - end vertex: d = sqrt(feature_squared); reject when d > r*r or
 *    d <= 1e-10, distance = sqrt(d) (the game square-roots twice);
 *  - edge closest point: reject when feature_squared <= 1e-5, no radius
 *    test, distance = sqrt(feature_squared). */
TMNF_HD int GmCollision_Sphere_Mesh(
	const LocatedGmSurf *sphere_located, const LocatedGmSurf *mesh_located,
	CHmsCollisionBuffer *buffer) {
	/* 0x00D1A938: 1e-10, the edge-length and vertex-distance floor. */
	static const float sphere_tiny_squared = 0x1.b7cdfcp-34f;
	const GmSurfSphere *sphere =
		(const GmSurfSphere *)sphere_located->surf;
	const GmSurfMesh *mesh = (const GmSurfMesh *)mesh_located->surf;
	const TmnfSphereFaceEdges *cached_edges =
		mesh_located->accel != NULL && mesh_located->accel->mesh_grid != NULL
		? mesh_located->accel->mesh_grid->sphere_edges : NULL;
	GmIso4 relative;
	relative_iso(&relative, sphere_located, mesh_located);

	uint32_t start = CHmsCollisionBuffer_GetCount(buffer);
	GmBoxAligned sphere_box = {
		{ 0.0f, 0.0f, 0.0f },
		{ sphere->radius, sphere->radius, sphere->radius },
	};
	GmBoxAligned transformed_box;
	GmBoxQuery transformed_query =
		GmBoxAligned_SetMultQuery(&transformed_box, &sphere_box, &relative);
	GmVec3 center = transformed_box.center;
	int result = 0;

	uint32_t node_count;
	const TmnfMeshListEntry *list = select_mesh_list(
		mesh, mesh_located->accel, &transformed_box, &node_count);
	uint32_t node_index = 0;
	while (node_index < node_count) {
		const GmSurfMeshNode *node = &mesh->nodes[
			list != NULL ? list[node_index].node : node_index];
		uint32_t skip_count = list != NULL ?
			list[node_index].skip : node->skip_count;
		if (!GmBoxQuery_TestInter(&transformed_query, &node->box)) {
			node_index += skip_count;
			continue;
		}
		if (node->face_index == UINT32_MAX) {
			node_index++;
			continue;
		}

		const GmSurfMeshFace *face = &mesh->faces[node->face_index];
		const GmVec3 *vertices[3] = {
			&mesh->vertices[face->vertex[0]],
			&mesh->vertices[face->vertex[1]],
			&mesh->vertices[face->vertex[2]],
		};
		float distance = x87_add(
			x87_add(
				x87_mul(
					x87_sub(center.y, vertices[0]->y),
					face->normal.y),
				x87_mul(
					face->normal.x,
					x87_sub(center.x, vertices[0]->x))),
			x87_mul(
				x87_sub(center.z, vertices[0]->z),
				face->normal.z));
		if (!(distance <= sphere->radius) || !(0.0f <= distance)) {
			node_index++;
			continue;
		}

		float section_squared = x87_sub(
			x87_mul(sphere->radius, sphere->radius),
			x87_mul(distance, distance));
		float section_radius = x87_sqrt(section_squared);
		float negative_distance = -distance;
		GmVec3 projected = {
			x87_add(
				x87_mul(face->normal.x, negative_distance),
				center.x),
			x87_add(
				x87_mul(face->normal.y, negative_distance),
				center.y),
			x87_add(
				x87_mul(negative_distance, face->normal.z),
				center.z),
		};

		int emitted = 0;
		for (uint32_t edge_index = 0; edge_index < 3; edge_index++) {
			uint32_t next_index = edge_index == 2 ? 0 : edge_index + 1;
			const GmVec3 *vertex = vertices[edge_index];
			const GmVec3 *next = vertices[next_index];
			TmnfSphereMeshEdge prepared = cached_edges != NULL
				? cached_edges[node->face_index].edges[edge_index]
				: TmnfSphereMeshEdge_Prepare(vertex, next, &face->normal);
			GmVec3 edge = prepared.direction;
			GmVec3 side = prepared.side;
			GmVec3 from_vertex = {
				x87_sub(projected.x, vertex->x),
				x87_sub(projected.y, vertex->y),
				x87_sub(projected.z, vertex->z),
			};
			float side_distance = x87_add(
				x87_add(
					x87_mul(from_vertex.x, side.x),
					x87_mul(from_vertex.y, side.y)),
				x87_mul(from_vertex.z, side.z));
			if (section_radius < side_distance) {
				emitted = -1;
				break;
			}
			if (!(0.0f < side_distance)) {
				continue;
			}

			float along_start = x87_add(
				x87_add(
					x87_mul(from_vertex.x, edge.x),
					x87_mul(from_vertex.y, edge.y)),
				x87_mul(from_vertex.z, edge.z));
			const GmVec3 *feature;
			GmVec3 closest;
			int is_edge = 0;
			int is_end_vertex = 0;
			if (0.0f <= along_start) {
				GmVec3 from_next = {
					x87_sub(projected.x, next->x),
					x87_sub(projected.y, next->y),
					x87_sub(projected.z, next->z),
				};
				float along_end = x87_add(
					x87_add(
						x87_mul(from_next.x, edge.x),
						x87_mul(from_next.y, edge.y)),
					x87_mul(from_next.z, edge.z));
				if (along_end <= 0.0f) {
					float move = -side_distance;
					closest.x = x87_add(
						x87_mul(move, side.x), projected.x);
					closest.y = x87_add(
						projected.y, x87_mul(side.y, move));
					closest.z = x87_add(
						projected.z, x87_mul(move, side.z));
					feature = &closest;
					is_edge = 1;
				} else {
					feature = next;
					is_end_vertex = 1;
				}
			} else {
				feature = vertex;
			}

			float dx = x87_sub(center.x, feature->x);
			float dy = x87_sub(center.y, feature->y);
			float dz = x87_sub(center.z, feature->z);
			float feature_squared = x87_add(
				x87_add(x87_mul(dx, dx), x87_mul(dy, dy)),
				x87_mul(dz, dz));
			float feature_distance;
			if (is_edge) {
				if (feature_squared <= 1.0e-5f) {
					emitted = -1;
					break;
				}
				feature_distance = x87_sqrt(feature_squared);
			} else {
				float radius_squared = x87_mul(
					sphere->radius, sphere->radius);
				float tested = is_end_vertex
					? x87_sqrt(feature_squared)
					: feature_squared;
				if (!(tested <= radius_squared) ||
					tested <= sphere_tiny_squared) {
					emitted = -1;
					break;
				}
				feature_distance = x87_sqrt(tested);
			}
			add_sphere_feature_collision(
				sphere, face, feature, dx, dy, dz,
				feature_distance, buffer);
			result = 1;
			emitted = 1;
			break;
		}

		if (emitted == 0 && 0.0f < distance) {
			float depth =
				x87_sub(distance, sphere->radius);
			GmVec3 separation = {
				x87_mul(depth, face->normal.x),
				x87_mul(depth, face->normal.y),
				x87_mul(depth, face->normal.z),
			};
			GmCollision *collision = CHmsCollisionBuffer_AddCollision(buffer);
			finish_mesh_collision(
				collision, &sphere->base, face, &projected,
				&face->normal, &separation, 1);
			result = 1;
		}
		node_index++;
	}

	if (result != 0) {
		uint32_t end = CHmsCollisionBuffer_GetCount(buffer);
		for (uint32_t i = start; i < end; i++) {
			GmCollision *collision =
				CHmsCollisionBuffer_GetCollision(buffer, i);
			transform_direction(&collision->normal, mesh_located->iso);
			transform_direction(&collision->separation, mesh_located->iso);
			GmVec3_Mult_Iso4(&collision->position, mesh_located->iso);
		}
	}
	return result;
}

TMNF_HD static float collision_dot3(const GmVec3 *a, const GmVec3 *b) {
	return x87_add(
		x87_add(x87_mul(a->x, b->x), x87_mul(a->y, b->y)),
		x87_mul(a->z, b->z));
}

TMNF_HD static float collision_length_squared(const GmVec3 *value) {
	return collision_dot3(value, value);
}

#if !TMNF_SSE
TMNF_HD static void GmIso4_SetNUScaleTrans(
	GmIso4 *self, const GmVec3 *scale, const GmVec3 *translation) {
	memset(self, 0, sizeof(*self));
	self->m[0] = scale->x;
	self->m[4] = scale->y;
	self->m[8] = scale->z;
	self->t[0] = translation->x;
	self->t[1] = translation->y;
	self->t[2] = translation->z;
}
#endif

TMNF_HD static void make_ellipsoid_inverse(
	GmIso4 *scaled_inverse, GmIso4 *inverse,
	const GmIso4 *relative, const GmVec3 *radii) {
	GmIso4_SetInverse(inverse, relative);
	*scaled_inverse = *inverse;

	float inverse_x = x87_rcp(radii->x);
	scaled_inverse->m[0] = x87_mul(inverse_x, scaled_inverse->m[0]);
	scaled_inverse->m[1] = x87_mul(scaled_inverse->m[1], inverse_x);
	scaled_inverse->m[2] = x87_mul(inverse_x, scaled_inverse->m[2]);
	scaled_inverse->t[0] = x87_mul(scaled_inverse->t[0], inverse_x);

	float inverse_y = x87_rcp(radii->y);
	scaled_inverse->m[3] = x87_mul(inverse_y, scaled_inverse->m[3]);
	scaled_inverse->m[4] = x87_mul(scaled_inverse->m[4], inverse_y);
	scaled_inverse->m[5] = x87_mul(inverse_y, scaled_inverse->m[5]);
	scaled_inverse->t[1] = x87_mul(scaled_inverse->t[1], inverse_y);

	float inverse_z = x87_rcp(radii->z);
	scaled_inverse->m[6] = x87_mul(inverse_z, scaled_inverse->m[6]);
	scaled_inverse->m[7] = x87_mul(scaled_inverse->m[7], inverse_z);
	scaled_inverse->m[8] = x87_mul(inverse_z, scaled_inverse->m[8]);
	scaled_inverse->t[2] = x87_mul(inverse_z, scaled_inverse->t[2]);
}

TMNF_HD static void add_ellipsoid_feature_collision(
	const GmSurfEllipsoid *ellipsoid, const GmSurfMeshFace *face,
	const GmVec3 *point, float distance,
	const GmVec3 *face_normal, CHmsCollisionBuffer *buffer) {
	float inverse_distance = x87_rcp(distance);
	GmVec3 radial = {
		-point->x,
		-point->y,
		-point->z,
	};
	GmVec3 normal = {
		x87_mul(inverse_distance, radial.x),
		x87_mul(radial.y, inverse_distance),
		x87_mul(radial.z, inverse_distance),
	};
	float scale = x87_mul(
		x87_sub(distance, 1.0f), inverse_distance);
	radial.x = x87_mul(scale, radial.x);
	radial.y = x87_mul(radial.y, scale);
	radial.z = x87_mul(scale, radial.z);
	float plane_scale = collision_dot3(face_normal, &radial);
	GmVec3 separation = {
		x87_mul(plane_scale, face_normal->x),
		x87_mul(face_normal->y, plane_scale),
		x87_mul(plane_scale, face_normal->z),
	};

	GmCollision *collision = CHmsCollisionBuffer_AddCollision(buffer);
	finish_mesh_collision(
		collision, &ellipsoid->base, face, point, &normal,
		&separation, 0);
	collision->face_normal = *face_normal;
}

/* The game rebuilds these two isos for every emitting face. Both depend only
 * on the ellipsoid radii, the relative transform and the mesh iso, which are
 * fixed for one GmCollision_Ellipsoid_Mesh call, so they are built once on
 * the first emission and reused. */
typedef struct {
	GmIso4 position_iso;
	GmIso4 normal_iso;
	int ready;
} EllipsoidOutputIsos;

TMNF_HD static void prepare_ellipsoid_output_isos(
	EllipsoidOutputIsos *isos, const GmSurfEllipsoid *ellipsoid,
	const GmIso4 *inverse_relative, const GmIso4 *mesh_iso) {
	if (isos->ready) {
		return;
	}
	const GmVec3 *radii = &ellipsoid->radii;
	float inverse_x = x87_rcp(radii->x);
	float inverse_y = x87_rcp(radii->y);
	float inverse_z = x87_rcp(radii->z);
#if TMNF_SSE
	/* Chained in registers: the scale iso would otherwise be assembled
	 * with scalar stores and re-read as 16-byte chunks, which stalls. */
	GmIsoChunks relative = gm_iso_inverse(gm_iso_load(inverse_relative));
	GmIsoChunks mesh = gm_iso_load(mesh_iso);
	gm_iso_store(&isos->position_iso, gm_iso_mult(gm_iso_mult(
		gm_iso_scale(radii->x, radii->y, radii->z), relative), mesh));
	gm_iso_store(&isos->normal_iso, gm_iso_mult(gm_iso_mult(
		gm_iso_scale(inverse_x, inverse_y, inverse_z), relative), mesh));
#else
	const GmVec3 zero = { 0.0f, 0.0f, 0.0f };
	GmIso4_SetNUScaleTrans(&isos->position_iso, radii, &zero);
	GmIso4_MultInverse(&isos->position_iso, inverse_relative);
	GmIso4_Mult(&isos->position_iso, mesh_iso);

	GmVec3 inverse_radii = { inverse_x, inverse_y, inverse_z };
	GmIso4_SetNUScaleTrans(&isos->normal_iso, &inverse_radii, &zero);
	GmIso4_MultInverse(&isos->normal_iso, inverse_relative);
	GmIso4_Mult(&isos->normal_iso, mesh_iso);
#endif
	isos->ready = 1;
}

TMNF_HD static void transform_ellipsoid_collisions(
	CHmsCollisionBuffer *buffer, uint32_t start,
	const EllipsoidOutputIsos *isos) {
	const GmIso4 *position_iso = &isos->position_iso;
	const GmIso4 *normal_iso = &isos->normal_iso;
	uint32_t end = CHmsCollisionBuffer_GetCount(buffer);
	for (uint32_t i = start; i < end; i++) {
		GmCollision *collision =
			CHmsCollisionBuffer_GetCollision(buffer, i);
		GmVec3_Mult_Iso4(&collision->position, position_iso);
		transform_direction(&collision->normal, normal_iso);
		float normal_squared =
			collision_length_squared(&collision->normal);
		if (0x1.b7cdfcp-34f < normal_squared) {
			float length = x87_sqrt(normal_squared);
			float inverse_length = x87_rcp(length);
			collision->normal.x =
				x87_mul(inverse_length, collision->normal.x);
			collision->normal.y =
				x87_mul(inverse_length, collision->normal.y);
			collision->normal.z =
				x87_mul(inverse_length, collision->normal.z);
		}
		transform_direction(&collision->separation, position_iso);
	}
}

/* One GmCollision_Ellipsoid_Mesh call's fixed state: everything the
 * per-face test reads besides the face itself. Built by
 * ellipsoid_mesh_begin, consumed by ellipsoid_mesh_face; the device's
 * warp-cooperative traversal builds one per query and tests the faces of
 * many queries side by side. */
typedef struct {
	const GmSurfEllipsoid *ellipsoid;
	const GmSurfMesh *mesh;
	const GmIso4 *mesh_iso;
	GmIso4 inverse_relative;
	GmIso4 scaled_inverse;
	GmBoxAligned mesh_box;
	GmBoxQuery mesh_query;
	const TmnfMeshListEntry *list;  /* NULL: the mesh's full node array */
	uint32_t node_count;
	EllipsoidOutputIsos output_isos;
#if TMNF_SSE
	/* Broadcast rows and translation of scaled_inverse: si_m[0..8] = m,
	 * si_m[9..11] = t. */
	__m128 si_m[12];
	GmIso4 relative;
	const TmnfSphereFaceEdges *face_cache;
	int face_transform_ready;
#endif
} EllipsoidMeshQuery;

TMNF_HD static void ellipsoid_mesh_begin(
	EllipsoidMeshQuery *q, const LocatedGmSurf *ellipsoid_located,
	const LocatedGmSurf *mesh_located) {
	q->ellipsoid = (const GmSurfEllipsoid *)ellipsoid_located->surf;
	q->mesh = (const GmSurfMesh *)mesh_located->surf;
	q->mesh_iso = mesh_located->iso;
	GmIso4 relative;
	relative_iso(&relative, ellipsoid_located, mesh_located);

	GmBoxAligned source_box = {
		{ 0.0f, 0.0f, 0.0f },
		q->ellipsoid->radii,
	};
	q->mesh_query =
		GmBoxAligned_SetMultQuery(&q->mesh_box, &source_box, &relative);
#if TMNF_SSE
	/* Most candidate mesh boxes reach no triangle. Delay the inverse and
	 * SIMD broadcasts until the first leaf survives its AABB test. */
	q->face_cache = mesh_located->accel && mesh_located->accel->mesh_grid
		? mesh_located->accel->mesh_grid->sphere_edges : NULL;
	q->relative = relative;
	q->face_transform_ready = 0;
#else
	make_ellipsoid_inverse(
		&q->scaled_inverse, &q->inverse_relative, &relative,
		&q->ellipsoid->radii);
#endif
	q->output_isos.ready = 0;
	q->list = select_mesh_list(
		q->mesh, mesh_located->accel, &q->mesh_box, &q->node_count);
}

/* The node at list position index and the count of positions to skip when
 * its box misses. */
TMNF_HD static const GmSurfMeshNode *ellipsoid_mesh_node(
	const EllipsoidMeshQuery *q, uint32_t index, uint32_t *skip_count) {
	if (q->list != NULL) {
		*skip_count = q->list[index].skip;
		return &q->mesh->nodes[q->list[index].node];
	}
	const GmSurfMeshNode *node = &q->mesh->nodes[index];
	*skip_count = node->skip_count;
	return node;
}

#if TMNF_SSE
static void ellipsoid_prepare_face_transform(EllipsoidMeshQuery *q)
{
	if (q->face_transform_ready)
		return;
	make_ellipsoid_inverse(&q->scaled_inverse, &q->inverse_relative,
		&q->relative, &q->ellipsoid->radii);
	for (int i = 0; i < 9; ++i)
		q->si_m[i] = _mm_set1_ps(q->scaled_inverse.m[i]);
	for (int i = 0; i < 3; ++i)
		q->si_m[9 + i] = _mm_set1_ps(q->scaled_inverse.t[i]);
	q->face_transform_ready = 1;
}
#endif

/* Tests one leaf whose box passed and appends its contact, if any, to
 * buffer. Returns 1 if a contact was appended. */
TMNF_HD static int ellipsoid_mesh_face(
	EllipsoidMeshQuery *q, const GmSurfMeshNode *node,
	CHmsCollisionBuffer *buffer) {
	static const float tiny_squared = 0x1.b7cdfcp-34f;
	static const float edge_closest_squared = 0x1.4f8b58p-17f;
	const GmSurfEllipsoid *ellipsoid = q->ellipsoid;
	const GmSurfMesh *mesh = q->mesh;
#if TMNF_SSE
	if (!q->face_transform_ready) {
		make_ellipsoid_inverse(&q->scaled_inverse, &q->inverse_relative,
			&q->relative, &ellipsoid->radii);
		for (int i = 0; i < 9; ++i)
			q->si_m[i] = _mm_set1_ps(q->scaled_inverse.m[i]);
		for (int i = 0; i < 3; ++i)
			q->si_m[9 + i] = _mm_set1_ps(q->scaled_inverse.t[i]);
		q->face_transform_ready = 1;
	}
	const __m128 *si_m = q->si_m;
#else
	const GmIso4 *scaled_inverse_ptr = &q->scaled_inverse;
#endif
	const GmSurfMeshFace *face = &mesh->faces[node->face_index];
	GmVec3 vertices[3];
#if TMNF_SSE
	/* Lane i holds vertex i (lane 3 repeats lane 0); every lane runs
	 * the scalar path's operations in the scalar path's order. */
	__m128 vx, vy, vz;
	if (q->face_cache) {
		const TmnfSphereFaceEdges *cached = &q->face_cache[node->face_index];
		vx = _mm_loadu_ps(cached->xyz[0]);
		vy = _mm_loadu_ps(cached->xyz[1]);
		vz = _mm_loadu_ps(cached->xyz[2]);
	} else {
		const GmVec3 *sv0 = &mesh->vertices[face->vertex[0]];
		const GmVec3 *sv1 = &mesh->vertices[face->vertex[1]];
		const GmVec3 *sv2 = &mesh->vertices[face->vertex[2]];
		vx = _mm_set_ps(sv0->x, sv2->x, sv1->x, sv0->x);
		vy = _mm_set_ps(sv0->y, sv2->y, sv1->y, sv0->y);
		vz = _mm_set_ps(sv0->z, sv2->z, sv1->z, sv0->z);
	}
	__m128 tx = _mm_add_ps(_mm_add_ps(_mm_add_ps(
		_mm_mul_ps(si_m[0], vx), _mm_mul_ps(si_m[1], vy)),
		_mm_mul_ps(si_m[2], vz)), si_m[9]);
	__m128 ty = _mm_add_ps(_mm_add_ps(_mm_add_ps(
		_mm_mul_ps(si_m[3], vx), _mm_mul_ps(si_m[4], vy)),
		_mm_mul_ps(si_m[5], vz)), si_m[10]);
	__m128 tz = _mm_add_ps(_mm_add_ps(_mm_add_ps(
		_mm_mul_ps(si_m[6], vx), _mm_mul_ps(si_m[7], vy)),
		_mm_mul_ps(si_m[8], vz)), si_m[11]);
	float lane_x[4], lane_y[4], lane_z[4];
	_mm_storeu_ps(lane_x, tx);
	_mm_storeu_ps(lane_y, ty);
	_mm_storeu_ps(lane_z, tz);
	for (uint32_t i = 0; i < 3; i++) {
		vertices[i].x = lane_x[i];
		vertices[i].y = lane_y[i];
		vertices[i].z = lane_z[i];
	}
#else
	for (uint32_t i = 0; i < 3; i++) {
		vertices[i] = mesh->vertices[face->vertex[i]];
		GmVec3_Mult_Iso4(&vertices[i], scaled_inverse_ptr);
	}
#endif
	const GmVec3 *vertex[3] = {
		&vertices[0], &vertices[1], &vertices[2],
	};

	GmVec3 edge01 = {
		x87_sub(vertices[1].x, vertices[0].x),
		x87_sub(vertices[1].y, vertices[0].y),
		x87_sub(vertices[1].z, vertices[0].z),
	};
	GmVec3 edge02 = {
		x87_sub(vertices[2].x, vertices[0].x),
		x87_sub(vertices[2].y, vertices[0].y),
		x87_sub(vertices[2].z, vertices[0].z),
	};
	GmVec3 face_normal = {
		x87_sub(
			x87_mul(edge01.y, edge02.z),
			x87_mul(edge01.z, edge02.y)),
		x87_sub(
			x87_mul(edge02.x, edge01.z),
			x87_mul(edge01.x, edge02.z)),
		x87_sub(
			x87_mul(edge02.y, edge01.x),
			x87_mul(edge01.y, edge02.x)),
	};
	float normal_squared =
		collision_length_squared(&face_normal);
	if (!(tiny_squared < normal_squared)) {
		return 0;
	}
	float normal_length = x87_sqrt(normal_squared);
	float inverse_normal_length =
		x87_rcp(normal_length);
	face_normal.x =
		x87_mul(inverse_normal_length, face_normal.x);
	face_normal.y =
		x87_mul(face_normal.y, inverse_normal_length);
	face_normal.z =
		x87_mul(inverse_normal_length, face_normal.z);

	uint32_t collision_start =
		CHmsCollisionBuffer_GetCount(buffer);
	GmVec3 negative_vertex0 = {
		-vertices[0].x,
		-vertices[0].y,
		-vertices[0].z,
	};
	float plane_distance =
		collision_dot3(&negative_vertex0, &face_normal);
	if (1.0f < plane_distance || plane_distance < 0.0f) {
		return 0;
	}

	float radius_squared = x87_mul(1.0f, 1.0f);
	float section_squared = x87_sub(
		radius_squared,
		x87_mul(plane_distance, plane_distance));
	float section_radius = x87_sqrt(section_squared);
	float negative_distance = -plane_distance;
	GmVec3 projected = {
		x87_add(
			x87_mul(face_normal.x, negative_distance), 0.0f),
		x87_add(
			x87_mul(face_normal.y, negative_distance), 0.0f),
		x87_add(
			x87_mul(face_normal.z, negative_distance), 0.0f),
	};

	int reject_face = 0;
	int emitted = 0;
#if TMNF_SSE
	/* Edge lane i runs from vertex i to vertex (i + 1) % 3. The
	 * per-edge quantities the scalar loop derives before its first
	 * decision are computed for all three edges at once. */
	__m128 ex_ = _mm_shuffle_ps(tx, tx, _MM_SHUFFLE(1, 0, 2, 1));
	__m128 ey_ = _mm_shuffle_ps(ty, ty, _MM_SHUFFLE(1, 0, 2, 1));
	__m128 ez_ = _mm_shuffle_ps(tz, tz, _MM_SHUFFLE(1, 0, 2, 1));
	__m128 ex = _mm_sub_ps(ex_, tx);
	__m128 ey = _mm_sub_ps(ey_, ty);
	__m128 ez = _mm_sub_ps(ez_, tz);
	__m128 esq = _mm_add_ps(_mm_add_ps(
		_mm_mul_ps(ex, ex), _mm_mul_ps(ey, ey)), _mm_mul_ps(ez, ez));
	__m128 normalize = _mm_cmplt_ps(_mm_set1_ps(tiny_squared), esq);
	__m128 inv = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(esq));
	ex = _mm_or_ps(_mm_and_ps(normalize, _mm_mul_ps(inv, ex)),
		_mm_andnot_ps(normalize, ex));
	ey = _mm_or_ps(_mm_and_ps(normalize, _mm_mul_ps(inv, ey)),
		_mm_andnot_ps(normalize, ey));
	ez = _mm_or_ps(_mm_and_ps(normalize, _mm_mul_ps(inv, ez)),
		_mm_andnot_ps(normalize, ez));
	__m128 nx = _mm_set1_ps(face_normal.x);
	__m128 ny = _mm_set1_ps(face_normal.y);
	__m128 nz = _mm_set1_ps(face_normal.z);
	__m128 sx = _mm_sub_ps(_mm_mul_ps(ey, nz), _mm_mul_ps(ez, ny));
	__m128 sy = _mm_sub_ps(_mm_mul_ps(ez, nx), _mm_mul_ps(nz, ex));
	__m128 sz = _mm_sub_ps(_mm_mul_ps(ny, ex), _mm_mul_ps(ey, nx));
	__m128 px = _mm_set1_ps(projected.x);
	__m128 py = _mm_set1_ps(projected.y);
	__m128 pz = _mm_set1_ps(projected.z);
	__m128 fx = _mm_sub_ps(px, tx);
	__m128 fy = _mm_sub_ps(py, ty);
	__m128 fz = _mm_sub_ps(pz, tz);
	__m128 side_distance4 = _mm_add_ps(_mm_add_ps(
		_mm_mul_ps(fx, sx), _mm_mul_ps(fy, sy)), _mm_mul_ps(fz, sz));
	/* The scalar loop uses only the first positive side. Interior face
	 * contacts need no endpoint projections or closest-point arithmetic.
	 * Keep that first-side order when rejecting an outside face as well. */
	int positive_sides = _mm_movemask_ps(
		_mm_cmpgt_ps(side_distance4, _mm_setzero_ps())) & 7;
	if (positive_sides != 0) {
		int first_side = positive_sides & -positive_sides;
		int too_far = _mm_movemask_ps(
			_mm_cmpgt_ps(side_distance4, _mm_set1_ps(section_radius)));
		if (first_side & too_far)
			return 0;
		__m128 along_start4 = _mm_add_ps(_mm_add_ps(
			_mm_mul_ps(fx, ex), _mm_mul_ps(fy, ey)), _mm_mul_ps(fz, ez));
		__m128 gx = _mm_sub_ps(px, ex_);
		__m128 gy = _mm_sub_ps(py, ey_);
		__m128 gz = _mm_sub_ps(pz, ez_);
		__m128 along_end4 = _mm_add_ps(_mm_add_ps(
			_mm_mul_ps(gx, ex), _mm_mul_ps(gy, ey)), _mm_mul_ps(gz, ez));
		__m128 move = _mm_xor_ps(side_distance4, _mm_set1_ps(-0.0f));
		__m128 cx = _mm_add_ps(_mm_mul_ps(move, sx), px);
		__m128 cy = _mm_add_ps(_mm_mul_ps(move, sy), py);
		__m128 cz = _mm_add_ps(_mm_mul_ps(move, sz), pz);
		__m128 closest_squared4 = _mm_add_ps(_mm_add_ps(
			_mm_mul_ps(cx, cx), _mm_mul_ps(cy, cy)), _mm_mul_ps(cz, cz));
		__m128 vertex_squared4 = _mm_add_ps(_mm_add_ps(
			_mm_mul_ps(tx, tx), _mm_mul_ps(ty, ty)), _mm_mul_ps(tz, tz));
		float side_distance_l[4], along_start_l[4], along_end_l[4];
		float closest_squared_l[4], vertex_squared_l[4];
		float cx_l[4], cy_l[4], cz_l[4];
		_mm_storeu_ps(side_distance_l, side_distance4);
		_mm_storeu_ps(along_start_l, along_start4);
		_mm_storeu_ps(along_end_l, along_end4);
		_mm_storeu_ps(closest_squared_l, closest_squared4);
		_mm_storeu_ps(vertex_squared_l, vertex_squared4);
		_mm_storeu_ps(cx_l, cx);
		_mm_storeu_ps(cy_l, cy);
		_mm_storeu_ps(cz_l, cz);
		for (uint32_t edge_index = 0;
			edge_index < 3; edge_index++) {
			uint32_t next_index =
				edge_index == 2 ? 0 : edge_index + 1;
			float side_distance = side_distance_l[edge_index];
			if (section_radius < side_distance) {
				reject_face = 1;
				break;
			}
			if (!(0.0f < side_distance)) {
				continue;
			}

			const GmVec3 *feature;
			GmVec3 closest;
			float feature_squared;
			int is_edge = 0;
			int is_end_vertex = 0;
			if (0.0f <= along_start_l[edge_index]) {
				if (along_end_l[edge_index] <= 0.0f) {
					closest.x = cx_l[edge_index];
					closest.y = cy_l[edge_index];
					closest.z = cz_l[edge_index];
					feature = &closest;
					feature_squared =
						closest_squared_l[edge_index];
					is_edge = 1;
				} else {
					feature = vertex[next_index];
					feature_squared =
						vertex_squared_l[next_index];
					is_end_vertex = 1;
				}
			} else {
				feature = vertex[edge_index];
				feature_squared = vertex_squared_l[edge_index];
			}

			if (is_edge) {
				if (feature_squared <=
					edge_closest_squared) {
					reject_face = 1;
					break;
				}
			} else if (
				1.0f < feature_squared ||
				feature_squared <= tiny_squared) {
				reject_face = 1;
				break;
			}

			float feature_distance =
				x87_sqrt(feature_squared);
			if (is_end_vertex)
				feature_distance = x87_sqrt(feature_distance);
			add_ellipsoid_feature_collision(
				ellipsoid, face, feature,
				feature_distance, &face_normal, buffer);
			emitted = 1;
			break;
		}
	}
#else
	for (uint32_t edge_index = 0;
		edge_index < 3; edge_index++) {
		uint32_t next_index =
			edge_index == 2 ? 0 : edge_index + 1;
		const GmVec3 *start = vertex[edge_index];
		const GmVec3 *end = vertex[next_index];
		GmVec3 edge = {
			x87_sub(end->x, start->x),
			x87_sub(end->y, start->y),
			x87_sub(end->z, start->z),
		};
		float edge_squared =
			collision_length_squared(&edge);
		if (tiny_squared < edge_squared) {
			float edge_length = x87_sqrt(edge_squared);
			float inverse_edge_length =
				x87_rcp(edge_length);
			edge.x =
				x87_mul(inverse_edge_length, edge.x);
			edge.y =
				x87_mul(edge.y, inverse_edge_length);
			edge.z =
				x87_mul(inverse_edge_length, edge.z);
		}

		GmVec3 side = {
			x87_sub(
				x87_mul(edge.y, face_normal.z),
				x87_mul(edge.z, face_normal.y)),
			x87_sub(
				x87_mul(edge.z, face_normal.x),
				x87_mul(face_normal.z, edge.x)),
			x87_sub(
				x87_mul(face_normal.y, edge.x),
				x87_mul(edge.y, face_normal.x)),
		};
		GmVec3 from_start = {
			x87_sub(projected.x, start->x),
			x87_sub(projected.y, start->y),
			x87_sub(projected.z, start->z),
		};
		float side_distance =
			collision_dot3(&from_start, &side);
		if (section_radius < side_distance) {
			reject_face = 1;
			break;
		}
		if (!(0.0f < side_distance)) {
			continue;
		}

		float along_start =
			collision_dot3(&from_start, &edge);
		const GmVec3 *feature;
		GmVec3 closest;
		int is_edge = 0;
		int is_end_vertex = 0;
		if (0.0f <= along_start) {
			GmVec3 from_end = {
				x87_sub(projected.x, end->x),
				x87_sub(projected.y, end->y),
				x87_sub(projected.z, end->z),
			};
			float along_end =
				collision_dot3(&from_end, &edge);
			if (along_end <= 0.0f) {
				float move = -side_distance;
				closest.x = x87_add(
					x87_mul(move, side.x),
					projected.x);
				closest.y = x87_add(
					projected.y,
					x87_mul(side.y, move));
				closest.z = x87_add(
					x87_mul(move, side.z),
					projected.z);
				feature = &closest;
				is_edge = 1;
			} else {
				feature = end;
				is_end_vertex = 1;
			}
		} else {
			feature = start;
		}

		float feature_squared =
			collision_length_squared(feature);
		if (is_edge) {
			if (feature_squared <=
				edge_closest_squared) {
				reject_face = 1;
				break;
			}
		} else if (
			1.0f < feature_squared ||
			feature_squared <= tiny_squared) {
			reject_face = 1;
			break;
		}

		float feature_distance =
			x87_sqrt(feature_squared);
		if (is_end_vertex)
			feature_distance = x87_sqrt(feature_distance);
		add_ellipsoid_feature_collision(
			ellipsoid, face, feature,
			feature_distance, &face_normal, buffer);
		emitted = 1;
		break;
	}
#endif

	if (!reject_face && !emitted && 0.0f < plane_distance) {
		float depth = x87_sub(plane_distance, 1.0f);
		GmVec3 separation = {
			x87_mul(depth, face_normal.x),
			x87_mul(face_normal.y, depth),
			x87_mul(depth, face_normal.z),
		};
		GmCollision *collision =
			CHmsCollisionBuffer_AddCollision(buffer);
		finish_mesh_collision(
			collision, &ellipsoid->base, face,
			&projected, &face_normal, &separation, 1);
		collision->face_normal = face_normal;
		emitted = 1;
	}

	if (emitted) {
		prepare_ellipsoid_output_isos(
			&q->output_isos, ellipsoid, &q->inverse_relative,
			q->mesh_iso);
		transform_ellipsoid_collisions(
			buffer, collision_start, &q->output_isos);
	}
	return emitted;
}

#if TMNF_SSE
static int ellipsoid_mesh_packet(
	EllipsoidMeshQuery *q, const GmSurfMeshNode *const *nodes,
	uint32_t count, CHmsCollisionBuffer *buffer)
{
	int result = 0;
	if (count < 4) {
		for (uint32_t i = 0; i < count; ++i)
			result |= ellipsoid_mesh_face(q, nodes[i], buffer);
		return result;
	}
	ellipsoid_prepare_face_transform(q);
	uint32_t indices[16] = {0};
	GmCollision contacts[16];
	for (uint32_t i = 0; i < count; ++i)
		indices[i] = nodes[i]->face_index;
	uint32_t valid = TmnfCollision_FaceContacts(
		&q->scaled_inverse, q->face_cache, indices, count, contacts);
	if (!valid)
		return 0;
	uint32_t start = CHmsCollisionBuffer_GetCount(buffer);
	for (uint32_t i = 0; i < count; ++i) {
		if (!(valid & (1u << i)))
			continue;
		GmCollision *contact = CHmsCollisionBuffer_AddCollision(buffer);
		*contact = contacts[i];
		contact->material1 = q->ellipsoid->base.material_index;
		contact->material2 = q->mesh->faces[indices[i]].material_index;
	}
	prepare_ellipsoid_output_isos(&q->output_isos, q->ellipsoid,
		&q->inverse_relative, q->mesh_iso);
	transform_ellipsoid_collisions(buffer, start, &q->output_isos);
	return 1;
}
#endif

/* 0x008EADC0  Computes ellipsoid contacts against a triangle mesh. */
/* VALIDATED: 128/128 graph-complete golden records. */
TMNF_HD int GmCollision_Ellipsoid_Mesh(
	const LocatedGmSurf *ellipsoid_located,
	const LocatedGmSurf *mesh_located,
	CHmsCollisionBuffer *buffer) {
	EllipsoidMeshQuery q;
	ellipsoid_mesh_begin(&q, ellipsoid_located, mesh_located);

#if TMNF_SSE
	uint32_t width = q.face_cache != NULL && q.mesh->face_count <= INT32_MAX / 32
		? TmnfCollision_PacketWidth() : 0;
	const GmSurfMeshNode *pending[16];
	uint32_t pending_count = 0;
#endif
	int result = 0;
	uint32_t node_index = 0;
	while (node_index < q.node_count) {
		uint32_t skip_count;
		const GmSurfMeshNode *node =
			ellipsoid_mesh_node(&q, node_index, &skip_count);
		if (!GmBoxQuery_TestInter(&q.mesh_query, &node->box)) {
			node_index += skip_count;
			continue;
		}
		if (node->face_index != UINT32_MAX) {
#if TMNF_SSE
			if (width != 0) {
				pending[pending_count++] = node;
				if (pending_count == width) {
					result |= ellipsoid_mesh_packet(&q, pending, pending_count, buffer);
					pending_count = 0;
				}
			} else
#endif
				result |= ellipsoid_mesh_face(&q, node, buffer);
		}
		node_index++;
	}
#if TMNF_SSE
	if (pending_count != 0)
		result |= ellipsoid_mesh_packet(&q, pending, pending_count, buffer);
#endif
	return result;
}

TMNF_HD static float min3(float a, float b, float c) {
	float result = a;
	if (b < result) {
		result = b;
	}
	if (c < result) {
		result = c;
	}
	return result;
}

TMNF_HD static float max3(float a, float b, float c) {
	float result = a;
	if (result < b) {
		result = b;
	}
	if (result < c) {
		result = c;
	}
	return result;
}

TMNF_HD static int axis_interval_overlap(
	float a, float b, float center, float radius) {
	float minimum = a;
	float maximum = a;
	if (b < minimum) {
		minimum = b;
	}
	if (maximum < b) {
		maximum = b;
	}
	return minimum <= radius + center && center - radius <= maximum;
}

TMNF_HD static int box_triangle_overlap(
	const GmSurfBox *box, const GmVec3 vertices[3]) {
	float x0 = x87_sub(vertices[0].x, box->center.x);
	float x1 = x87_sub(vertices[1].x, box->center.x);
	float x2 = x87_sub(vertices[2].x, box->center.x);
	if (!(min3(x0, x1, x2) <= box->half_extent.x) ||
		!(-box->half_extent.x <= max3(x0, x1, x2))) {
		return 0;
	}

	float y0 = x87_sub(vertices[0].y, box->center.y);
	float y1 = x87_sub(vertices[1].y, box->center.y);
	float y2 = x87_sub(vertices[2].y, box->center.y);
	if (!(min3(y0, y1, y2) <= box->half_extent.y) ||
		!(-box->half_extent.y <= max3(y0, y1, y2))) {
		return 0;
	}

	float z0 = x87_sub(vertices[0].z, box->center.z);
	float z1 = x87_sub(vertices[1].z, box->center.z);
	float z2 = x87_sub(vertices[2].z, box->center.z);
	if (!(min3(z0, z1, z2) <= box->half_extent.z) ||
		!(-box->half_extent.z <= max3(z0, z1, z2))) {
		return 0;
	}

	GmVec3 edge0 = {
		x87_sub(x1, x0),
		x87_sub(y1, y0),
		x87_sub(z1, z0),
	};
	GmVec3 edge1 = {
		x87_sub(x2, x1),
		x87_sub(y2, y1),
		x87_sub(z2, z1),
	};
	GmVec3 triangle_normal = {
		x87_sub(
			x87_mul(edge0.y, edge1.z),
			x87_mul(edge0.z, edge1.y)),
		x87_sub(
			x87_mul(edge1.x, edge0.z),
			x87_mul(edge0.x, edge1.z)),
		x87_sub(
			x87_mul(edge1.y, edge0.x),
			x87_mul(edge1.x, edge0.y)),
	};
	float plane = -x87_add(
		x87_add(
			x87_mul(triangle_normal.x, x0),
			x87_mul(triangle_normal.y, y0)),
		x87_mul(triangle_normal.z, z0));
	GmVec3 positive;
	GmVec3 negative;
	const float extents[3] = {
		box->half_extent.x,
		box->half_extent.y,
		box->half_extent.z,
	};
	const float normal[3] = {
		triangle_normal.x,
		triangle_normal.y,
		triangle_normal.z,
	};
	float *positive_values = &positive.x;
	float *negative_values = &negative.x;
	for (uint32_t axis = 0; axis < 3; axis++) {
		if (normal[axis] <= 0.0f) {
			positive_values[axis] = extents[axis];
			negative_values[axis] = -extents[axis];
		} else {
			positive_values[axis] = -extents[axis];
			negative_values[axis] = extents[axis];
		}
	}
	float positive_plane = x87_add(
		x87_add(
			x87_add(
				x87_mul(positive.x, triangle_normal.x),
				x87_mul(positive.y, triangle_normal.y)),
			x87_mul(positive.z, triangle_normal.z)),
		plane);
	float negative_plane = x87_add(
		x87_add(
			x87_add(
				x87_mul(negative.x, triangle_normal.x),
				x87_mul(negative.y, triangle_normal.y)),
			x87_mul(negative.z, triangle_normal.z)),
		plane);
	if (!(positive_plane <= 0.0) || !(0.0 <= negative_plane)) {
		return 0;
	}

	GmVec3 edges[3] = {
		edge0,
		edge1,
		{
			x87_sub(x0, x2),
			x87_sub(y0, y2),
			x87_sub(z0, z2),
		},
	};
	const GmVec3 centered[3] = {
		{ x0, y0, z0 },
		{ x1, y1, z1 },
		{ x2, y2, z2 },
	};

	for (uint32_t edge_index = 0; edge_index < 3; edge_index++) {
		const GmVec3 *edge = &edges[edge_index];
		const GmVec3 *base = &centered[edge_index];
		const GmVec3 *opposite = &centered[(edge_index + 2) % 3];
		float projections[2];
		float radius;

		projections[0] = x87_sub(
			x87_mul(edge->z, base->y),
			x87_mul(edge->y, base->z));
		projections[1] = x87_sub(
			x87_mul(edge->z, opposite->y),
			x87_mul(edge->y, opposite->z));
		radius = x87_add(
			x87_mul(box->half_extent.y, fabsf(edge->z)),
			x87_mul(box->half_extent.z, fabsf(edge->y)));
		if (!axis_interval_overlap(
				projections[0], projections[1], 0.0f, radius)) {
			return 0;
		}

		projections[0] = x87_sub(
			x87_mul(edge->x, base->z),
			x87_mul(edge->z, base->x));
		projections[1] = x87_sub(
			x87_mul(edge->x, opposite->z),
			x87_mul(edge->z, opposite->x));
		radius = x87_add(
			x87_mul(box->half_extent.z, fabsf(edge->x)),
			x87_mul(box->half_extent.x, fabsf(edge->z)));
		if (!axis_interval_overlap(
				projections[0], projections[1], 0.0f, radius)) {
			return 0;
		}

		projections[0] = x87_sub(
			x87_mul(edge->y, base->x),
			x87_mul(edge->x, base->y));
		projections[1] = x87_sub(
			x87_mul(edge->y, opposite->x),
			x87_mul(edge->x, opposite->y));
		radius = x87_add(
			x87_mul(box->half_extent.y, fabsf(edge->x)),
			x87_mul(box->half_extent.x, fabsf(edge->y)));
		if (!axis_interval_overlap(
				projections[0], projections[1], 0.0f, radius)) {
			return 0;
		}
	}
	return 1;
}

/* 0x008F5200  Tests an oriented box against a triangle mesh. */
/* UNVALIDATED */
TMNF_HD int GmCollision_Box_Mesh(
	const LocatedGmSurf *box_located, const LocatedGmSurf *mesh_located,
	CHmsCollisionBuffer *buffer) {
	const GmSurfBox *box = (const GmSurfBox *)box_located->surf;
	const GmSurfMesh *mesh = (const GmSurfMesh *)mesh_located->surf;
	GmIso4 relative;
	relative_iso(&relative, box_located, mesh_located);

	GmBoxAligned source_box = { box->center, box->half_extent };
	GmBoxAligned mesh_box;
	GmBoxQuery mesh_query =
		GmBoxAligned_SetMultQuery(&mesh_box, &source_box, &relative);

	uint32_t node_index = 0;
	while (node_index < mesh->node_count) {
		const GmSurfMeshNode *node = &mesh->nodes[node_index];
		if (!GmBoxQuery_TestInter(&mesh_query, &node->box)) {
			node_index += node->skip_count;
			continue;
		}
		if (node->face_index == UINT32_MAX) {
			node_index++;
			continue;
		}

		const GmSurfMeshFace *face = &mesh->faces[node->face_index];
		GmVec3 vertices[3];
		for (uint32_t i = 0; i < 3; i++) {
			const GmVec3 *source = &mesh->vertices[face->vertex[i]];
			vertices[i].x =
				x87_sub(source->x, relative.t[0]);
			vertices[i].y =
				x87_sub(source->y, relative.t[1]);
			vertices[i].z =
				x87_sub(source->z, relative.t[2]);
			GmVec3_MultTranspose(
				&vertices[i], (const GmMat3 *)&relative);
		}
		if (!box_triangle_overlap(box, vertices)) {
			node_index++;
			continue;
		}

		GmCollision *collision = CHmsCollisionBuffer_AddCollision(buffer);
		collision->separation.x = 0.0f;
		collision->separation.y = 0.0f;
		collision->separation.z = 0.0f;
		collision->position =
			mesh->vertices[face->vertex[0]];
		GmVec3_Mult_Iso4(&collision->position, mesh_located->iso);
		collision->normal = face->normal;
		transform_direction(&collision->normal, mesh_located->iso);
		collision->material1 = box->base.material_index;
		collision->material2 = face->material_index;
		return 1;
	}
	return 0;
}

TMNF_HD static void GmCollision_Neg(GmCollision *collision) {
	collision->normal.x = -collision->normal.x;
	collision->normal.y = -collision->normal.y;
	collision->normal.z = -collision->normal.z;
	uint16_t material = collision->material1;
	collision->material1 = collision->material2;
	collision->separation.x = -collision->separation.x;
	collision->material2 = material;
	collision->separation.y = -collision->separation.y;
	collision->separation.z = -collision->separation.z;
	collision->face_normal.x = -collision->face_normal.x;
	collision->face_normal.y = -collision->face_normal.y;
	collision->face_normal.z = -collision->face_normal.z;
}

TMNF_HD static GmCollisionHandler resolved_handler(
	uint8_t type1, uint8_t type2, const CollisionRuntime *runtime) {
	if (type1 == GM_SURF_SPHERE && type2 == GM_SURF_BOX) {
		return runtime->shapes.sphere_box;
	}
	if (type1 == GM_SURF_SPHERE && type2 == GM_SURF_ELLIPSOID) {
		return runtime->shapes.sphere_ellipsoid;
	}
	if (type1 == GM_SURF_SPHERE && type2 == GM_SURF_POLYGON) {
		return runtime->shapes.sphere_polygon;
	}
	if (type1 == GM_SURF_ELLIPSOID && type2 == GM_SURF_POLYGON) {
		return runtime->shapes.ellipsoid_polygon;
	}
	if (type1 == GM_SURF_SPHERE && type2 == GM_SURF_MESH) {
		return runtime->shapes.sphere_mesh;
	}
	if (type1 == GM_SURF_ELLIPSOID && type2 == GM_SURF_MESH) {
		return runtime->shapes.ellipsoid_mesh;
	}
	if (type1 == GM_SURF_BOX && type2 == GM_SURF_BOX) {
		return runtime->shapes.box_box;
	}
	if (type1 == GM_SURF_BOX && type2 == GM_SURF_MESH) {
		return runtime->shapes.box_mesh;
	}
	if (type1 == GM_SURF_MESH && type2 == GM_SURF_MESH) {
		return runtime->shapes.mesh_mesh;
	}
	return NULL;
}

TMNF_HD static int is_resolved_pair(uint8_t type1, uint8_t type2) {
	return
		(type1 == GM_SURF_SPHERE && type2 == GM_SURF_BOX) ||
		(type1 == GM_SURF_SPHERE && type2 == GM_SURF_ELLIPSOID) ||
		(type1 == GM_SURF_SPHERE && type2 == GM_SURF_POLYGON) ||
		(type1 == GM_SURF_ELLIPSOID && type2 == GM_SURF_POLYGON) ||
		(type1 == GM_SURF_SPHERE && type2 == GM_SURF_MESH) ||
		(type1 == GM_SURF_ELLIPSOID && type2 == GM_SURF_MESH) ||
		(type1 == GM_SURF_BOX && type2 == GM_SURF_BOX) ||
		(type1 == GM_SURF_BOX && type2 == GM_SURF_MESH) ||
		(type1 == GM_SURF_MESH && type2 == GM_SURF_MESH);
}

/* 0x008E8890  Dispatches a geometry pair and fixes reversed output. */
/* UNVALIDATED */
TMNF_HD int GmSurf_ComputeCollision(
	const LocatedGmSurf *surf1, const LocatedGmSurf *surf2,
	CHmsCollisionBuffer *buffer, const CollisionRuntime *runtime) {
	const LocatedGmSurf *first = surf1;
	const LocatedGmSurf *second = surf2;
	int reversed = 0;
	if (surf1->surf->type > surf2->surf->type) {
		first = surf2;
		second = surf1;
		reversed = 1;
	}

	uint8_t type1 = first->surf->type;
	uint8_t type2 = second->surf->type;
	uint32_t start = CHmsCollisionBuffer_GetCount(buffer);
	int result;
	if (type1 == GM_SURF_SPHERE && type2 == GM_SURF_SPHERE) {
		result = GmCollision_Sphere_Sphere(first, second, buffer);
	} else if (!is_resolved_pair(type1, type2)) {
		result = 0;
	} else {
		GmCollisionHandler handler = resolved_handler(type1, type2, runtime);
		if (handler == NULL) {
			tmnf_abort();
		}
		result = handler(first, second, buffer);
	}
	if (result == 0 || !reversed) {
		return result;
	}

	uint32_t end = CHmsCollisionBuffer_GetCount(buffer);
	for (uint32_t i = start; i < end; i++) {
		GmCollision_Neg(CHmsCollisionBuffer_GetCollision(buffer, i));
	}
	return 1;
}

/* 0x00537150  Computes surface collision and remaps material indices. */
/* UNVALIDATED */
TMNF_HD static int compute_surface_collision(
	const CPlugSurface *surface1, const GmIso4 *iso1,
	const CPlugSurface *surface2, const GmIso4 *iso2,
	const TmnfMeshQueryAccel *accel2,
	CHmsCollisionBuffer *buffer, const CollisionRuntime *runtime) {
	uint32_t start = CHmsCollisionBuffer_GetCount(buffer);
	LocatedGmSurf surf1 = { surface1->geom, iso1, 1, NULL };
	LocatedGmSurf surf2 = { surface2->geom, iso2, 1, accel2 };
	if (!GmSurf_ComputeCollision(&surf1, &surf2, buffer, runtime)) {
		return 0;
	}

	uint32_t end = CHmsCollisionBuffer_GetCount(buffer);
	for (uint32_t i = start; i < end; i++) {
		GmCollision *collision = CHmsCollisionBuffer_GetCollision(buffer, i);
		if (collision->material1 >= surface1->material_count ||
			collision->material2 >= surface2->material_count) {
			tmnf_abort();
		}
		collision->material1 =
			surface1->material_ids[collision->material1];
		collision->material2 =
			surface2->material_ids[collision->material2];
	}
	return 1;
}

TMNF_HD int CPlugSurface_ComputeCollision(
	const CPlugSurface *surface1, const GmIso4 *iso1,
	const CPlugSurface *surface2, const GmIso4 *iso2,
	CHmsCollisionBuffer *buffer, const CollisionRuntime *runtime) {
	return compute_surface_collision(
		surface1, iso1, surface2, iso2, NULL, buffer, runtime);
}

TMNF_HD void CollisionRuntime_Init(CollisionRuntime *runtime) {
	memset(runtime, 0, sizeof(*runtime));
	runtime->shapes.sphere_mesh = GmCollision_Sphere_Mesh;
	runtime->shapes.ellipsoid_mesh = GmCollision_Ellipsoid_Mesh;
	runtime->shapes.box_mesh = GmCollision_Box_Mesh;
}

TMNF_HD void CHmsCollisionManager_SZone_Init(
	CHmsCollisionManager_SZone *zone, const CollisionRuntime *runtime) {
	memset(zone, 0, sizeof(*zone));
	zone->runtime = runtime;
	for (uint32_t i = 0; i < 5; i++) {
		zone->groups[i].runtime = runtime;
	}
}

TMNF_HD void CHmsCollisionManager_SZone_Destroy(CHmsCollisionManager_SZone *zone) {
	free(zone->merge_buffers);
	zone->merge_buffers = NULL;
	zone->merge_buffer_count = 0;
	zone->merge_buffer_capacity = 0;
}

/* 0x00537E80  Computes squared speeds for one non-static group. */
/* UNVALIDATED */
TMNF_HD void CHmsCollisionManager_SGroup_ComputeNonStaticCorpusInfos(
	CHmsCollisionManager_SGroup *self) {
	if (self->is_static != 0) {
		return;
	}
	for (uint32_t i = 0; i < self->corpus_count; i++) {
		GmVec3 speed;
		if (self->corpora[i]->dyna == NULL) {
			speed.x = 0.0f;
			speed.y = 0.0f;
			speed.z = 0.0f;
		} else {
			if (self->runtime == NULL ||
				self->runtime->get_linear_speed == NULL) {
				tmnf_abort();
			}
			self->runtime->get_linear_speed(self->corpora[i], &speed);
		}
		self->speed_sq[i] = x87_add(
			x87_add(
				x87_mul(speed.x, speed.x),
				x87_mul(speed.y, speed.y)),
			x87_mul(speed.z, speed.z));
	}
}

TMNF_HD static int32_t *rect_at(
	CFastRectTableInt *table, uint32_t row, uint32_t column) {
	return table->data + (size_t)table->stride * row + column;
}

/* 0x00537F30  Fills pairwise collision-enable tables for one group. */
/* UNVALIDATED */
TMNF_HD void CHmsCollisionManager_SGroup_ComputeIsToPerformCollisions(
	CHmsCollisionManager_SGroup *self) {
	if (self->is_static != 0) {
		return;
	}
	for (uint32_t m = 0; m < self->device_mat_count; m++) {
		CPlugMaterial_SDeviceMat *device = &self->device_mats[m];
		CHmsCollisionManager_SGroup *other = device->group;
		for (uint32_t i = 0; i < device->perform.rows; i++) {
			for (uint32_t j = 0; j < device->perform.columns; j++) {
				int32_t enabled;
				if (other->is_static != 0) {
					enabled = 1;
				} else {
					float difference =
						x87_sub(self->speed_sq[i], other->speed_sq[j]);
					if (1.0e-5f < difference) {
						enabled = 1;
					} else if (
						difference <= 1.0e-5f &&
						((self->priority == other->priority && j != i) ||
						 self->priority < other->priority)) {
						enabled = 1;
					} else {
						enabled = 0;
					}
				}
				*rect_at(&device->perform, i, j) = enabled;
			}
		}
	}
}

/* 0x0053A0E0  Prepares all five collision groups. */
/* UNVALIDATED */
TMNF_HD void CHmsCollisionManager_SZone_PrepareCollisions(
	CHmsCollisionManager_SZone *self) {
	for (uint32_t i = 0; i < 5; i++) {
		CHmsCollisionManager_SGroup_ComputeNonStaticCorpusInfos(
			&self->groups[i]);
	}
	for (uint32_t i = 0; i < 5; i++) {
		CHmsCollisionManager_SGroup_ComputeIsToPerformCollisions(
			&self->groups[i]);
	}
}

TMNF_HD static SHmsSphereBufferContact *tree_contact_buffer(CPlugTree *tree) {
	if (tree->contact_buffer == NULL) {
#if defined(__CUDA_ARCH__)
		tmnf_fail("collision tree has no bound contact buffer");
#endif
		tree->contact_buffer = (SHmsSphereBufferContact *)malloc(
			sizeof(*tree->contact_buffer));
		if (tree->contact_buffer == NULL) {
			tmnf_abort();
		}
		SHmsSphereBufferContact_Init(tree->contact_buffer);
	}
	return tree->contact_buffer;
}

TMNF_HD static void append_merge_buffer(
	CHmsCollisionManager_SZone *zone, SHmsSphereBufferContact *buffer) {
	if (zone->merge_buffer_count == zone->merge_buffer_capacity) {
#if defined(__CUDA_ARCH__)
		tmnf_fail("merge buffer list capacity exceeded");
#else
		uint32_t capacity = zone->merge_buffer_capacity +
			(zone->merge_buffer_capacity >> 1);
		if (capacity <= zone->merge_buffer_capacity) {
			capacity = zone->merge_buffer_capacity + 1;
		}
		SHmsSphereBufferContact **data = (SHmsSphereBufferContact **)realloc(
			zone->merge_buffers, (size_t)capacity * sizeof(*data));
		if (data == NULL) {
			tmnf_abort();
		}
		zone->merge_buffers = data;
		zone->merge_buffer_capacity = capacity;
#endif
	}
	zone->merge_buffers[zone->merge_buffer_count++] = buffer;
}

TMNF_HD static int compute_surface_pair(
	CHmsCollisionManager_SZone *zone,
	CPlugTree *tree1, const GmIso4 *iso1,
	CPlugTree *tree2, const GmIso4 *iso2,
	const TmnfMeshQueryAccel *accel2,
	CPlugSurface *surface1, CPlugSurface *surface2,
	uint32_t corpus1, uint32_t corpus2,
	uint32_t tree1_ref, uint32_t tree2_ref) {
	if (surface1 == NULL || surface2 == NULL) {
		return 0;
	}

	CHmsCollisionBuffer *buffer;
	SHmsSphereBufferContact *mergeable = NULL;
	if (surface1->geom->type == GM_SURF_SPHERE ||
		surface1->geom->type == GM_SURF_ELLIPSOID) {
		mergeable = tree_contact_buffer(tree1);
		buffer = &mergeable->base;
	} else {
		buffer = zone->general_buffer;
		if (buffer == NULL) {
			tmnf_abort();
		}
	}

	uint32_t start = CHmsCollisionBuffer_GetCount(buffer);
	if (!compute_surface_collision(
			surface1, iso1, surface2, iso2, accel2, buffer,
			zone->runtime)) {
		return 0;
	}

	if (mergeable != NULL && mergeable->active == 0) {
		mergeable->active = 1;
		append_merge_buffer(zone, mergeable);
	}

	uint32_t end = CHmsCollisionBuffer_GetCount(buffer);
	for (uint32_t i = start; i < end; i++) {
		SHmsPhysicalCollision *collision =
			CFastBuffer_SHmsPhysicalCollision_At(&buffer->collisions, i);
		collision->corpus1 = corpus1;
		collision->tree1 = tree1_ref;
		collision->corpus2 = corpus2;
		collision->tree2 = tree2_ref;
		collision->material = zone->current_material;
	}
	(void)tree2;
	return 1;
}

/* A query is served from the grid when its box lies inside the region its
 * cell's list was built for and the surface is a sphere or ellipsoid no larger than
 * the region expansion (that bound is what makes the leaf mesh lists
 * complete, see track.c). NaN coordinates fail every comparison here and
 * take the full scan, which also rejects them. */
TMNF_HD static int static_grid_locate(
	const TmnfStaticGrid *grid, const GmBoxAligned *box,
	const CPlugSurface *surface, uint32_t *cell) {
	GmVec3 shape_radii;
	if (surface->geom->type == GM_SURF_ELLIPSOID) {
		shape_radii = ((const GmSurfEllipsoid *)surface->geom)->radii;
	} else if (surface->geom->type == GM_SURF_SPHERE) {
		float radius = ((const GmSurfSphere *)surface->geom)->radius;
		shape_radii = (GmVec3){ radius, radius, radius };
	} else {
		return 0;
	}
	const float *radii = &shape_radii.x;
	const float *c = &box->center.x;
	const float *h = &box->half_extent.x;
	double inner = grid->expand - grid->margin;
	uint32_t index[3];
	for (int a = 0; a < 3; ++a) {
		if (!(F(radii[a]) <= grid->expand)) {
			return 0;
		}
		double offset = F(c[a]) - grid->origin[a];
		if (!(offset >= 0.0 && offset < grid->dims[a] * grid->cell_size)) {
			return 0;
		}
		index[a] = (uint32_t)(offset * grid->inv_cell_size);
		if (index[a] >= grid->dims[a]) {
			return 0;
		}
		double lo = grid->origin[a] + index[a] * grid->cell_size;
		double hi = lo + grid->cell_size;
		if (!(F(c[a]) - F(h[a]) >= lo - inner &&
			F(c[a]) + F(h[a]) <= hi + inner)) {
			return 0;
		}
	}
	*cell = (index[2] * grid->dims[1] + index[1]) * grid->dims[0] +
		index[0];
	return 1;
}

TMNF_HD static void static_grid_scan(
	CHmsCollisionManager_SZone *self, const TmnfStaticGrid *grid,
	uint32_t cell, const GmBoxQuery *world_query, CPlugTree *tree,
	const GmIso4 *world) {
	const TmnfStaticCellNode *nodes = grid->nodes + grid->cell_offsets[cell];
	uint32_t count = grid->cell_counts[cell];
	const HmsStaticCollisionEntry *entries =
		self->static_group->static_entries;
	uint32_t i = 0;
	while (i < count) {
		const TmnfStaticCellNode *node = &nodes[i];
		if (!GmBoxQuery_TestInter(world_query, &node->box)) {
			i += node->skip;
			continue;
		}
		i++;
		if (node->entry_index >= TMNF_CELL_NODE_EMPTY) {
			continue;
		}
		TmnfMeshQueryAccel accel = {
			.inverse_iso = &grid->entry_inverse_isos[node->entry_index],
			.mesh_grid = grid->entry_mesh_grids[node->entry_index],
		};
		const HmsStaticCollisionEntry *entry = &entries[node->entry_index];
		(void)compute_surface_pair(
			self,
			tree, world,
			NULL, &entry->iso, &accel,
			tree->surface, entry->surface,
			self->current_corpus1, entry->corpus_ref,
			tree->object_ref, entry->tree_ref);
	}
}

/* 0x0053A120  Collides a tree against the current static collision tree. */
/* UNVALIDATED */
TMNF_HD static void static_tree_detect(
	CHmsCollisionManager_SZone *self, const GmIso4 *iso, CPlugTree *tree) {
	if ((tree->flags & 0x80u) == 0) {
		return;
	}

	GmIso4 world;
	tree_world_iso(&world, tree, iso);
	for (uint32_t i = 0; i < tree->child_count; i++) {
		static_tree_detect(self, &world, tree->children[i]);
	}
	if (tree->surface == NULL || self->static_group == NULL) {
		return;
	}

	GmBoxAligned world_box;
	GmBoxQuery world_query =
		GmBoxAligned_SetMultQuery(&world_box, &tree->box, iso);
	const TmnfStaticGrid *grid = self->static_group->static_grid;
	if (grid != NULL) {
		uint32_t cell;
		if (static_grid_locate(grid, &world_box, tree->surface, &cell)) {
			static_grid_scan(self, grid, cell, &world_query, tree, &world);
			return;
		}
	}

	uint32_t i = 0;
	while (i < self->static_group->static_entry_count) {
		HmsStaticCollisionEntry *entry =
			&self->static_group->static_entries[i];
		if (!GmBoxQuery_TestInter(&world_query, &entry->box)) {
			i += entry->skip_count;
			continue;
		}
		if (entry->surface != NULL &&
			(entry->tree_flags & 0x80u) != 0) {
			(void)compute_surface_pair(
				self,
				tree, &world,
				NULL, &entry->iso, NULL,
				tree->surface, entry->surface,
				self->current_corpus1, entry->corpus_ref,
				tree->object_ref, entry->tree_ref);
		}
		i++;
	}
}

#if defined(__CUDACC__)
/* src/cuda/dev/collision.cu: the same detection, cooperatively across the
 * warp. */
__device__ void tmnf_dev_static_tree_detect(
	CHmsCollisionManager_SZone *self, const GmIso4 *iso, CPlugTree *tree,
	int active);
#endif

/* active = 0: the caller has no detection to run and only accompanies the
 * warp through the cooperative device version (tmnf_warp.h). */
TMNF_HD void CHmsCollisionManager_SZone_DetectCollisionBetweenTreeAndStaticCollisionTree(
	CHmsCollisionManager_SZone *self, const GmIso4 *iso, CPlugTree *tree,
	int active) {
#if defined(__CUDA_ARCH__)
	tmnf_dev_static_tree_detect(self, iso, tree, active);
#else
	if (active) {
		static_tree_detect(self, iso, tree);
	}
#endif
}

/* 0x0053A3D0  Collides tree two's root/subtree against tree one's root. */
/* UNVALIDATED */
TMNF_HD int CHmsCollisionManager_SZone_ComputeCollisionTree1RootOnly(
	CHmsCollisionManager_SZone *self, const SPlugTreeLocatedPair *pair,
	const GmBoxAligned *tree1_box) {
	CPlugTree *tree2 = pair->tree2;
	if ((tree2->flags & 0x80u) == 0) {
		return 0;
	}

	GmBoxAligned tree2_box;
	GmBoxAligned_SetMult(&tree2_box, &tree2->box, pair->iso2);
	if (!GmBoxAligned_TestInter(&tree2_box, tree1_box)) {
		return 0;
	}

	GmIso4 world2;
	tree_world_iso(&world2, tree2, pair->iso2);
	int result = compute_surface_pair(
		self,
		pair->tree1, pair->iso1,
		tree2, &world2, NULL,
		pair->tree1->surface, tree2->surface,
		self->current_corpus1, self->current_corpus2,
		pair->tree1->object_ref, tree2->object_ref);

	for (uint32_t i = 0; i < tree2->child_count; i++) {
		SPlugTreeLocatedPair child = {
			pair->tree1,
			pair->iso1,
			tree2->children[i],
			&world2,
		};
		if (CHmsCollisionManager_SZone_ComputeCollisionTree1RootOnly(
				self, &child, tree1_box)) {
			result = 1;
		}
	}
	return result;
}

/* 0x0053A660  Collides tree one's root/subtree against tree two's root. */
/* UNVALIDATED */
TMNF_HD int CHmsCollisionManager_SZone_ComputeCollisionTree2RootOnly(
	CHmsCollisionManager_SZone *self, const SPlugTreeLocatedPair *pair,
	const GmBoxAligned *tree2_box) {
	CPlugTree *tree1 = pair->tree1;
	if ((tree1->flags & 0x80u) == 0) {
		return 0;
	}

	GmBoxAligned tree1_box;
	GmBoxAligned_SetMult(&tree1_box, &tree1->box, pair->iso1);
	if (!GmBoxAligned_TestInter(&tree1_box, tree2_box)) {
		return 0;
	}

	GmIso4 world1;
	tree_world_iso(&world1, tree1, pair->iso1);
	int result = compute_surface_pair(
		self,
		tree1, &world1,
		pair->tree2, pair->iso2, NULL,
		tree1->surface, pair->tree2->surface,
		self->current_corpus1, self->current_corpus2,
		tree1->object_ref, pair->tree2->object_ref);

	for (uint32_t i = 0; i < tree1->child_count; i++) {
		SPlugTreeLocatedPair child = {
			tree1->children[i],
			&world1,
			pair->tree2,
			pair->iso2,
		};
		if (CHmsCollisionManager_SZone_ComputeCollisionTree2RootOnly(
				self, &child, tree2_box)) {
			result = 1;
		}
	}
	return result;
}

/* 0x0053A8F0  Recursively collides two located plug trees. */
/* UNVALIDATED */
TMNF_HD int CHmsCollisionManager_SZone_ComputeCollision(
	CHmsCollisionManager_SZone *self, const SPlugTreeLocatedPair *pair) {
	CPlugTree *tree1 = pair->tree1;
	CPlugTree *tree2 = pair->tree2;
	if ((tree1->flags & 0x80u) == 0 || (tree2->flags & 0x80u) == 0) {
		return 0;
	}

	GmBoxAligned box1;
	GmBoxAligned box2;
	GmBoxAligned_SetMult(&box1, &tree1->box, pair->iso1);
	GmBoxAligned_SetMult(&box2, &tree2->box, pair->iso2);
	if (!GmBoxAligned_TestInter(&box1, &box2)) {
		return 0;
	}

	GmIso4 world1;
	GmIso4 world2;
	tree_world_iso(&world1, tree1, pair->iso1);
	tree_world_iso(&world2, tree2, pair->iso2);

	int result = compute_surface_pair(
		self,
		tree1, &world1,
		tree2, &world2, NULL,
		tree1->surface, tree2->surface,
		self->current_corpus1, self->current_corpus2,
		tree1->object_ref, tree2->object_ref);

	if (tree1->surface != NULL) {
		for (uint32_t i = 0; i < tree2->child_count; i++) {
			SPlugTreeLocatedPair child = {
				tree1,
				&world1,
				tree2->children[i],
				&world2,
			};
			if (CHmsCollisionManager_SZone_ComputeCollisionTree1RootOnly(
					self, &child, &box1)) {
				result = 1;
			}
		}
	}

	if (tree2->surface != NULL) {
		for (uint32_t i = 0; i < tree1->child_count; i++) {
			SPlugTreeLocatedPair child = {
				tree1->children[i],
				&world1,
				tree2,
				&world2,
			};
			if (CHmsCollisionManager_SZone_ComputeCollisionTree2RootOnly(
					self, &child, &box2)) {
				result = 1;
			}
		}
	}

	for (uint32_t i = 0; i < tree1->child_count; i++) {
		for (uint32_t j = 0; j < tree2->child_count; j++) {
			SPlugTreeLocatedPair child = {
				tree1->children[i],
				&world1,
				tree2->children[j],
				&world2,
			};
			if (CHmsCollisionManager_SZone_ComputeCollision(self, &child)) {
				result = 1;
			}
		}
	}
	return result;
}

TMNF_HD static const GmIso4 *corpus_iso(const CHmsCorpus *corpus) {
	if (corpus->dyna == NULL) {
		return &corpus->local_iso;
	}
	if (corpus->live_iso == NULL) {
		tmnf_abort();
	}
	return corpus->live_iso;
}

/* 0x0053AFB0  Builds root pairs and collides two corpora. */
/* UNVALIDATED */
TMNF_HD void CHmsCollisionManager_SZone_DetectCollisionBetween(
	CHmsCollisionManager_SZone *self, CHmsCorpus *corpus1,
	CHmsCorpus *corpus2) {
	self->current_corpus1 = corpus1->object_ref;
	self->current_corpus2 = corpus2->object_ref;
	SPlugTreeLocatedPair pair = {
		corpus1->tree,
		corpus_iso(corpus1),
		corpus2->tree,
		corpus_iso(corpus2),
	};
	(void)CHmsCollisionManager_SZone_ComputeCollision(self, &pair);
}

/* 0x0053B1C0  Detects all enabled collisions for one corpus. */
/* UNVALIDATED */
/* active = 0: no detection for this corpus; the call only walks the
 * material loop (the same for every environment of a track) so that the
 * static detection's warp-cooperative device version sees every lane. */
TMNF_HD void CHmsCollisionManager_SZone_DetectCollisionsCorpus(
	CHmsCollisionManager_SZone *self, CHmsCollisionBuffer *buffer,
	CHmsCorpus *corpus, int active) {
	if (active) {
		self->general_buffer = buffer;
	}
	uint32_t group_index = (corpus->flags >> 13) & 0x0fu;
	CHmsCollisionManager_SGroup *group = &self->groups[group_index - 1];

	for (uint32_t m = 0; m < group->device_mat_count; m++) {
		CPlugMaterial_SDeviceMat *device = &group->device_mats[m];
		if (active) {
			self->current_material = device->material_ref;
			for (uint32_t j = 0; j < device->perform.columns; j++) {
				if (*rect_at(&device->perform, corpus->group_index, j) !=
					0) {
					CHmsCollisionManager_SZone_DetectCollisionBetween(
						self, corpus, device->group->corpora[j]);
				}
			}
			self->static_group = device->group;
		}
		if (device->group->static_entry_count > 1) {
			if (active) {
				self->current_corpus1 = corpus->object_ref;
			}
			CHmsCollisionManager_SZone_DetectCollisionBetweenTreeAndStaticCollisionTree(
				self, corpus_iso(corpus), corpus->tree, active);
		}
	}

	if (!active) {
		return;
	}
	for (uint32_t i = 0; i < self->merge_buffer_count; i++) {
		SHmsSphereBufferContact_MergeAndAddToCollisions(
			self->merge_buffers[i], buffer);
	}
	self->merge_buffer_count = 0;
}
