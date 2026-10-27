#include "race.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

TMNF_HD static void race_fail(const char *message)
{
	tmnf_fail(message);
}

TMNF_HD static void validate_route(const TmnfRoute *route)
{
	if (route == NULL || route->metadata == NULL || route->start == NULL ||
		route->checkpoints == NULL || route->finish == NULL ||
		route->centerline == NULL) {
		race_fail("route is incomplete");
	}
	const TmnfRouteMetadata *metadata = route->metadata;
	/* checkpoint_count may be zero: tracks like A10-Acrobatic have only a
	 * start and a finish. The visited set, the finish gate, and the
	 * total-checkpoint arithmetic below are all well defined for zero. */
	if (metadata->lap_count == 0 ||
		!TmnfRoute_TriggerCountsSupported(metadata) ||
		metadata->checkpoint_count > TMNF_RACE_MAX_CHECKPOINTS ||
		metadata->reference_count != metadata->checkpoint_count + 2 ||
		metadata->centerline_count < 2) {
		race_fail("route metadata is inconsistent");
	}
	if (((metadata->flags & TMNF_ROUTE_MULTILAP) != 0) !=
		(metadata->lap_count > 1)) {
		race_fail("route multilap flag is inconsistent");
	}
	uint64_t expected = (uint64_t)(metadata->checkpoint_count + 1) *
		metadata->lap_count;
	if (expected > UINT32_MAX ||
		metadata->total_race_checkpoints != (uint32_t)expected) {
		race_fail("route total checkpoint count is inconsistent");
	}
}

/* self = self * parent, the collision manager's GmIso4::Mult order. */
TMNF_HD static void compose_with_parent(GmIso4 *transform, const GmIso4 *parent)
{
	GmMat3 rotation;
	GmVec3 translation = {
		transform->t[0],
		transform->t[1],
		transform->t[2],
	};
	memcpy(rotation.m, transform->m, sizeof(rotation.m));
	GmMat3_Mult(&rotation, (const GmMat3 *)parent);
	memcpy(transform->m, rotation.m, sizeof(rotation.m));
	GmVec3_Mult_Iso4(&translation, parent);
	transform->t[0] = translation.x;
	transform->t[1] = translation.y;
	transform->t[2] = translation.z;
}

/* Mirrors the collision manager's tree_world_iso: flag 4 applies local_iso. */
TMNF_HD static void tree_world_iso(
	GmIso4 *out, const CPlugTree *tree, const GmIso4 *parent)
{
	if ((tree->flags & 4u) == 0) {
		*out = *parent;
		return;
	}
	*out = tree->local_iso;
	compose_with_parent(out, parent);
}

/* --- trigger volume as the game sees it: a twelve-face box mesh ---------- */

typedef struct {
	GmSurfMesh mesh;
	GmVec3 vertices[8];
	GmSurfMeshFace faces[12];
	GmSurfMeshNode nodes[12];
	CPlugSurface surface;
	uint8_t material_id;
} TriggerMesh;

TMNF_HD static void face_box(
	const GmVec3 *a, const GmVec3 *b, const GmVec3 *c, GmBoxAligned *box)
{
	GmVec3 low = *a;
	GmVec3 high = *a;
	const GmVec3 *points[2] = { b, c };
	for (uint32_t i = 0; i < 2; ++i) {
		const GmVec3 *p = points[i];
		if (p->x < low.x) low.x = p->x;
		if (p->y < low.y) low.y = p->y;
		if (p->z < low.z) low.z = p->z;
		if (p->x > high.x) high.x = p->x;
		if (p->y > high.y) high.y = p->y;
		if (p->z > high.z) high.z = p->z;
	}
	box->center.x = 0.5f * (low.x + high.x);
	box->center.y = 0.5f * (low.y + high.y);
	box->center.z = 0.5f * (low.z + high.z);
	box->half_extent.x = 0.5f * (high.x - low.x);
	box->half_extent.y = 0.5f * (high.y - low.y);
	box->half_extent.z = 0.5f * (high.z - low.z);
}

TMNF_HD static void add_face(
	TriggerMesh *m, uint32_t index, uint32_t v0, uint32_t v1, uint32_t v2,
	const GmVec3 *outward)
{
	const GmVec3 *a = &m->vertices[v0];
	const GmVec3 *b = &m->vertices[v1];
	const GmVec3 *c = &m->vertices[v2];
	GmVec3 e1 = { b->x - a->x, b->y - a->y, b->z - a->z };
	GmVec3 e2 = { c->x - a->x, c->y - a->y, c->z - a->z };
	GmVec3 n = {
		e1.y * e2.z - e1.z * e2.y,
		e1.z * e2.x - e1.x * e2.z,
		e1.x * e2.y - e1.y * e2.x,
	};
	if (n.x * outward->x + n.y * outward->y + n.z * outward->z < 0.0f) {
		uint32_t swap = v1;
		v1 = v2;
		v2 = swap;
	}
	GmSurfMeshFace *face = &m->faces[index];
	memset(face, 0, sizeof(*face));
	face->normal = *outward;
	face->vertex[0] = v0;
	face->vertex[1] = v1;
	face->vertex[2] = v2;
	face->material_index = 0;
	GmSurfMeshNode *node = &m->nodes[index];
	node->skip_count = 1;
	face_box(&m->vertices[v0], &m->vertices[v1], &m->vertices[v2],
		&node->box);
	node->face_index = index;
}

TMNF_HD static void build_trigger_mesh(TriggerMesh *m, const GmBoxAligned *box)
{
	memset(m, 0, sizeof(*m));
	if (!(box->half_extent.x > 0.0f) || !(box->half_extent.y > 0.0f) ||
		!(box->half_extent.z > 0.0f)) {
		race_fail("trigger box has a non-positive extent");
	}
	for (uint32_t i = 0; i < 8; ++i) {
		m->vertices[i].x = box->center.x +
			((i & 1u) ? box->half_extent.x : -box->half_extent.x);
		m->vertices[i].y = box->center.y +
			((i & 2u) ? box->half_extent.y : -box->half_extent.y);
		m->vertices[i].z = box->center.z +
			((i & 4u) ? box->half_extent.z : -box->half_extent.z);
	}
	uint32_t face = 0;
	for (uint32_t axis = 0; axis < 3; ++axis) {
		uint32_t bit = 1u << axis;
		for (uint32_t side = 0; side < 2; ++side) {
			GmVec3 outward = { 0.0f, 0.0f, 0.0f };
			float sign = side ? 1.0f : -1.0f;
			if (axis == 0) outward.x = sign;
			else if (axis == 1) outward.y = sign;
			else outward.z = sign;
			uint32_t corners[4];
			uint32_t count = 0;
			for (uint32_t v = 0; v < 8; ++v) {
				if (((v & bit) != 0) == (side != 0))
					corners[count++] = v;
			}
			/* corners are in increasing index order: (00, 01, 10, 11) over
			 * the two remaining axes, so 0-1-3 and 0-3-2 tile the quad. */
			add_face(m, face++, corners[0], corners[1], corners[3], &outward);
			add_face(m, face++, corners[0], corners[3], corners[2], &outward);
		}
	}
	m->mesh.base.type = GM_SURF_MESH;
	m->mesh.vertex_count = 8;
	m->mesh.vertices = m->vertices;
	m->mesh.face_count = 12;
	m->mesh.faces = m->faces;
	m->mesh.node_count = 12;
	m->mesh.nodes = m->nodes;
	m->material_id = 0;
	m->surface.geom = &m->mesh.base;
	m->surface.material_ids = &m->material_id;
	m->surface.material_count = 1;
}

/* Car subtrees against the trigger, mirroring 0x0053A660
 * ComputeCollisionTree2RootOnly with the trigger as tree two. The vehicle
 * tree is a root with leaf children; the explicit stack replaces the game's
 * recursion so the same code runs on a GPU thread. Only the OR of the
 * per-pair results matters, so the visiting order is free. */
enum {
	TRIGGER_TREE_STACK_CAPACITY = 16,
	TRIGGER_CONTACT_CAPACITY = 64,
};

typedef struct {
	const CPlugTree *tree;
	GmIso4 parent_world;
} TriggerTreeFrame;

TMNF_HD static int car_tree_contact(
	const CPlugTree *car_tree, const GmIso4 *car_world_transform,
	const GmBoxAligned *trigger_world_box, const CPlugSurface *surface,
	const GmIso4 *trigger_iso, CHmsCollisionBuffer *buffer,
	const CollisionRuntime *runtime)
{
	TriggerTreeFrame stack[TRIGGER_TREE_STACK_CAPACITY];
	uint32_t depth = 0;
	int contact = 0;
	GmIso4 car_world;
	tree_world_iso(&car_world, car_tree, car_world_transform);
	if (car_tree->surface != NULL && car_tree->surface->geom != NULL) {
		contact |= CPlugSurface_ComputeCollision(
			car_tree->surface, &car_world, surface,
			trigger_iso, buffer, runtime);
	}
	if (car_tree->child_count > TRIGGER_TREE_STACK_CAPACITY)
		race_fail("car collision tree is wider than the trigger stack");
	for (uint32_t i = car_tree->child_count; i-- > 0;) {
		stack[depth].tree = car_tree->children[i];
		stack[depth].parent_world = car_world;
		depth++;
	}
	while (depth != 0) {
		TriggerTreeFrame frame = stack[--depth];
		const CPlugTree *tree = frame.tree;
		if ((tree->flags & 0x80u) == 0)
			continue;
		GmBoxAligned box;
		GmBoxAligned_SetMult(&box, &tree->box, &frame.parent_world);
		if (!GmBoxAligned_TestInter(&box, trigger_world_box))
			continue;
		GmIso4 world;
		tree_world_iso(&world, tree, &frame.parent_world);
		if (tree->surface != NULL && tree->surface->geom != NULL) {
			contact |= CPlugSurface_ComputeCollision(
				tree->surface, &world, surface, trigger_iso,
				buffer, runtime);
		}
		if (depth + tree->child_count > TRIGGER_TREE_STACK_CAPACITY)
			race_fail("car collision tree is deeper than the trigger stack");
		for (uint32_t i = tree->child_count; i-- > 0;) {
			stack[depth].tree = tree->children[i];
			stack[depth].parent_world = world;
			depth++;
		}
	}
	return contact;
}

TMNF_HD static int trigger_contact(
	const TmnfRouteTrigger *trigger,
	const CPlugTree *car_tree,
	const GmIso4 *car_world_transform, const TmnfTrack *geometry,
	uint32_t corpus_ref, CHmsCollisionBuffer *output)
{
	if (trigger == NULL || car_tree == NULL || car_world_transform == NULL)
		race_fail("trigger test argument is null");
	if ((car_tree->flags & 0x80u) == 0)
		race_fail("car collision tree has collisions disabled");
	if ((trigger->tree_flags & 0x80u) == 0)
		race_fail("trigger tree has collisions disabled");

	GmBoxAligned car_box;
	GmBoxAligned trigger_box;
	GmBoxAligned_SetMult(&car_box, &car_tree->box, car_world_transform);
	GmBoxAligned_SetMult(&trigger_box, &trigger->box, &trigger->transform);
	if (!GmBoxAligned_TestInter(&car_box, &trigger_box))
		return 0;

	CollisionRuntime runtime;
	CollisionRuntime_Init(&runtime);
	/* The contacts themselves are discarded; only their existence counts.
	 * A device thread cannot allocate, so it records into a fixed-capacity
	 * array and traps if a trigger ever produces more. */
	CHmsCollisionBuffer local;
	CHmsCollisionBuffer *buffer = output != NULL ? output : &local;
#if defined(__CUDA_ARCH__)
	SHmsPhysicalCollision storage[TRIGGER_CONTACT_CAPACITY];
	if (output == NULL) {
		local.collisions.count = 0;
		local.collisions.data = storage;
		local.collisions.capacity = TRIGGER_CONTACT_CAPACITY;
	}
#else
	if (output == NULL) CHmsCollisionBuffer_Init(&local);
#endif
	int contact = 0;
	if (geometry != NULL) {
		for (uint32_t i = 0; i < geometry->entry_count; ++i) {
			const HmsStaticCollisionEntry *entry = &geometry->entries[i];
			if (entry->corpus_ref != corpus_ref || entry->surface == NULL ||
				(entry->tree_flags & 0x80u) == 0) continue;
			if (output == NULL) buffer->collisions.count = 0;
			contact |= car_tree_contact(car_tree, car_world_transform,
				&trigger_box, entry->surface, &entry->iso, buffer, &runtime);
			if (contact && output == NULL) break;
		}
	} else {
		TriggerMesh mesh;
		build_trigger_mesh(&mesh, &trigger->box);
		contact = car_tree_contact(car_tree, car_world_transform, &trigger_box,
			&mesh.surface, &trigger->transform, buffer, &runtime);
	}
#if !defined(__CUDA_ARCH__)
	if (output == NULL) CHmsCollisionBuffer_Destroy(&local);
#endif
	return contact;
}

TMNF_HD int TmnfRace_TriggerContact(const TmnfRouteTrigger *trigger,
	const CPlugTree *car_tree, const GmIso4 *car_world_transform)
{
	return trigger_contact(trigger, car_tree, car_world_transform, NULL, 0, NULL);
}

TMNF_HD uint64_t TmnfRace_TriggerContactMask(
	const TmnfRoute *route, const CPlugTree *car_tree,
	const GmIso4 *car_world_transform)
{
	validate_route(route);
	if (!TmnfRoute_ContactMaskSupported(route->metadata))
		race_fail("contact mask cannot represent this route; use individual trigger callbacks");
	uint64_t mask = 0;
	for (uint32_t i = 0; i < route->metadata->checkpoint_count; ++i) {
		if (trigger_contact(&route->checkpoints[i], car_tree, car_world_transform,
				route->trigger_geometry, i + 1, NULL))
			mask |= (uint64_t)1 << i;
	}
	/* Keep every finish geometry. Single-lap overflow alternatives share a
	 * signal; multilap routes preserve each alternative's contact edge. */
	for (uint32_t i = 0; i < route->metadata->finish_count; ++i) {
		if (trigger_contact(&route->finish[i], car_tree, car_world_transform,
				route->trigger_geometry, route->metadata->checkpoint_count + i + 1, NULL))
			mask |= TmnfRoute_FinishContactBit(route->metadata, i);
	}
	return mask;
}

TMNF_HD static void detect_trigger_surface(CHmsCollisionManager_SZone *zone,
	CPlugTree *car_tree, const GmIso4 *car_world_transform,
	const GmBoxAligned *trigger_box, CPlugSurface *surface, const GmIso4 *iso)
{
	CPlugTree trigger_tree;
	memset(&trigger_tree, 0, sizeof(trigger_tree));
	trigger_tree.flags = 0x80u;
	trigger_tree.surface = surface;
	SPlugTreeLocatedPair pair = { car_tree, car_world_transform, &trigger_tree, iso };
	(void)CHmsCollisionManager_SZone_ComputeCollisionTree2RootOnly(
		zone, &pair, trigger_box);
}

TMNF_HD void TmnfRace_DetectTriggerCollisions(const TmnfRoute *route,
	CHmsCollisionManager_SZone *zone, CPlugTree *car_tree,
	const GmIso4 *car_world_transform, uint32_t car_corpus_ref)
{
	validate_route(route);
	uint32_t checkpoints = route->metadata->checkpoint_count;
	uint32_t count = checkpoints + route->metadata->finish_count;
	if (count >= TMNF_RACE_TRIGGER_CORPUS_BASE)
		race_fail("too many trigger corpus references");
	GmBoxAligned car_box;
	GmBoxAligned_SetMult(&car_box, &car_tree->box, car_world_transform);
	zone->current_corpus1 = car_corpus_ref;
	zone->current_material = 0;
	if (route->trigger_geometry != NULL) {
		const TmnfTrack *geometry = route->trigger_geometry;
		CHmsCollisionManager_SGroup group;
		memset(&group, 0, sizeof(group));
		/* Detection borrows these entries without modifying the mapping. */
		group.static_entries = (HmsStaticCollisionEntry *)geometry->entries;
		group.static_entry_count = geometry->entry_count;
		CHmsCollisionManager_SGroup *previous = zone->static_group;
		zone->static_group = &group;
		CHmsCollisionManager_SZone_DetectStaticTreeWithCorpusOffset(
			zone, car_world_transform, car_tree, TMNF_RACE_TRIGGER_CORPUS_BASE - 1);
		zone->static_group = previous;
		return;
	}
	/* Box-only routes lack the game's trigger collision tree. Preserve
	 * registration order for this approximation; exact playback uses the
	 * mesh sidecar and its spatial traversal above. Synthetic routes may
	 * omit the loader's precomputed registration order. */
	for (uint32_t pass = 0; pass < count; ++pass) {
		uint32_t i = route->trigger_order ? route->trigger_order[pass] : pass;
		const TmnfRouteTrigger *trigger = i < checkpoints
			? &route->checkpoints[i] : &route->finish[i - checkpoints];
		GmBoxAligned box;
		GmBoxAligned_SetMult(&box, &trigger->box, &trigger->transform);
		if (!GmBoxAligned_TestInter(&car_box, &box)) continue;
		zone->current_corpus2 = TMNF_RACE_TRIGGER_CORPUS_BASE + i;
		TriggerMesh mesh;
		build_trigger_mesh(&mesh, &trigger->box);
		detect_trigger_surface(zone, car_tree, car_world_transform,
			&box, &mesh.surface, &trigger->transform);
	}
}

/* --- dense projection ----------------------------------------------------- */

/* Projection plus the offset vector from the projected centerline point to
 * the car, split into the horizontal (XZ) distance and the signed height the
 * corridor rule tests. */
typedef struct {
	TmnfRouteProjection projection;
	float lateral_horizontal;
	float vertical;
} DenseProjection;

TMNF_HD static TmnfRouteProjection project_segment(
	const TmnfRoute *route, const GmVec3 *position, uint32_t segment,
	float *distance_sq, GmVec3 *offset)
{
	const TmnfRouteReferencePoint *points =
		TmnfRoute_GetReferencePoints(route);
	const TmnfRouteReferencePoint *a = &points[segment];
	const TmnfRouteReferencePoint *b = &points[segment + 1];
	float dx = b->position.x - a->position.x;
	float dy = b->position.y - a->position.y;
	float dz = b->position.z - a->position.z;
	float px = position->x - a->position.x;
	float py = position->y - a->position.y;
	float pz = position->z - a->position.z;
	float length_sq = dx * dx + dy * dy + dz * dz;
	if (!(length_sq > 0.0f))
		race_fail("route centerline contains a degenerate segment");
	float t = (px * dx + py * dy + pz * dz) / length_sq;
	if (t < 0.0f)
		t = 0.0f;
	else if (t > 1.0f)
		t = 1.0f;
	float ox = px - t * dx;
	float oy = py - t * dy;
	float oz = pz - t * dz;
	*distance_sq = ox * ox + oy * oy + oz * oz;
	*offset = (GmVec3){ ox, oy, oz };
	TmnfRouteProjection projection = {
		.arc_length = a->arc_length +
			t * (b->arc_length - a->arc_length),
		.half_width = a->half_width +
			t * (b->half_width - a->half_width),
		.segment_index = a->leg_index,
		.centerline_segment_index = segment,
		.segments_tested = 1,
	};
	return projection;
}

TMNF_HD static DenseProjection dense_from_offset(
	TmnfRouteProjection projection, const GmVec3 *offset)
{
	return (DenseProjection){
		.projection = projection,
		.lateral_horizontal = sqrtf(
			offset->x * offset->x + offset->z * offset->z),
		.vertical = offset->y,
	};
}

/* Full-route search (reset, teleport): TmnfRoute_Project selects the segment;
 * the offset vector is recomputed on that segment. */
TMNF_HD static DenseProjection project_full(
	const TmnfRoute *route, const GmVec3 *position)
{
	TmnfRouteProjection projection = TmnfRoute_Project(route, position);
	float distance_sq;
	GmVec3 offset;
	(void)project_segment(
		route, position, projection.centerline_segment_index,
		&distance_sq, &offset);
	return dense_from_offset(projection, &offset);
}

TMNF_HD static DenseProjection project_local(
	const TmnfRoute *route, const TmnfRaceState *state,
	const GmVec3 *position)
{
	uint32_t segment_count =
		TmnfRoute_GetReferencePointCount(route) - 1;
	if (state->centerline_segment >= segment_count)
		race_fail("dense projection cursor is out of range");
	uint32_t minimum =
		state->centerline_segment >
			TMNF_RACE_PROJECTION_WINDOW_SEGMENTS
		? state->centerline_segment -
			TMNF_RACE_PROJECTION_WINDOW_SEGMENTS
		: 0;
	uint32_t maximum = state->centerline_segment +
		TMNF_RACE_PROJECTION_WINDOW_SEGMENTS;
	if (maximum >= segment_count)
		maximum = segment_count - 1;

	float best_distance_sq;
	GmVec3 best_offset;
	TmnfRouteProjection best = project_segment(
		route, position, minimum, &best_distance_sq, &best_offset);
	for (uint32_t segment = minimum + 1;
		segment <= maximum; ++segment) {
		float candidate_distance_sq;
		GmVec3 candidate_offset;
		TmnfRouteProjection candidate =
			project_segment(
				route, position, segment,
				&candidate_distance_sq, &candidate_offset);
		if (candidate_distance_sq < best_distance_sq) {
			best = candidate;
			best_distance_sq = candidate_distance_sq;
			best_offset = candidate_offset;
		}
	}
	best.lateral_offset = sqrtf(best_distance_sq);
	best.segments_tested = maximum - minimum + 1;
	return dense_from_offset(best, &best_offset);
}

TMNF_HD static DenseProjection project_dense(
	const TmnfRoute *route, const TmnfRaceState *state,
	const GmVec3 *position)
{
	/*
	 * A 65-segment dense window (cursor +/- 32) keeps ordinary motion on its
	 * current road branch at self-intersections. More than 32 metres in one
	 * 10 ms race tick is a restore/teleport, so reacquire with one full dense
	 * search. A spatially close move to overlapping geometry remains local.
	 */
	if (!isfinite(position->x) || !isfinite(position->y) ||
		!isfinite(position->z)) {
		race_fail("cannot project a non-finite position");
	}
	float dx = position->x - state->previous_car_transform.t[0];
	float dy = position->y - state->previous_car_transform.t[1];
	float dz = position->z - state->previous_car_transform.t[2];
	float displacement_sq = dx * dx + dy * dy + dz * dz;
	float teleport_distance =
		(float)TMNF_RACE_TELEPORT_DISTANCE_METERS;
	if (displacement_sq > teleport_distance * teleport_distance)
		return project_full(route, position);
	return project_local(route, state, position);
}

TMNF_HD static void apply_projection(
	TmnfRaceState *state, const DenseProjection *dense)
{
	state->arc_length = dense->projection.arc_length;
	state->lateral_offset = dense->projection.lateral_offset;
	state->half_width = dense->projection.half_width;
	state->projection_segment = dense->projection.segment_index;
	state->centerline_segment =
		dense->projection.centerline_segment_index;
	state->corridor_lateral = dense->lateral_horizontal;
	state->corridor_vertical = dense->vertical;
	float allowed = TMNF_RACE_CORRIDOR_WIDTH_FACTOR *
		dense->projection.half_width;
	if (allowed < TMNF_RACE_CORRIDOR_WIDTH_FLOOR_METERS)
		allowed = TMNF_RACE_CORRIDOR_WIDTH_FLOOR_METERS;
	state->outside_corridor =
		dense->lateral_horizontal > allowed ||
		dense->vertical > TMNF_RACE_CORRIDOR_ABOVE_METERS ||
		dense->vertical < -TMNF_RACE_CORRIDOR_BELOW_METERS;
}

/* A fresh player has no contact history (0x004831F0 ResetPlayer clears the
 * checkpoint flags); the first physics step's detection passes decide the
 * first contacts. */
TMNF_HD void TmnfRace_Reset(
	const TmnfRoute *route,
	TmnfRaceState *state,
	const GmIso4 *car_world_transform)
{
	validate_route(route);
	if (state == NULL || car_world_transform == NULL)
		race_fail("reset argument is null");
	memset(state, 0, sizeof(*state));
	state->lap_limit = route->metadata->lap_count;
	state->previous_car_transform = *car_world_transform;
	state->respawn_location = route->start->spawn;
	DenseProjection dense = project_full(
		route, (const GmVec3 *)car_world_transform->t);
	apply_projection(state, &dense);
	state->unwrapped_progress = dense.projection.arc_length;
	state->best_progress = dense.projection.arc_length;
	state->previous_progress = dense.projection.arc_length;
}

TMNF_HD int TmnfRace_UpdateOffTrack(
	TmnfRaceState *state,
	uint32_t grace_ticks,
	uint32_t wheels_in_contact,
	uint32_t wheels_on_ground_plane)
{
	if (state == NULL || grace_ticks == 0 ||
		wheels_on_ground_plane > wheels_in_contact) {
		race_fail("invalid off-track state");
	}
	if (wheels_in_contact == 0)
		return 0;
	if (wheels_on_ground_plane != wheels_in_contact) {
		state->off_track_ticks = 0;
		return 0;
	}
	if (state->off_track_ticks < grace_ticks)
		state->off_track_ticks++;
	return state->off_track_ticks == grace_ticks;
}

TMNF_HD void TmnfRace_BeginStep(const TmnfRoute *route,
	TmnfRaceState *state, TmnfRaceStepResult *result)
{
	validate_route(route);
	if (state == NULL || result == NULL)
		race_fail("step argument is null");
	if (state->finished)
		race_fail("cannot step a finished race");
	memset(result, 0, sizeof(*result));
	if (state->phase == TMNF_RACE_RUNNING) state->elapsed_ticks++;
	TmnfCheckpointSet_Clear(&state->contacts);
}

TMNF_HD int TmnfRace_ProcessContact(const TmnfRoute *route,
	TmnfRaceState *state, uint32_t trigger_index, TmnfRaceStepResult *result)
{
	uint32_t checkpoint_count = route->metadata->checkpoint_count;
	if (trigger_index >= checkpoint_count + route->metadata->finish_count)
		race_fail("race contact trigger index is out of range");
	if (state->finished) return 0;
	if (trigger_index < checkpoint_count) {
		uint32_t i = trigger_index;
		TmnfCheckpointSet_Add(&state->contacts, i);
		/* 0x0047C330 tests this lap's visited flag on every contact.
		 * A held contact can count again immediately after a lap resets it. */
		if (TmnfCheckpointSet_Test(&state->visited, i)) {
			result->checkpoint_repeated = 1;
			return route->checkpoints[i].no_respawn != 0;
		}
		TmnfCheckpointSet_Add(&state->visited, i);
		state->visited_count++;
		state->passed_race_checkpoints++;
		result->checkpoint_accepted = 1;
		result->checkpoint_index = i;
		/* 0x0047C330 OnCheckpoint: a block whose info has +0x120 set keeps
		 * the previous spawn and does not arm the respawn. */
		if (route->checkpoints[i].no_respawn == 0) {
			state->respawn_location = route->checkpoints[i].spawn;
			state->respawn_available = 1;
		}
		return 1;
	}

	TmnfCheckpointSet_Add(&state->contacts, TmnfRoute_FinishContactIndex(route->metadata,
		trigger_index - checkpoint_count));
	/* 0x00480820 OnFinishLine requires the player's racing phase. Ordinary
	 * checkpoints have no such gate and survive the blocked countdown. */
	if (state->phase == TMNF_RACE_COUNTDOWN) return 0;
	if (state->visited_count == checkpoint_count) {
		state->passed_race_checkpoints++;
		state->completed_laps++;
		result->lap_completed = 1;
		if (state->completed_laps == state->lap_limit) {
			if (state->passed_race_checkpoints !=
				(checkpoint_count + 1) * state->lap_limit) {
				race_fail("finished with the wrong checkpoint total");
			}
			state->finished = 1;
			state->finish_time_ms =
				state->elapsed_ticks * TMNF_RACE_TICK_MS;
			result->finished = 1;
			result->race_time_ms = state->finish_time_ms;
		} else {
			const TmnfRouteReferencePoint *origin =
				TmnfRoute_GetReferencePoints(route);
			TmnfCheckpointSet_Clear(&state->visited);
			state->visited_count = 0;
			/* 0x00480820 OnFinishLine -> 0x0047E400 InternalOnCheckpoint
			 * with a null spawn: +0x274 = +0x244, the start spawn. */
			state->respawn_location = route->start->spawn;
			state->projection_segment = origin[0].leg_index;
			state->centerline_segment = 0;
			state->arc_length = 0.0f;
			state->lateral_offset = 0.0f;
			state->half_width = origin[0].half_width;
			state->corridor_lateral = 0.0f;
			state->corridor_vertical = 0.0f;
			state->outside_corridor = 0;
		}
		return 1;
	}
	return 0;
}

TMNF_HD TmnfRaceStepResult TmnfRace_EndStep(const TmnfRoute *route,
	TmnfRaceState *state, const GmIso4 *car_world_transform,
	TmnfRaceStepResult result)
{
	int began_new_lap = result.lap_completed && !state->finished;
	if (state->finished) {
		uint32_t point_count =
			TmnfRoute_GetReferencePointCount(route);
		const TmnfRouteReferencePoint *points =
			TmnfRoute_GetReferencePoints(route);
		state->arc_length = TmnfRoute_GetReferenceLength(route);
		state->lateral_offset = 0.0f;
		state->half_width = points[point_count - 1].half_width;
		state->centerline_segment = point_count - 2;
		state->corridor_lateral = 0.0f;
		state->corridor_vertical = 0.0f;
		state->outside_corridor = 0;
	} else if (!began_new_lap) {
		DenseProjection dense = project_dense(
			route, state, (const GmVec3 *)car_world_transform->t);
		apply_projection(state, &dense);
	}
	/*
	 * Progress is credited only inside the corridor. Outside it the value
	 * freezes: it neither advances (A04's pool glide collected the last
	 * 90 m of the route from 18 m below it) nor reverses, so the
	 * progress-only stuck rule ends an excursion that does not come back
	 * and the potential pays only for corridor progress. A finish contact
	 * is the game's own rule and always anchors the new lap.
	 */
	state->previous_progress = state->unwrapped_progress;
	if (state->finished || began_new_lap) {
		state->unwrapped_progress =
			state->completed_laps *
			TmnfRoute_GetReferenceLength(route);
	} else if (!state->outside_corridor) {
		state->unwrapped_progress =
			state->completed_laps *
				TmnfRoute_GetReferenceLength(route) +
			state->arc_length;
	}
	if (state->unwrapped_progress > state->best_progress)
		state->best_progress = state->unwrapped_progress;
	state->previous_car_transform = *car_world_transform;
	return result;
}

/* Mask-only callers have no within-tick collision order. Keep this helper
 * for synthetic race rules; real physics dispatches sorted contact records. */
TMNF_HD TmnfRaceStepResult TmnfRace_Step(const TmnfRoute *route,
	TmnfRaceState *state, uint64_t trigger_contacts,
	const GmIso4 *car_world_transform)
{
	TmnfRaceStepResult result;
	if (!TmnfRoute_ContactMaskSupported(route->metadata))
		race_fail("contact mask cannot represent this route; use individual trigger callbacks");
	TmnfRace_BeginStep(route, state, &result);
	uint32_t checkpoints = route->metadata->checkpoint_count;
	for (uint32_t i = 0; i < checkpoints; ++i)
		if (trigger_contacts & (UINT64_C(1) << i))
			(void)TmnfRace_ProcessContact(route, state, i, &result);
	for (uint32_t i = 0; i < route->metadata->finish_count; ++i)
		if (trigger_contacts & TmnfRoute_FinishContactBit(route->metadata, i))
			(void)TmnfRace_ProcessContact(route, state, checkpoints + i, &result);
	return TmnfRace_EndStep(route, state, car_world_transform, result);
}

TMNF_HD const GmIso4 *TmnfRace_RespawnLocation(const TmnfRaceState *state)
{
	if (state == NULL)
		race_fail("respawn location argument is null");
	if (state->finished)
		race_fail("cannot respawn a finished race");
	return state->respawn_available ? &state->respawn_location : NULL;
}
