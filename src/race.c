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
		metadata->finish_count == 0 ||
		(uint64_t)metadata->finish_count + metadata->checkpoint_count > 64 ||
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
	const GmBoxAligned *trigger_world_box, const TriggerMesh *mesh,
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
			car_tree->surface, &car_world, &mesh->surface,
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
				tree->surface, &world, &mesh->surface, trigger_iso,
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

TMNF_HD int TmnfRace_TriggerContact(
	const TmnfRouteTrigger *trigger,
	const CPlugTree *car_tree,
	const GmIso4 *car_world_transform)
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

	TriggerMesh mesh;
	build_trigger_mesh(&mesh, &trigger->box);
	CollisionRuntime runtime;
	CollisionRuntime_Init(&runtime);
	/* The contacts themselves are discarded; only their existence counts.
	 * A device thread cannot allocate, so it records into a fixed-capacity
	 * array and traps if a trigger ever produces more. */
	CHmsCollisionBuffer buffer;
#if defined(__CUDA_ARCH__)
	SHmsPhysicalCollision storage[TRIGGER_CONTACT_CAPACITY];
	buffer.collisions.count = 0;
	buffer.collisions.data = storage;
	buffer.collisions.capacity = TRIGGER_CONTACT_CAPACITY;
#else
	CHmsCollisionBuffer_Init(&buffer);
#endif
	int contact = car_tree_contact(
		car_tree, car_world_transform, &trigger_box, &mesh,
		&trigger->transform, &buffer, &runtime);
#if !defined(__CUDA_ARCH__)
	CHmsCollisionBuffer_Destroy(&buffer);
#endif
	return contact;
}

TMNF_HD uint64_t TmnfRace_TriggerContactMask(
	const TmnfRoute *route, const CPlugTree *car_tree,
	const GmIso4 *car_world_transform)
{
	validate_route(route);
	uint64_t mask = 0;
	for (uint32_t i = 0; i < route->metadata->checkpoint_count; ++i) {
		if (TmnfRace_TriggerContact(
				&route->checkpoints[i], car_tree, car_world_transform))
			mask |= (uint64_t)1 << i;
	}
	/* Each alternative owns an edge bit: merging them would miss entering a
	 * second finish while still touching a previously rejected first one. */
	for (uint32_t i = 0; i < route->metadata->finish_count; ++i) {
		if (TmnfRace_TriggerContact(&route->finish[i], car_tree, car_world_transform))
			mask |= UINT64_C(1) << (TMNF_RACE_FINISH_CONTACT_BIT - i);
	}
	return mask;
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

TMNF_HD TmnfRaceStepResult TmnfRace_Step(
	const TmnfRoute *route,
	TmnfRaceState *state,
	uint64_t trigger_contacts,
	const GmIso4 *car_world_transform)
{
	validate_route(route);
	if (state == NULL || car_world_transform == NULL)
		race_fail("step argument is null");
	if (state->finished)
		race_fail("cannot step a finished race");

	TmnfRaceStepResult result = {0};
	const uint32_t checkpoint_count = route->metadata->checkpoint_count;
	state->elapsed_ticks++;

	/* The game's per-checkpoint flags (0x0047E400 InternalOnCheckpoint)
	 * make every contact after the first of a lap inert; the contact edge
	 * against the previous tick's passes is the same rule. */
	uint64_t entered = trigger_contacts & ~state->trigger_contacts;
	state->trigger_contacts = trigger_contacts;

	for (uint32_t i = 0; i < checkpoint_count; ++i) {
		uint64_t bit = (uint64_t)1 << i;
		if ((entered & bit) == 0)
			continue;
		if (state->visited_checkpoints & bit) {
			result.checkpoint_repeated = 1;
			continue;
		}
		state->visited_checkpoints |= bit;
		state->visited_count++;
		state->passed_race_checkpoints++;
		result.checkpoint_accepted = 1;
		result.checkpoint_index = i;
		/* 0x0047C330 OnCheckpoint: a block whose info has +0x120 set keeps
		 * the previous spawn and does not arm the respawn. */
		if (route->checkpoints[i].no_respawn == 0) {
			state->respawn_location = route->checkpoints[i].spawn;
			state->respawn_available = 1;
		}
	}

	int entered_finish =
		(entered & (UINT64_MAX << (64 - route->metadata->finish_count))) != 0;
	int began_new_lap = 0;
	if (entered_finish && state->visited_count == checkpoint_count) {
		state->passed_race_checkpoints++;
		state->completed_laps++;
		result.lap_completed = 1;
		if (state->completed_laps == route->metadata->lap_count) {
			if (state->passed_race_checkpoints !=
				route->metadata->total_race_checkpoints) {
				race_fail("finished with the wrong checkpoint total");
			}
			state->finished = 1;
			state->finish_time_ms =
				state->elapsed_ticks * TMNF_RACE_TICK_MS;
			result.finished = 1;
			result.race_time_ms = state->finish_time_ms;
		} else {
			const TmnfRouteReferencePoint *origin =
				TmnfRoute_GetReferencePoints(route);
			state->visited_checkpoints = 0;
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
			began_new_lap = 1;
		}
	}

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

TMNF_HD const GmIso4 *TmnfRace_RespawnLocation(const TmnfRaceState *state)
{
	if (state == NULL)
		race_fail("respawn location argument is null");
	if (state->finished)
		race_fail("cannot respawn a finished race");
	return state->respawn_available ? &state->respawn_location : NULL;
}
