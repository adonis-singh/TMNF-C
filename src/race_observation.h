/* Bounded, car-relative gate geometry. Pure observation: no race mutation. */
#ifndef TMNF_RACE_OBSERVATION_H
#define TMNF_RACE_OBSERVATION_H

#include "race.h"
#include "gm.h"
#include <string.h>
#include <math.h>

/* Seven nearest outstanding checkpoints and a dedicated finish slot.
 * Distance is squared distance to the transformed trigger-box centre; ties
 * keep the smaller checkpoint array index. All checkpoints remain legal in
 * any order. Omission is explicit, never interpreted as completion. */
enum { TMNF_GATE_CHECKPOINT_SLOTS = 7, TMNF_GATE_SLOTS = 8 };

typedef struct {
	GmVec3 center;
	GmMat3 orientation;           /* trigger axes in car frame */
	GmVec3 half_extent;           /* exact root-box extents, trigger frame */
	float valid;
	float is_finish;
	float ready;                 /* finish needs every checkpoint */
	float respawnable;
	uint32_t checkpoint_index;    /* diagnostic identity; not actor input */
} TmnfGateObservation;

typedef struct {
	TmnfGateObservation gates[TMNF_GATE_SLOTS];
	uint32_t remaining_checkpoints;
	uint32_t omitted_checkpoints;
	uint32_t remaining_laps;
	uint32_t finished;
} TmnfGateObservations;

_Static_assert(sizeof(TmnfGateObservation) == 80, "gate observation layout");
_Static_assert(sizeof(TmnfGateObservations) == 656, "gate batch layout");

TMNF_HD static inline GmVec3 tmnf_gate_center(
	const TmnfRouteTrigger *gate)
{
	GmVec3 center = gate->box.center;
	GmVec3_Mult_Iso4(&center, &gate->transform);
	return center;
}

TMNF_HD static inline void tmnf_observe_gate(
	const TmnfRouteTrigger *gate, const GmIso4 *inverse_car,
	uint32_t index, int is_finish, int ready, TmnfGateObservation *out)
{
	out->center = tmnf_gate_center(gate);
	GmVec3_Mult_Iso4(&out->center, inverse_car);
	GmIso4 relative = gate->transform;
	GmIso4_Mult(&relative, inverse_car);
	memcpy(out->orientation.m, relative.m, sizeof(out->orientation.m));
	out->half_extent = gate->box.half_extent;
	out->valid = 1.0f;
	out->is_finish = (float)is_finish;
	out->ready = (float)ready;
	out->respawnable = (!is_finish && gate->no_respawn == 0) ? 1.0f : 0.0f;
	out->checkpoint_index = index;
}

/* The caller supplies the current post-step car transform, including after
 * respawn/restore. Trigger boxes and transforms are captured game geometry,
 * not route anchors. Zeroed slots are padding; index is UINT32_MAX there and
 * in the finish slot. The output contains no pointers, clock or global
 * coordinates. Finished races have no actionable gates and zero remaining
 * laps. This does not change TmnfObservation or snapshot ABI. */
TMNF_HD static inline void TmnfRace_ObserveGates(
	const TmnfRoute *route, const TmnfRaceState *state,
	const GmIso4 *car_transform, TmnfGateObservations *out)
{
	if (!route || !route->metadata || !route->finish || !state ||
		!car_transform || !out ||
		route->metadata->checkpoint_count > TMNF_RACE_MAX_CHECKPOINTS ||
		(route->metadata->checkpoint_count && !route->checkpoints) ||
		route->metadata->finish_count == 0 ||
		(uint64_t)route->metadata->finish_count + route->metadata->checkpoint_count > 64 ||
		route->metadata->lap_count == 0 ||
		state->completed_laps > route->metadata->lap_count)
		tmnf_fail("invalid gate observation arguments");
	memset(out, 0, sizeof(*out));
	for (uint32_t i = 0; i < TMNF_GATE_SLOTS; ++i)
		out->gates[i].checkpoint_index = UINT32_MAX;
	out->finished = state->finished != 0;
	if (out->finished)
		return;
	out->remaining_laps = route->metadata->lap_count - state->completed_laps;
	float distances[TMNF_GATE_CHECKPOINT_SLOTS];
	uint32_t indices[TMNF_GATE_CHECKPOINT_SLOTS];
	uint32_t selected = 0;
	for (uint32_t i = 0; i < route->metadata->checkpoint_count; ++i) {
		if (state->visited_checkpoints & (UINT64_C(1) << i))
			continue;
		out->remaining_checkpoints++;
		GmVec3 center = tmnf_gate_center(&route->checkpoints[i]);
		float dx = center.x - car_transform->t[0];
		float dy = center.y - car_transform->t[1];
		float dz = center.z - car_transform->t[2];
		float distance = (dx * dx + dy * dy) + dz * dz;
		uint32_t at = 0;
		while (at < selected && distances[at] <= distance)
			at++;
		if (at == TMNF_GATE_CHECKPOINT_SLOTS)
			continue;
		if (selected < TMNF_GATE_CHECKPOINT_SLOTS)
			selected++;
		for (uint32_t j = selected - 1; j > at; --j) {
			distances[j] = distances[j - 1];
			indices[j] = indices[j - 1];
		}
		distances[at] = distance;
		indices[at] = i;
	}
	out->omitted_checkpoints = out->remaining_checkpoints - selected;
	GmIso4 inverse_car;
	GmIso4_SetInverse(&inverse_car, car_transform);
	for (uint32_t i = 0; i < selected; ++i)
		tmnf_observe_gate(&route->checkpoints[indices[i]], &inverse_car,
			indices[i], 0, 1, &out->gates[i]);
	/* Keep the fixed observation width. Expose the nearest finish alternative,
	 * with stable route-index ties; the reference line still uses finish zero. */
	uint32_t finish_index = 0;
	float finish_distance = INFINITY;
	for (uint32_t i = 0; route->metadata->finish_count > 1 && i < route->metadata->finish_count; ++i) {
		GmVec3 center = tmnf_gate_center(&route->finish[i]);
		float dx = center.x - car_transform->t[0];
		float dy = center.y - car_transform->t[1];
		float dz = center.z - car_transform->t[2];
		float distance = (dx * dx + dy * dy) + dz * dz;
		if (distance < finish_distance) {
			finish_distance = distance;
			finish_index = i;
		}
	}
	tmnf_observe_gate(&route->finish[finish_index], &inverse_car, UINT32_MAX, 1,
		out->remaining_checkpoints == 0,
		&out->gates[TMNF_GATE_CHECKPOINT_SLOTS]);
}
#endif
