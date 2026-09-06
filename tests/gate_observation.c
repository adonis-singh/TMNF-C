/* Geometry, obligations and transform invariance, independent of learning. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "race_observation.h"

static GmIso4 identity(float x, float y, float z)
{
	GmIso4 iso = {0};
	iso.m[0] = iso.m[4] = iso.m[8] = 1;
	iso.t[0] = x; iso.t[1] = y; iso.t[2] = z;
	return iso;
}

int main(void)
{
	TmnfRouteMetadata metadata = {0};
	metadata.checkpoint_count = 63; metadata.finish_count = 1; metadata.lap_count = 3;
	TmnfRouteTrigger gates[63] = {0}, finish = {0};
	TmnfRoute route = {0};
	route.metadata = &metadata; route.checkpoints = gates; route.finish = &finish;
	for (int i = 0; i < 63; ++i) {
		gates[i].transform = identity((float)(i + 1), 0, 0);
		gates[i].box.half_extent = (GmVec3){1, 2, 3};
		gates[i].no_respawn = i % 2;
	}
	finish.transform = identity(100, 0, 0);
	finish.box.half_extent = (GmVec3){4, 5, 6};
	TmnfRaceState state = {0}, before;
	GmIso4 car = identity(0, 0, 0);
	TmnfGateObservations out, baseline;
	before = state;
	TmnfRace_ObserveGates(&route, &state, &car, &out);
	assert(memcmp(&before, &state, sizeof(state)) == 0);
	assert(out.remaining_checkpoints == 63 && out.omitted_checkpoints == 56);
	assert(out.remaining_laps == 3 && !out.finished);
	for (int i = 0; i < 7; ++i) {
		assert(out.gates[i].checkpoint_index == (uint32_t)i);
		assert(out.gates[i].center.x == (float)(i + 1));
		assert(out.gates[i].respawnable == (float)(!(i % 2)));
		assert(out.gates[i].valid && out.gates[i].ready && !out.gates[i].is_finish);
	}
	assert(out.gates[7].valid && out.gates[7].is_finish && !out.gates[7].ready);
	assert(out.gates[7].half_extent.z == 6 && !out.gates[7].respawnable);
	/* Any-order visits, including the highest bit; nearby visited gates
	 * disappear even when a more distant earlier checkpoint remains. */
	state.visited_checkpoints = (UINT64_C(1) << 0) | (UINT64_C(1) << 2) | (UINT64_C(1) << 62);
	TmnfRace_ObserveGates(&route, &state, &car, &out);
	assert(out.remaining_checkpoints == 60 && out.gates[0].checkpoint_index == 1);
	assert(out.gates[1].checkpoint_index == 3);
	/* Ties are stable by route index, not memory or visitation order. */
	gates[4].transform = gates[3].transform;
	TmnfRace_ObserveGates(&route, &state, &car, &out);
	assert(out.gates[1].checkpoint_index == 3 && out.gates[2].checkpoint_index == 4);
	state.visited_checkpoints = (UINT64_C(1) << 63) - 1;
	TmnfRace_ObserveGates(&route, &state, &car, &out);
	assert(!out.remaining_checkpoints && out.gates[7].ready);
	for (int i = 0; i < 7; ++i) assert(!out.gates[i].valid);
	/* The race layer clears visits on a new lap; obligations reappear. */
	state.visited_checkpoints = 0; state.completed_laps = 1;
	TmnfRace_ObserveGates(&route, &state, &car, &out);
	assert(out.remaining_laps == 2 && out.remaining_checkpoints == 63 && !out.gates[7].ready);
	state.finished = 1; state.completed_laps = 3;
	TmnfRace_ObserveGates(&route, &state, &car, &out);
	assert(out.finished && !out.remaining_laps);
	for (int i = 0; i < 8; ++i) assert(!out.gates[i].valid);
	/* Zero-checkpoint tracks expose the ready finish in the same slot. */
	state = (TmnfRaceState){0}; metadata.checkpoint_count = 0;
	TmnfRace_ObserveGates(&route, &state, &car, &out);
	assert(out.gates[7].ready && !out.omitted_checkpoints);
	/* Nonzero box centre plus rotated car/trigger; moving the entire scene
	 * must preserve observations. Extents stay in trigger axes. */
	metadata.checkpoint_count = 1;
	gates[0].box.center = (GmVec3){2, 3, 4};
	gates[0].transform = identity(10, 20, 30);
	car = identity(5, 6, 7);
	TmnfRace_ObserveGates(&route, &state, &car, &baseline);
	assert(baseline.gates[0].center.x == 7 && baseline.gates[0].center.y == 17 && baseline.gates[0].center.z == 27);
	GmIso4 scene = identity(100, -50, 200);
	scene.m[0] = 0; scene.m[2] = -1; scene.m[6] = 1; scene.m[8] = 0;
	GmIso4_Mult(&car, &scene); GmIso4_Mult(&gates[0].transform, &scene); GmIso4_Mult(&finish.transform, &scene);
	TmnfRace_ObserveGates(&route, &state, &car, &out);
	/* Signed zero may differ through an inverse; compare numeric values. */
	for (int i = 0; i < 8; ++i) {
		assert(out.gates[i].center.x == baseline.gates[i].center.x);
		assert(out.gates[i].center.y == baseline.gates[i].center.y);
		assert(out.gates[i].center.z == baseline.gates[i].center.z);
		for (int j = 0; j < 9; ++j) assert(out.gates[i].orientation.m[j] == baseline.gates[i].orientation.m[j]);
	}
	/* Nearest alternative, stable ties, fixed width and no race mutation. */
	TmnfRouteTrigger alternatives[2] = {finish, finish};
	alternatives[0].transform = identity(10, 0, 0);
	alternatives[1].transform = identity(3, 0, 0);
	metadata.checkpoint_count = 0; metadata.finish_count = 2;
	route.finish = alternatives;
	memset(&state, 0, sizeof(state)); car = identity(0, 0, 0);
	TmnfRace_ObserveGates(&route, &state, &car, &out);
	assert(out.gates[7].center.x == 3 && out.gates[7].ready);
	alternatives[0].transform = identity(-3, 0, 0);
	TmnfRace_ObserveGates(&route, &state, &car, &out);
	assert(out.gates[7].center.x == -3);
	puts("gate observation: geometry, 63 checkpoints, masking, laps, ties, invariance passed");
	return 0;
}
