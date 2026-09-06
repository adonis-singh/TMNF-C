/* CHmsZoneDynamic top-level force and tick orchestration, transcribed from the
 * 2.11.26 disassembly. Runtime x87 PC=24 is represented by per-operation
 * x87_* rounding from tmnf_fp.h.
 *
 * UNVALIDATED end-to-end: the control flow is complete for the standard
 * car-on-track path, pending linkage of the separately ported vehicle-force
 * entry and collision-response solver plus full PhysicsStep2 golden replay.
 */
#include "physics.h"
#include "tmnf_warp.h"

#include <stdlib.h>

#include "race.h"
#include "tmnf_fp.h"

TMNF_HD static float x87_length3(const GmVec3 *v) {
	float square = x87_add(
		x87_add(x87_mul(v->y, v->y), x87_mul(v->x, v->x)),
		x87_mul(v->z, v->z));
	return x87_sqrt(square);
}

TMNF_HD static TmnfPhysicsCorpus *find_corpus(
	TmnfPhysicsWorld *world, const CHmsCorpus *collision_corpus) {
	for (uint32_t i = 0; i < world->corpus_count; ++i) {
		if (world->corpora[i].collision_corpus == collision_corpus) {
			return &world->corpora[i];
		}
	}
	tmnf_abort();
}

TMNF_HD static void reset_collision_buffer(TmnfPhysicsWorld *world) {
	world->collision_buffer.collisions.count = 0;
}

/* active = 0 is a lane keeping the warp company (tmnf_warp.h): it takes part
 * in the cooperative detection with no work of its own and touches nothing. */
TMNF_HD static void detect_and_respond(
	TmnfPhysicsWorld *world, TmnfPhysicsCorpus *corpus, int active) {
	if (active) {
		reset_collision_buffer(world);
	}
	CHmsCollisionManager_SZone_DetectCollisionsCorpus(
		world->collision_zone, &world->collision_buffer,
		corpus->collision_corpus, active);
	if (!active) {
		return;
	}
	/* The waypoint corpora are static-tree entries of the game's zone, so
	 * this pass detected the car against them with the same predicted iso
	 * (corpus_iso: live_iso of a dynamic corpus), and ComputeCollisionResponse
	 * below is where their contact sink raises OnCheckpoint/OnFinishLine
	 * (analysis/game_rules.md, "Trigger timing"). The port's zone has no
	 * trigger corpora; the route's volumes are tested here instead. */
	if (world->route != NULL && corpus->vehicle != NULL) {
		world->trigger_contacts |= TmnfRace_TriggerContactMask(
			world->route, corpus->collision_corpus->tree,
			corpus->collision_corpus->live_iso);
	}
	world->response_zone.collisions =
		&world->collision_buffer.collisions;
	CHmsZoneDynamic_ComputeCollisionResponse(&world->response_zone);
}

/* 0x0055F3B0. The position argument is intentionally unused: a uniform field
 * returns the same vector everywhere. */
TMNF_HD int CHmsForceFieldUniform_GetValue(
	const CHmsForceFieldUniform *self, const GmVec3 *position, GmVec3 *out) {
	(void)position;
	if (self->active == 0) {
		return 0;
	}
	*out = self->value;
	return 1;
}

/* 0x005481A0 */
TMNF_HD void CHmsZoneDynamic_ComputeCorpusForces(
	TmnfPhysicsWorld *world, TmnfPhysicsCorpus *corpus, float dt) {
	CHmsDyna *dyna = corpus->dyna;
	CHmsDynaParams *params = dyna->params;

	CHmsDyna_ValidateDynamicState(dyna);
	if ((corpus->scene_flags & 0x00100000u) != 0) {
		const GmVec3 zero = { 0.0f, 0.0f, 0.0f };
		CHmsDyna_SetForce(dyna, &zero);
		CHmsDyna_SetTorque(dyna, &zero);
		return;
	}

	GmVec3 force = { 0.0f, 0.0f, 0.0f };
	for (uint32_t i = 0; i < world->force_field_count; ++i) {
		GmVec3 value;
		if (CHmsForceFieldUniform_GetValue(
				&world->force_fields[i], &dyna->liveState->pos, &value)) {
			float scale = x87_mul(params->forceFieldScale, params->mass);
			force.x = x87_add(x87_mul(scale, value.x), force.x);
			force.y = x87_add(x87_mul(value.y, scale), force.y);
			force.z = x87_add(x87_mul(scale, value.z), force.z);
		}
	}

	GmVec3 linear_speed;
	CHmsDyna_GetLinearSpeed(dyna, &linear_speed);
	float linear_drag = -x87_mul(
		world->linear_drag_scale, params->dragLinear);
	force.x = x87_add(x87_mul(linear_drag, linear_speed.x), force.x);
	force.y = x87_add(x87_mul(linear_speed.y, linear_drag), force.y);
	force.z = x87_add(x87_mul(linear_drag, linear_speed.z), force.z);
	CHmsDyna_SetForce(dyna, &force);

	if (dyna->mode == 1) {
		GmVec3 angular_speed;
		CHmsDyna_GetAngularSpeed(dyna, &angular_speed);
		float angular_drag = -x87_mul(
			world->angular_drag_scale, params->dragAngular);
		GmVec3 torque = {
			x87_mul(angular_drag, angular_speed.x),
			x87_mul(angular_speed.y, angular_drag),
			x87_mul(angular_drag, angular_speed.z),
		};
		CHmsDyna_SetTorque(dyna, &torque);
	}

	if (corpus->vehicle_compute != NULL) {
		TMNFVehicleComputeForces_PhysicsStep2Adapter(
			corpus->vehicle_compute, corpus->vehicle, dt);
	}
}

/* Executes one adaptive integration/collision sequence for a dynamic corpus.
 * The substep count depends on the corpus's speed, so the loop runs for the
 * warp's largest count and a lane past its own count only accompanies the
 * cooperative detection (tmnf_warp.h); on the host the two counts agree. A
 * clean corpus takes no step and accompanies every detection. */
TMNF_HD static void step_dynamic_corpus(
	TmnfPhysicsWorld *world, TmnfPhysicsCorpus *corpus, float dt) {
	CHmsDyna *dyna = corpus->dyna;
	int active = dyna->dirtyFlag != 0;
	uint32_t substeps = 0;
	float remaining = dt;
	float step = 0.0f;
	if (active) {
		CHmsDyna_CopyStateToTemp(dyna);

		GmVec3 linear_speed;
		GmVec3 angular_speed;
		CHmsDyna_GetLinearSpeed(dyna, &linear_speed);
		CHmsDyna_GetAngularSpeed(dyna, &angular_speed);
		float speed_sum = x87_add(
			x87_length3(&linear_speed), x87_length3(&angular_speed));
		float numerator = x87_mul(dt, speed_sum);
		float quotient = x87_div(numerator, dyna->params->substepLen);
		substeps = (uint32_t)(ftol(F(quotient)) + 1);
		if (substeps > 1000) {
			substeps = 1000;
		}
		if (substeps > 1) {
			step = x87_r24(F(dt) / (double)substeps);
		}
	}

	uint32_t rounds = tmnf_warp_max_u32(substeps);
	for (uint32_t i = 1; i < rounds; ++i) {
		int stepping = active && i < substeps;
		if (stepping) {
			CHmsZoneDynamic_ComputeCorpusForces(world, corpus, step);
			CHmsDyna_DoPreCollisionDynamic(dyna, step);
		}
		detect_and_respond(world, corpus, stepping);
		if (stepping) {
			CHmsDyna_DoPostCollisionDynamic(dyna);
			remaining = x87_sub(remaining, step);
		}
	}

	if (active) {
		CHmsZoneDynamic_ComputeCorpusForces(world, corpus, remaining);
		CHmsDyna_DoPreCollisionDynamic(dyna, remaining);
	}
	detect_and_respond(world, corpus, active);
	if (active) {
		CHmsDyna_DoPostCollisionDynamic(dyna);
		CHmsDyna_CopyTempToState(dyna);
	}
}

/* 0x00549C90 */
TMNF_HD void CHmsZoneDynamic_PhysicsStep2(TmnfPhysicsWorld *world, uint32_t tick_ms) {
	float dt = x87_mul((float)(int32_t)tick_ms, 0.001f);
	world->trigger_contacts = 0;

	/* Initial force/integration pass over the zone's dynamic corpora. */
	for (uint32_t i = 0; i < world->corpus_count; ++i) {
		TmnfPhysicsCorpus *corpus = &world->corpora[i];
		if ((corpus->scene_flags & 0x0001E000u) == 0) {
			CHmsZoneDynamic_ComputeCorpusForces(world, corpus, dt);
			CHmsDyna_DoPreCollisionDynamic(corpus->dyna, dt);
		}
	}

	CHmsCollisionManager_SZone_PrepareCollisions(world->collision_zone);

	/* The game has exactly five 0x44-byte collision groups. Static groups are
	 * excluded here and are queried by the dynamic groups' detection pass. */
	for (uint32_t group_index = 0; group_index < 5; ++group_index) {
		CHmsCollisionManager_SGroup *group =
			&world->collision_zone->groups[group_index];
		if (group->is_static != 0) {
			continue;
		}
		for (uint32_t i = 0; i < group->corpus_count; ++i) {
			TmnfPhysicsCorpus *corpus =
				find_corpus(world, group->corpora[i]);
			if (corpus->dyna == NULL) {
				detect_and_respond(world, corpus, 1);
				continue;
			}
			step_dynamic_corpus(world, corpus, dt);
		}
	}

	/* The final scene callback at PhysicsStep2+0x380 is a no-op for the
	 * standard CSceneVehicleCar runtime target observed on the A01 oracle. */
}
