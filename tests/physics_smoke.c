#include <string.h>
#include <stdio.h>

#include "physics.h"
#include "tmnf_fp.h"
#include "vehicle.h"

static void set_identity(GmMat3 *matrix) {
	memset(matrix, 0, sizeof(*matrix));
	matrix->m[0] = 1.0f;
	matrix->m[4] = 1.0f;
	matrix->m[8] = 1.0f;
}

int main(void) {
	CHmsStateDyna live = { 0 };
	CHmsStateDyna committed = { 0 };
	live.quat.w = 1.0f;
	set_identity(&live.rot);
	set_identity(&live.invInertiaWorld);

	CHmsDynaParams params = {
		.mass = 1000.0f,
		.substepLen = 1.0f,
		.forceFieldScale = 1.0f,
	};
	set_identity(&params.invInertiaBody);
	CHmsDyna dyna = {
		.params = &params,
		.stateB = &committed,
		.liveState = &live,
		.dirtyFlag = 1,
		.mode = 1,
	};

	CHmsCorpus collision_corpus = {
		.dyna = &dyna,
		.live_iso = (const GmIso4 *)&live.rot,
	};
	CSceneVehicleCarWheel wheels[4] = { 0 };
	CSceneVehicleCar car = {
		.dyna_state = &live,
		.dyna_params = &params,
		.wheels = wheels,
		.wheel_count = 4,
	};
	TmnfPhysicsCorpus corpus = {
		.collision_corpus = &collision_corpus,
		.dyna = &dyna,
		.vehicle = &car,
	};

	CollisionRuntime collision_runtime;
	CollisionRuntime_Init(&collision_runtime);
	CHmsCollisionManager_SZone collision_zone;
	CHmsCollisionManager_SZone_Init(
		&collision_zone, &collision_runtime);
	CHmsCollisionBuffer collision_buffer;
	CHmsCollisionBuffer_Init(&collision_buffer);
	CHmsForceFieldUniform gravity = {
		.active = 1,
		.value = { 0.0f, -9.81f, 0.0f },
	};
	TmnfPhysicsWorld world = {
		.force_fields = &gravity,
		.force_field_count = 1,
		.corpora = &corpus,
		.corpus_count = 1,
		.collision_zone = &collision_zone,
		.collision_buffer = collision_buffer,
	};

	TMNFRaceInputs input = {
		.steer_right_time = 1,
		.steer_right = 1,
		.accelerate_time = 1,
		.accelerate = 1,
	};
	CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(&input, &car);
	CHmsZoneDynamic_PhysicsStep2(&world, 10);

	float scale = x87_mul(params.forceFieldScale, params.mass);
	float force_y = x87_mul(gravity.value.y, scale);
	float inv_mass = x87_r24(1.0 / F(params.mass));
	float dt = x87_mul(10.0f, 0.001f);
	float expected_speed_y = x87_mul(
		x87_mul(force_y, inv_mass), dt);
	if (live.pos.x != 0.0f || live.pos.y != 0.0f ||
		live.pos.z != 0.0f ||
		live.linVel.x != 0.0f ||
		live.linVel.y != expected_speed_y ||
		live.linVel.z != 0.0f ||
		car.input_gas != 1.0f ||
		car.input_brake != 0.0f ||
		car.input_steer != 1.0f) {
		fprintf(stderr,
			"step mismatch pos=(%g,%g,%g) speed=(%g,%g,%g) "
			"expected_y=%g bits=(%a,%a) input=(%g,%g,%g)\n",
			live.pos.x, live.pos.y, live.pos.z,
			live.linVel.x, live.linVel.y, live.linVel.z,
			expected_speed_y, live.linVel.y, expected_speed_y,
			car.input_gas, car.input_brake, car.input_steer);
		return 1;
	}

	CHmsCollisionBuffer_Destroy(&world.collision_buffer);
	CHmsCollisionManager_SZone_Destroy(&collision_zone);
	return 0;
}
