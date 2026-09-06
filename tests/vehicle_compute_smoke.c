#include <assert.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "vehicle_compute.h"

struct TMNFVehicleModel6Context {
	int marker;
};

typedef struct {
	CSceneVehicleCar vehicle;
	CSceneVehicleCarTuning vehicle_tuning;
	CHmsStateDyna dyna_state;
	CHmsStateDyna committed_state;
	CHmsDynaParams dyna_params;
	CHmsDyna dyna;
	CHmsCorpus corpus;
	CHmsCorpus *corpora[1];
	void *zones[1];
	CHmsItem item;

	CSceneVehicleCarTuningAux aux_tuning;
	CSceneVehicleCarAuxContext aux;
	TMNFVehicleContactTuning contact_tuning;
	TMNFVehicleContactTimer timer;
	TMNFVehicleContactContext contact;
	TMNFVehicleComputeTuning compute_tuning;
	TMNFVehicleComputeState compute_state;
	TMNFVehicleModel6Context model6;
	TMNFVehicleComputeContext compute;

	float zero_position;
	float zero_value;
	float one_value;
	float two_value;
	CFuncKeysReal zero_curve;
	CFuncKeysReal one_curve;
	CFuncKeysReal two_curve;
	TMNFVehicleAuxCurve one_aux_curve;

	int finish_calls;
	int post_calls;
} Fixture;

static int model6_calls;

static void set_identity(GmMat3 *matrix)
{
	memset(matrix, 0, sizeof(*matrix));
	matrix->m[0] = 1.0f;
	matrix->m[4] = 1.0f;
	matrix->m[8] = 1.0f;
}

static void finish_integration(void *runtime, CSceneVehicleCar *vehicle)
{
	Fixture *fixture = runtime;

	assert(vehicle == &fixture->vehicle);
	fixture->finish_calls++;
}

static void post_force(void *runtime, CSceneVehicleCar *vehicle)
{
	Fixture *fixture = runtime;

	assert(vehicle == &fixture->vehicle);
	fixture->post_calls++;
}

void VehicleModel6_ComputeForces(
	TMNFVehicleComputeContext *context, float dt,
	const GmVec3 *existing_force, float slope_adherence,
	float slope_secondary, const GmVec3 *linear_speed,
	const GmVec3 *angular_speed, float steering_angle,
	int has_ground_material,
	const CSceneVehicleMaterialBlendableVals *ground_material,
	int *air_control_reset, float *effect_curve_position)
{
	assert(context->model6->marker == 6);
	assert(dt == 0.1f);
	assert(existing_force->x == 3.0f);
	assert(existing_force->y == 4.0f);
	assert(existing_force->z == 0.0f);
	assert(slope_adherence == 1.0f);
	assert(slope_secondary == 1.0f);
	assert(linear_speed->x == 0.0f);
	assert(linear_speed->y == 0.0f);
	assert(linear_speed->z == 0.0f);
	assert(angular_speed->x == 0.0f);
	assert(angular_speed->y == 0.0f);
	assert(angular_speed->z == 0.0f);
	assert(steering_angle == 0.0f);
	assert(has_ground_material == 0);
	assert(ground_material->acceleration == 0.0f);
	assert(ground_material->braking == 0.0f);
	assert(ground_material->steering == 0.0f);
	assert(ground_material->lateral_grip == 0.0f);

	model6_calls++;
	*air_control_reset = 1;
	*effect_curve_position = 0.5f;
}

static CFuncKeysReal constant_curve(
	const float *position, const float *value)
{
	CFuncKeysReal curve = {
		.keys = {
			.count = 1,
			.positions = position,
		},
		.values = value,
		.interpolation = 1,
	};

	CFuncKeys_Compile(&curve.keys);
	return curve;
}

static void fixture_init(Fixture *fixture)
{
	memset(fixture, 0, sizeof(*fixture));
	fixture->zero_position = 0.0f;
	fixture->zero_value = 0.0f;
	fixture->one_value = 1.0f;
	fixture->two_value = 2.0f;
	fixture->zero_curve = constant_curve(
		&fixture->zero_position, &fixture->zero_value);
	fixture->one_curve = constant_curve(
		&fixture->zero_position, &fixture->one_value);
	fixture->two_curve = constant_curve(
		&fixture->zero_position, &fixture->two_value);
	fixture->one_aux_curve.count = 1;
	fixture->one_aux_curve.positions = &fixture->zero_position;
	fixture->one_aux_curve.values = &fixture->one_value;
	fixture->one_aux_curve.interpolation = 1;
	{
		CFuncKeys keys = { 1, &fixture->zero_position, NULL, NULL };
		CFuncKeys_Compile(&keys);
		fixture->one_aux_curve.lower_bounds = keys.lower_bounds;
		fixture->one_aux_curve.upper_bounds = keys.upper_bounds;
	}

	set_identity(&fixture->dyna_state.rot);
	set_identity(&fixture->dyna_state.invInertiaWorld);
	set_identity(&fixture->dyna_params.invInertiaBody);
	fixture->dyna_state.quat.w = 1.0f;
	fixture->dyna_state.force = (GmVec3){ 3.0f, 4.0f, 0.0f };
	fixture->dyna_params.mass = 1000.0f;
	fixture->dyna.params = &fixture->dyna_params;
	fixture->dyna.liveState = &fixture->dyna_state;
	fixture->dyna.stateB = &fixture->committed_state;
	fixture->corpus.dyna = &fixture->dyna;
	fixture->corpora[0] = &fixture->corpus;
	fixture->item.corpora = fixture->corpora;
	fixture->item.zones = fixture->zones;
	fixture->item.corpus_count = 1;

	fixture->vehicle.hms_item = &fixture->item;
	fixture->vehicle.dyna_state = &fixture->dyna_state;
	fixture->vehicle.dyna_params = &fixture->dyna_params;
	fixture->vehicle.tuning = &fixture->vehicle_tuning;

	fixture->aux_tuning.air_vertical_curve = &fixture->one_aux_curve;
	fixture->aux.vehicle = &fixture->vehicle;
	fixture->aux.tuning = &fixture->aux_tuning;
	fixture->aux.integration_flags = 2;
	fixture->aux.runtime = fixture;
	fixture->aux.finish_integration = finish_integration;

	fixture->contact_tuning.friction_model = 5;
	fixture->contact_tuning.curves = NULL;
	fixture->contact.vehicle = &fixture->vehicle;
	fixture->contact.tuning = &fixture->contact_tuning;
	fixture->contact.timer = &fixture->timer;
	fixture->timer.tick_time = 100;

	fixture->compute_tuning.normalized_force_divisor = 2.0f;
	fixture->compute_tuning.effect_curve = &fixture->two_curve;
	fixture->compute_tuning.effect_curve_bias = 1.0f;
	fixture->compute_tuning.contact_rise_curve = &fixture->zero_curve;
	fixture->compute_tuning.contact_decay_curve = &fixture->zero_curve;
	fixture->compute_state.simulation_gate = 0.0f;
	fixture->compute_state.history_force_limit = 10.0f;
	fixture->compute_state.history_force_scale = 1.0f;
	fixture->compute_state.spring_value_limit = 1.0f;
	fixture->model6.marker = 6;

	fixture->compute.vehicle = &fixture->vehicle;
	fixture->compute.item = &fixture->item;
	fixture->compute.aux = &fixture->aux;
	fixture->compute.contact = &fixture->contact;
	fixture->compute.tuning = &fixture->compute_tuning;
	fixture->compute.state = &fixture->compute_state;
	fixture->compute.model6 = &fixture->model6;
	fixture->compute.runtime = fixture;
	fixture->compute.post_force = post_force;
}

static void expect_abort(Fixture *fixture)
{
	pid_t child = fork();
	int status;

	assert(child >= 0);
	if (child == 0) {
		TMNFVehicleComputeForces_PhysicsStep2Adapter(
			&fixture->compute, &fixture->vehicle, 0.1f);
		_exit(0);
	}
	assert(waitpid(child, &status, 0) == child);
	assert(WIFSIGNALED(status));
	assert(WTERMSIG(status) == SIGABRT);
}

static void fixture_destroy(Fixture *fixture)
{
	CFuncKeys_Release(&fixture->zero_curve.keys);
	CFuncKeys_Release(&fixture->one_curve.keys);
	CFuncKeys_Release(&fixture->two_curve.keys);
	free((void *)fixture->one_aux_curve.lower_bounds);
}

int main(void)
{
	Fixture fixture;

	fixture_init(&fixture);
	TMNFVehicleComputeForces_PhysicsStep2Adapter(
		&fixture.compute, &fixture.vehicle, 0.1f);
	assert(model6_calls == 1);
	assert(fixture.finish_calls == 1);
	assert(fixture.post_calls == 1);
	assert(fixture.compute_state.normalized_force.x == 1.5f);
	assert(fixture.compute_state.normalized_force.y == 2.0f);
	assert(fixture.compute_state.normalized_force.z == 0.0f);
	assert(fixture.compute_state.effect_accumulator == 0.3f);
	assert(fixture.compute_state.last_force_tick == 100);
	fixture_destroy(&fixture);

	fixture_init(&fixture);
	fixture.compute.fake_contacts_active = 1;
	expect_abort(&fixture);
	fixture_destroy(&fixture);

	fixture_init(&fixture);
	fixture.compute.water_forces_active = 1;
	expect_abort(&fixture);
	fixture_destroy(&fixture);

	fixture_init(&fixture);
	fixture.contact_tuning.friction_model = 4;
	expect_abort(&fixture);
	fixture_destroy(&fixture);
	return 0;
}
