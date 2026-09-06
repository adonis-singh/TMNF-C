#include <assert.h>
#include <string.h>

#include "vehicle_aux.h"

static void finish_integration(void *runtime, CSceneVehicleCar *vehicle)
{
	int *calls = runtime;

	(void)vehicle;
	(*calls)++;
}

int main(void)
{
	CSceneVehicleCar vehicle;
	CSceneVehicleCarTuning tuning;
	CSceneVehicleCarWheel wheel;
	CSceneVehicleCarWheelAux wheel_aux;
	CSceneVehicleCarTuningAux tuning_aux;
	CSceneVehicleCarAuxContext context;
	CHmsStateDyna state;
	int finish_calls = 0;
	GmVec3 angular_speed = { 1.0f, -2.0f, 3.0f };

	memset(&vehicle, 0, sizeof(vehicle));
	memset(&tuning, 0, sizeof(tuning));
	memset(&wheel, 0, sizeof(wheel));
	memset(&wheel_aux, 0, sizeof(wheel_aux));
	memset(&tuning_aux, 0, sizeof(tuning_aux));
	memset(&context, 0, sizeof(context));
	memset(&state, 0, sizeof(state));

	assert(CSceneVehicleCar_GetRouletteValue01(0, 7) == 0.0f);
	assert(CSceneVehicleCar_GetRouletteValue01(4, 7) == 0.5f);
	assert(CSceneVehicleCar_GetRouletteValue01(6, 7) == 1.0f);
	assert(
		CSceneVehicleCar_GetRouletteBoostFactorFromValue01(0.5f)
		== 1.5f);

	context.vehicle = &vehicle;
	context.tuning = &tuning_aux;
	context.wheels = &wheel_aux;
	context.wheel_count = 1;
	context.roulette_modulus = 7;
	context.runtime = &finish_calls;
	context.finish_integration = finish_integration;
	vehicle.tuning = &tuning;
	vehicle.wheels = &wheel;
	vehicle.wheel_count = 1;
	vehicle.dyna_state = &state;
	state.rot.m[0] = 1.0f;
	state.rot.m[4] = 1.0f;
	state.rot.m[8] = 1.0f;

	CSceneVehicleCar_EnableTurbo(
		&context, 100, 20, 2.0f, TMNF_TURBO_NORMAL, 0);
	CSceneVehicleCar_UpdateTurbo(&context, 110);
	assert(context.turbo_progress == 0.5f);
	assert(context.turbo_factor == 2.0f);
	context.turbo_epoch_tick = 100;
	CSceneVehicleCar_EnableTurbo(
		&context, 106, 20, 3.0f, TMNF_TURBO_ROULETTE, 1);
	assert(context.roulette_value == 1.0f);
	assert(context.turbo_factor == 6.0f);

	tuning.suspension_model = 0;
	tuning.suspension_stiffness = 10.0f;
	tuning.suspension_damping = 2.0f;
	tuning.suspension_rest_length = 1.0f;
	wheel.real_time.damper_absorb = 0.5f;
	wheel.real_time.field04 = 0.1f;
	wheel_aux.surface_source.t[1] = 2.0f;
	CSceneVehicleCar_WheelIntegrate(&context, 0, 0.1f);
	assert(wheel.real_time.field08 == 0.0f);
	assert(wheel.real_time.damper_absorb > 0.5f);
	assert(wheel_aux.surface_location.t[1] < 2.0f);

	tuning.engine_model = 5;
	context.air_control_locked = 0;
	CSceneVehicleCar_ComputeAirControl(
		&context, &angular_speed, 200, 1, 1);
	assert(context.air_control_tick == 200);
	assert(context.air_control_speed.y == -2.0f);

	tuning_aux.steering_slew_rate = 1.0f;
	vehicle.input_steer = 1.0f;
	context.steering_value = 0.0f;
	context.integration_flags = 0;
	CSceneVehicleCar_IntegrateVehicle(&context, 0.1f);
	assert(context.steering_value == 0.1f);
	assert(finish_calls == 1);
	return 0;
}
