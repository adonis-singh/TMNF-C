#include "vehicle_respawn.h"

#include <string.h>

#include "tmnf_fp.h"
#include "vehicle_aux.h"
#include "vehicle_compute.h"
#include "vehicle_contact.h"
#include "vehicle_model6.h"

#define respawn_fail(message) tmnf_fail("respawn: " message)

/* Spelled out per use: the device pass cannot read file-scope aggregates. */
#define IDENTITY3_INIT \
	{ { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f } }

/* 0x007BCA20 CSceneVehicleCar::SSimulationWheel::SState::Reset. Bytes
 * 0x0e..0x0f are not written (the store at +0x0c is 16-bit). */
TMNF_HD static void wheel_state_reset(CSceneVehicleCarWheelState *state)
{
	const GmMat3 IDENTITY3 = IDENTITY3_INIT;
	const uint32_t zeroed[] = {
		0x00, 0x04, 0x08, 0x10, 0x14, 0x18, 0x1c, 0x20, 0x24, 0x28,
		0x2c, 0x54, 0x58, 0x5c, 0x60,
	};
	for (uint32_t i = 0; i < sizeof(zeroed) / sizeof(zeroed[0]); ++i)
		memset(state->bytes + zeroed[i], 0, 4);
	memset(state->bytes + 0x0c, 0, 2);
	memcpy(state->bytes + 0x30, &IDENTITY3, sizeof(IDENTITY3));
}

/* 0x007BD2A0 CSceneVehicleCar::WheelReset. */
TMNF_HD static void wheel_reset(
	CSceneVehicleCar *vehicle, CSceneVehicleCarAuxContext *aux_context,
	uint32_t index)
{
	const GmMat3 IDENTITY3 = IDENTITY3_INIT;
	CSceneVehicleCarWheel *wheel = &vehicle->wheels[index];
	CSceneVehicleCarWheelRealTimeState *rt = &wheel->real_time;
	CSceneVehicleCarWheelAux *aux = &aux_context->wheels[index];
	float rest = vehicle->tuning->suspension_rest_length;
	float minus_rest = -rest;
	float minus_zero = x87_mul(minus_rest, 0.0f);

	rt->field08 = 0.0f;
	rt->field04 = 0.0f;
	rt->damper_absorb = rest;
	/* 0x007C93A0 SSurfaceHandler::Reset: location = source, then the wheel
	 * hangs one rest length below it; 0x007C8AA0 UpdateSurface moves the
	 * collision tree. */
	aux->surface_location = aux->surface_source;
	aux->surface_location.t[0] =
		x87_add(aux->surface_location.t[0], minus_zero);
	aux->surface_location.t[1] =
		x87_add(aux->surface_location.t[1], minus_rest);
	aux->surface_location.t[2] =
		x87_add(aux->surface_location.t[2], minus_zero);
	aux_context->set_surface_location(
		aux_context->runtime, wheel->surface_handler,
		&aux->surface_location);

	rt->field6c = 0.0f;
	rt->rotation_phase = 0.0f;
	rt->has_ground_contact = 0;
	rt->basis1 = IDENTITY3;
	const GmVec3 zero3 = { 0.0f, 0.0f, 0.0f };
	rt->field54 = zero3;
	rt->field90 = zero3;
	memset(rt->reserved60, 0, sizeof(rt->reserved60));
	rt->blend_value = 0.0f;
	rt->blend_target = 0.0f;
	/* 16-bit store: the upper half of the material word is never written. */
	rt->contact_material_id =
		(int32_t)((uint32_t)rt->contact_material_id & 0xffff0000u);
	rt->is_sliding = 0;
	rt->ground_contact_count = 0;
	wheel->field15c = 0;
	wheel->contact_relative_local_distance = zero3;
	if (wheel->history != NULL) {
		wheel_state_reset(&wheel->history->field234);
		wheel_state_reset(&wheel->history->async_state);
		wheel_state_reset(&wheel->history->previous_sync);
		wheel_state_reset(&wheel->history->sync);
	}
}

/* 0x00534CD0 CHmsDyna::CHmsStateDyna::Reset: velocities, accumulators and
 * the trailing words; orientation and position are left to SetLocation. */
TMNF_HD static void dyna_state_reset(CHmsStateDyna *state)
{
	const GmVec3 zero3 = { 0.0f, 0.0f, 0.0f };
	state->linVel = zero3;
	state->angVel = zero3;
	state->force = zero3;
	state->torque = zero3;
	state->linVelAdded = zero3;
	state->tail[0] = 0.0f;
	state->tail[1] = 0.0f;
	state->tail[2] = 0.0f;
	state->tail[3] = 0.0f;
}

/* 0x0053D340 CHmsItem::ResetDynamicState -> 0x005474E0 CHmsCorpus::Reset ->
 * 0x00535CB0 CHmsDyna::Reset. */
TMNF_HD static void dyna_reset(CHmsDyna *dyna)
{
	dyna_state_reset(dyna->liveState);
	dyna_state_reset(dyna->stateB);
	dyna_state_reset(&dyna->tempState);
	dyna->replacementBuf.count = 0;
}

/* 0x005338F0 CHmsDyna::SetLocation: state B first, then the live copy. */
TMNF_HD static void dyna_set_location(CHmsDyna *dyna, const GmIso4 *spawn)
{
	CHmsStateDyna *b = dyna->stateB;
	CHmsStateDyna *live = dyna->liveState;
	GmQuat_SetFromMat3(&b->quat, (const GmMat3 *)spawn);
	memcpy(&b->rot, spawn->m, sizeof(b->rot));
	memcpy(&b->pos, spawn->t, sizeof(b->pos));
	GmMat3_SetMult(&b->invInertiaWorld, &b->rot, &dyna->params->invInertiaBody);
	GmMat3_MultTranspose(&b->invInertiaWorld, &b->rot);
	GmVec4_Set(&live->quat, &b->quat);
	memcpy(&live->rot, spawn->m, sizeof(live->rot));
	memcpy(&live->pos, spawn->t, sizeof(live->pos));
	GmMat3_Set(&live->invInertiaWorld, &b->invInertiaWorld);
}

/* 0x007C0320 CSceneVehicleCar::VehicleReset and its base 0x007CB6B0
 * CSceneVehicle::VehicleReset, restricted to the state the port simulates.
 * Every store below is one of the game's; the offsets are the CSceneVehicleCar
 * words. */
TMNF_HD static void vehicle_reset(TmnfPhysicsCorpus *corpus)
{
	const GmIso4 identity = {
		{ 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f },
		{ 0.0f, 0.0f, 0.0f },
	};
	CSceneVehicleCar *vehicle = corpus->vehicle;
	TMNFVehicleComputeContext *context = corpus->vehicle_compute;
	CSceneVehicleCarAuxContext *aux = context->aux;
	TMNFVehicleContactContext *contact = context->contact;
	CSceneVehicleCarModel6State *model6 = context->model6->state;
	TMNFVehicleComputeState *compute = context->state;
	CHmsDynaParams *params = corpus->dyna->params;
	const GmVec3 zero = { 0.0f, 0.0f, 0.0f };

	/* CSceneVehicle::VehicleReset */
	vehicle->input_gas = 0.0f;                 /* +0x050 */
	vehicle->input_brake = 0.0f;               /* +0x054 */
	vehicle->input_steer = 0.0f;               /* +0x058 */
	compute->air_effect_threshold = 0.0f;      /* +0x05c */
	compute->spring_a.value = 0.0f;            /* +0x228 GmSpring::ClearVals */
	compute->spring_a.target = 0.0f;
	compute->spring_a.velocity = 0.0f;
	compute->spring_c.value = 0.0f;            /* +0x214 */
	compute->spring_c.target = 0.0f;
	compute->spring_c.velocity = 0.0f;
	compute->contact_rise = 0.0f;              /* +0x23c */
	compute->contact_decay = 0.0f;             /* +0x240 */
	dyna_reset(corpus->dyna);

	/* CSceneVehicleCar::VehicleReset */
	aux->steering_value = 0.0f;                /* +0x5e8 */
	aux->turbo_progress = 0.0f;                /* +0x5f0 */
	aux->turbo_type = (TMNFVehicleTurboType)0;  /* +0x600 */
	compute->air_impulse_cooldown_tick = 0;    /* +0x610 */
	aux->air_control_immediate = 0;            /* +0x5d4 */
	compute->state_5d8 = 0;                    /* +0x5d8 */
	model6->contact_block_count = 0;
	contact->side_contact = 0;                 /* +0x5dc */
	model6->side_contact = 0;
	aux->air_control_tick = 0;                 /* +0x614 */
	aux->air_control_speed = zero;             /* +0x618 */
	aux->turbo_factor = 0.0f;                  /* +0x5f4 */
	vehicle->turbo_active = 0;                 /* +0x628 */
	contact->last_side_contact_tick = UINT32_MAX; /* +0x5e0 */
	model6->last_sliding_tick = UINT32_MAX;    /* +0x62c */
	model6->sliding_start_tick = UINT32_MAX;   /* +0x630 */
	aux->air_control_locked = 0;               /* +0x5e4 */
	contact->airborne_friction_gate = 0;
	vehicle->force_wheel_speed = 0;            /* +0x6a0 */
	model6->burnout_start_tick = UINT32_MAX;   /* +0x6f4 */
	model6->burnout_transition_tick = UINT32_MAX; /* +0x6f8 */
	model6->model_iso = identity;              /* +0x6a4 */
	model6->orbit_axis = zero;                 /* +0x6fc */
	vehicle->current_local_speed = zero;       /* +0x70c */
	compute->effect_accumulator = 0.0f;        /* +0x624 */
	vehicle->drive_mode = 0;                   /* +0x69c */
	vehicle->engine_mode = 0;                  /* +0x2e4 */
	vehicle->engine_limit_flag = 0;            /* +0x744 */
	vehicle->block_wheel_speed = 0;            /* +0x73c */
	compute->event_level_a = 0;                /* +0x654 */
	compute->event_level_b = 0;                /* +0x658 */
	compute->event_level_c = 0;                /* +0x1fc */
	compute->event_source_ab = 0;              /* +0x201 */
	compute->event_source_c = 0;               /* +0x200 */
	compute->peak_event_level_b = 0;           /* +0x660 */
	compute->peak_event_level_a = 0;           /* +0x664 */
	compute->peak_event_level_c = 0;           /* +0x668 */
	compute->event_metric_a = 0.0f;            /* +0x670 */
	compute->peak_event_source_ab = 0;         /* +0x66c */
	compute->event_metric_b = 0.0f;            /* +0x674 */
	compute->peak_event_source_c = 0;          /* +0x66d */
	compute->event_metric_c = 0.0f;            /* +0x678 */
	compute->last_force_tick = 0;              /* +0x650 */
	vehicle->total_force_added = zero;         /* +0x818 */
	vehicle->total_impulse_added = zero;       /* +0x824 */
	for (uint32_t i = 0; i < vehicle->wheel_count; ++i)
		wheel_reset(vehicle, aux, i);
	/* 0x007BC9A0 SEngine::Reset */
	vehicle->engine.reverse = 0;
	vehicle->engine.rpm = 0.0f;
	vehicle->engine.gear = 1;
	vehicle->engine.target_rpm = 0.0f;
	vehicle->engine.braking_factor = 0.0f;
	vehicle->engine.shift_timer = 0.0f;
	vehicle->engine.clutch = 1.0f;
	/* The dyna model words the force caller rewrites every tick
	 * (0x007C69E0): grounded force-field scale, no linear drag. */
	params->forceFieldScale = context->tuning->grounded_force_field_scale;
	params->dragLinear = 0.0f;
	contact->body_contact_position_sum = zero;  /* +0x684 */
	contact->body_contact_normal_sum = zero;    /* +0x690 */
	contact->body_contact_count = 0;            /* +0x680 */
	contact->wheel_contact_absorb_count = 0;    /* +0x67c */
	vehicle->flag_60c = 0;                      /* +0x60c */
}

TMNF_HD void TmnfVehicle_Respawn(TmnfPhysicsCorpus *corpus, const GmIso4 *spawn)
{
	if (corpus == NULL || spawn == NULL)
		respawn_fail("respawn argument is null");
	TMNFVehicleComputeContext *context = corpus->vehicle_compute;
	if (corpus->vehicle == NULL || corpus->dyna == NULL ||
		corpus->dyna->params == NULL || corpus->vehicle->tuning == NULL ||
		corpus->vehicle->wheels == NULL || context == NULL ||
		context->aux == NULL || context->aux->wheels == NULL ||
		context->aux->set_surface_location == NULL ||
		context->contact == NULL || context->model6 == NULL ||
		context->model6->state == NULL || context->state == NULL ||
		context->tuning == NULL) {
		respawn_fail("vehicle context graph is incomplete");
	}
	vehicle_reset(corpus);
	dyna_reset(corpus->dyna);
	/* 0x007BC800 VehicleBlockSpeed2Set(0) */
	context->aux->integration_flags &= ~0x20u;
	dyna_set_location(corpus->dyna, spawn);
}
