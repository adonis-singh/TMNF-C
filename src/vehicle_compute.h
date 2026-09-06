#ifndef TMNF_VEHICLE_COMPUTE_H
#define TMNF_VEHICLE_COMPUTE_H

#include "tmnf_hd.h"
#include <stdint.h>

#include "hms_item.h"
#include "vehicle_aux.h"
#include "vehicle_contact.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TMNFVehicleModel6Context TMNFVehicleModel6Context;
typedef struct TMNFVehicleOldModelsContext TMNFVehicleOldModelsContext;
typedef struct TMNFVehicleComputeContext TMNFVehicleComputeContext;

/*
 * Native view of the selected tuning fields read directly by
 * CSceneVehicleCar::ComputeForces. The comments retain their game offsets.
 */
typedef struct {
	float event_c_level1_max;             /* tuning +0x028 */
	float grounded_drag_term;             /* tuning +0x058 */
	float active_contact_stop_threshold;  /* tuning +0x0a4 */
	float normal_turbo_factor;            /* tuning +0x0f0 */
	float roulette_turbo_factor;          /* tuning +0x0f4 */
	uint32_t normal_turbo_duration;        /* tuning +0x0f8 */
	uint32_t roulette_turbo_duration;      /* tuning +0x0fc */
	float air_impulse_scale;               /* tuning +0x104 */
	float special_force_field_scale;       /* tuning +0x108 */
	float airborne_linear_drag;            /* tuning +0x154 */
	float grounded_force_field_scale;      /* tuning +0x160 */
	float airborne_force_field_scale;      /* tuning +0x164 */
	float normalized_force_divisor;        /* tuning +0x228 */
	const CFuncKeysReal *effect_curve;      /* tuning +0x380 */
	float effect_curve_bias;               /* tuning +0x384 */
	float event_ab_level2_min;              /* tuning +0x398 */
	float event_ab_trigger;                 /* tuning +0x39c */
	float event_c_trigger;                  /* tuning +0x3a0 */

	/* These two curves are read through the car's primary tuning at +0x60. */
	const CFuncKeysReal *contact_decay_curve; /* primary tuning +0x044 */
	const CFuncKeysReal *contact_rise_curve;  /* primary tuning +0x048 */
} TMNFVehicleComputeTuning;

/*
 * Car fields used only by the top force caller. Fields already represented by
 * CSceneVehicleCar, CSceneVehicleCarAuxContext, or TMNFVehicleContactContext
 * remain in those components instead of being duplicated here.
 */
typedef struct {
	float simulation_gate;                 /* car +0x1e8 */
	uint32_t event_level_c;                /* car +0x1fc */
	uint8_t event_source_c;                /* car +0x200 */
	uint8_t event_source_ab;               /* car +0x201 */

	GmSpringFloat spring_c;                 /* car +0x214 */
	GmSpringFloat spring_a;                 /* car +0x228 */
	float contact_rise;                    /* car +0x23c */
	float contact_decay;                   /* car +0x240 */
	float history_force_limit;             /* car +0x244 */
	float history_force_scale;             /* car +0x24c */
	float spring_value_limit;               /* car +0x250 */
	float local_speed_limit;                /* car +0x2e0 */

	float air_effect_threshold;             /* car +0x05c */
	float brake_input_scale;                /* car +0x5a8 */
	float grounded_drag_scale;              /* car +0x5ac */
	float computed_brake_force;             /* car +0x5b0 */
	int32_t state_5d8;                      /* car +0x5d8 */
	uint32_t air_impulse_cooldown_tick;     /* car +0x610 */
	float effect_accumulator;               /* car +0x624 */

	uint32_t last_force_tick;               /* car +0x650 */
	uint32_t event_level_a;                 /* car +0x654 */
	uint32_t event_level_b;                 /* car +0x658 */
	uint32_t peak_event_level_b;            /* car +0x660 */
	uint32_t peak_event_level_a;            /* car +0x664 */
	uint32_t peak_event_level_c;            /* car +0x668 */
	uint8_t peak_event_source_ab;           /* car +0x66c */
	uint8_t peak_event_source_c;            /* car +0x66d */
	float event_metric_a;                   /* car +0x670 */
	float event_metric_b;                   /* car +0x674 */
	float event_metric_c;                   /* car +0x678 */

	GmVec3 normalized_force;                /* car +0x6d4 */
	int32_t air_effect_mode;                /* car +0x74c */
} TMNFVehicleComputeState;

typedef void (*TMNFVehiclePostForceFn)(
	void *runtime, CSceneVehicleCar *vehicle);
typedef uint32_t (*TMNFVehicleContactTokenFn)(
	void *runtime, const CHmsResponseBody *body);

/*
 * Composition root for the 0x007C69E0 caller. `fake_contacts_active` and
 * `water_forces_active` are adapter-supplied branch facts. Their unported
 * branches abort instead of silently dropping effects.
 */
struct TMNFVehicleComputeContext {
	CSceneVehicleCar *vehicle;
	CHmsItem *item;
	CSceneVehicleCarAuxContext *aux;
	TMNFVehicleContactContext *contact;
	const TMNFVehicleComputeTuning *tuning;
	TMNFVehicleComputeState *state;
	TMNFVehicleModel6Context *model6;       /* tuning +0x354 == 5 */
	TMNFVehicleOldModelsContext *oldmodels; /* tuning +0x354 in 0..4 */

	int32_t fake_contacts_active;
	int32_t water_forces_active;

	void *runtime;
	TMNFVehiclePostForceFn post_force;
	TMNFVehicleContactTokenFn contact_token;
};

/*
 * External Model 6 seam. Its argument order matches the original
 * 0x007C3E80 call after replacing the game `this` pointer with the composed
 * native context.
 */
TMNF_HD void VehicleModel6_ComputeForces(
	TMNFVehicleComputeContext *context, float dt,
	const GmVec3 *existing_force, float slope_adherence,
	float slope_secondary, const GmVec3 *linear_speed,
	const GmVec3 *angular_speed, float steering_angle,
	int has_ground_material,
	const CSceneVehicleMaterialBlendableVals *ground_material,
	int *air_control_reset, float *effect_curve_position);

/*
 * Models 3, 4 and 5 (vehicle_model3.c, vehicle_model4.c, vehicle_model5.c),
 * selected on tuning +0x354 exactly as 0x007C69E0 does: 3 -> Model 4,
 * 4 -> Model 5, anything else but 5 -> Model 3. Ported, no oracle trace yet.
 */
TMNF_HD void VehicleOldModels_ComputeForces(
	TMNFVehicleComputeContext *context, float dt,
	const GmVec3 *existing_force, float slope_adherence,
	float slope_secondary, const GmVec3 *linear_speed,
	const GmVec3 *angular_speed, float steering_angle,
	int has_ground_material,
	const CSceneVehicleMaterialBlendableVals *ground_material,
	int *air_control_reset, float *effect_curve_position);

/* 0x007C69E0 */
TMNF_HD void CSceneVehicleCar_ComputeForces(
	TMNFVehicleComputeContext *context, float dt);

/* 0x007C7D40 */
TMNF_HD void CCallbackSceneVehicleCarComputeForces_ComputeForces(
	TMNFVehicleComputeContext *context, CHmsItem *item, float dt);

/*
 * Closure-style adapter required by PhysicsStep2:
 *   callback(user, corpus->vehicle, dt)
 */
TMNF_HD void TMNFVehicleComputeForces_PhysicsStep2Adapter(
	void *user, CSceneVehicleCar *vehicle, float dt);

#ifdef __cplusplus
}
#endif

#endif
