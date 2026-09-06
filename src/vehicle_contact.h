#ifndef TMNF_VEHICLE_CONTACT_H
#define TMNF_VEHICLE_CONTACT_H

#include "tmnf_hd.h"
#include <stddef.h>
#include <stdint.h>

#include "collision_response.h"
#include "fastbuffer.h"
#include "vehicle.h"
#include "vehicle_curve.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	float acceleration;
	float braking;
	float steering;
	float lateral_grip;
} CSceneVehicleMaterialBlendableVals;

_Static_assert(sizeof(CSceneVehicleMaterialBlendableVals) == 0x10,
	"material blendable values size");
_Static_assert(offsetof(CSceneVehicleMaterialBlendableVals, lateral_grip) == 0x0c,
	"material lateral grip");

typedef struct {
	float values[4];
	/* 0x007C3C00 CreateFakeContacts descriptor: material +0x24 mask image
	 * (null: no bumps), +0x28/+0x2c world periods, +0x30 impulse scale,
	 * +0x34 impulse limit. */
	const uint8_t *fake_contact_mask;
	float fake_contact_period_x;
	float fake_contact_period_z;
	float fake_contact_impulse_scale;
	float fake_contact_impulse_limit;
} TMNFVehicleGroundMaterial;

typedef struct {
	CSceneVehicleCarWheel *wheel;
	GmVec3 impulse_point;
	CHmsResponseBody *contact_body;
} TMNFVehicleContactWheelState;

typedef struct {
	CSceneVehicleCarTuningCurveSet *curves;
	float friction_force;
	float extra_friction_force;
	float slope_adherence_min;
	float slope_adherence_max;
	float slope_secondary_min;
	float slope_secondary_max;
	float angular_y_scale;
	float angular_xz_scale;
	float damper_max;
	float max_angular_speed;
	float max_linear_speed_delta;
	float body_tangent_ratio;
	float body_tangent_ratio_material4;
	float restitution_air_material4;
	float restitution_air;
	float restitution_ground;
	float restitution_ground_material4;
	float lateral_linear;
	float lateral_quadratic;
	float lateral_ground_scale;
	uint32_t lateral_contact_duration_ticks;
	int32_t wheel_contact_model;
	int32_t friction_model;
} TMNFVehicleContactTuning;

typedef struct {
	uint32_t tick_time;
} TMNFVehicleContactTimer;

typedef const GmIso4 *(*TMNFVehicleResolveBodyIsoFn)(
	void *user, const CHmsResponseBody *body);

typedef struct TMNFVehicleContactContext {
	CSceneVehicleCar *vehicle;
	TMNFVehicleContactTuning *tuning;
	TMNFVehicleContactWheelState *wheels;
	uint32_t wheel_count;
	const uint32_t *ground_material_indices;
	uint32_t ground_material_index_count;
	const TMNFVehicleGroundMaterial *const *ground_materials;
	uint32_t ground_material_count;
	TMNFVehicleContactTimer *timer;
	const GmMat3 *vehicle_contact_rotation;
	TMNFVehicleResolveBodyIsoFn resolve_body_iso;
	void *resolve_body_iso_user;
	const uint32_t *wheel_tree_refs;
	const uint32_t *body_tree_refs;
	uint32_t body_tree_count;
	int32_t *air_control_immediate;
	int32_t *contact_block_count;
	uint8_t *event_source_c;
	uint8_t *event_source_ab;
	float *event_metric_a;
	float *event_metric_b;
	float *event_metric_c;
	uint32_t wheel_contact_absorb_count;
	uint32_t body_contact_count;
	GmVec3 body_contact_position_sum;
	GmVec3 body_contact_normal_sum;
	int32_t friction_input_selector;
	int32_t side_contact;
	uint32_t last_side_contact_tick;
	int32_t airborne_friction_gate;
} TMNFVehicleContactContext;

/*
 * Exact 32-bit game layouts for the portions consumed by this subsystem.
 * Pointer fields remain uint32_t so host pointer size cannot alter offsets.
 */
typedef struct {
	uint8_t reserved000[0x028];
	uint32_t hms_item;
	uint8_t reserved02c[0x024];
	float input_gas;
	float input_brake;
	float input_steer;
	uint32_t reserved05c;
	uint32_t tuning_ref;
	uint32_t tuning_selector;
	uint32_t material_manager;
	CFastBufferLayout32 ground_material_ids;
	uint8_t reserved078[0x270];
	CFastBufferLayout32 wheels;
	uint8_t reserved2f4[0x2d0];
	int32_t friction_input_selector;
	uint8_t reserved5c8[0x014];
	int32_t side_contact;
	uint32_t last_side_contact_tick;
	int32_t airborne_friction_gate;
	uint8_t reserved5e8[0x024];
	int32_t flag_60c;
	uint8_t reserved610[0x214];
	GmVec3 total_impulse_added;
	uint8_t reserved830[0x048];
} TMNFVehicleContactCarLayout32;

typedef struct {
	uint8_t reserved000[0x058];
	float friction_force;
	float extra_friction_force;
	uint8_t reserved060[0x008];
	uint32_t lateral_contact_slowdown_curve;
	uint8_t reserved06c[0x068];
	float slope_adherence_min;
	float slope_adherence_max;
	float slope_secondary_min;
	float slope_secondary_max;
	uint8_t reserved0e4[0x004];
	float angular_y_scale;
	float angular_xz_scale;
	uint8_t reserved0f0[0x02c];
	float damper_max;
	uint8_t reserved120[0x02c];
	float max_angular_speed;
	float max_linear_speed_delta;
	uint8_t reserved154[0x01c];
	float body_tangent_ratio;
	float body_tangent_ratio_material4;
	float restitution_air_material4;
	float restitution_air;
	uint8_t reserved180[0x004];
	float restitution_ground;
	uint8_t reserved188[0x004];
	float restitution_ground_material4;
	uint8_t reserved190[0x018];
	float lateral_linear;
	float lateral_quadratic;
	uint8_t reserved1b0[0x00c];
	uint32_t m4_max_friction_curve;
	float lateral_ground_scale;
	uint8_t reserved1c4[0x024];
	uint32_t lateral_contact_duration_ticks;
	uint8_t reserved1ec[0x164];
	int32_t wheel_contact_model;
	int32_t friction_model;
	uint8_t reserved358[0x054];
} TMNFVehicleContactTuningLayout32;

typedef struct {
	uint8_t reserved000[0x064];
	GmVec3 impulse_point;
	uint8_t reserved070[0x044];
	CSceneVehicleCarWheelRealTimeState real_time;
	int32_t contact_relative_valid;
	GmVec3 contact_relative_local_distance;
	uint8_t reserved16c[0x190];
} TMNFVehicleContactWheelLayout32;

typedef struct {
	uint8_t reserved000[0x014];
	CSceneVehicleMaterialBlendableVals blend;
} TMNFVehicleGroundMaterialLayout32;

typedef struct {
	uint8_t reserved000[0x018];
	float mass;
	GmMat3 inverse_inertia;
	uint8_t reserved040[0x010];
	GmVec3 center_of_mass;
} TMNFVehicleBodyParamsLayout32;

typedef struct {
	uint8_t reserved000[0x01c];
	uint32_t tick_time;
} TMNFVehicleContactTimerLayout32;

_Static_assert(sizeof(TMNFVehicleContactCarLayout32) == 0x878,
	"contact car game size");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32, hms_item) == 0x028,
	"contact car hms item");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32, input_gas) == 0x050,
	"contact car gas");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32, input_brake) == 0x054,
	"contact car brake");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32, tuning_selector) == 0x064,
	"contact car tuning selector");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32, material_manager) == 0x068,
	"contact car material manager");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32, ground_material_ids) == 0x06c,
	"contact car material ids");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32, wheels) == 0x2e8,
	"contact car wheels");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32,
	friction_input_selector) == 0x5c4, "contact car friction input");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32, side_contact) == 0x5dc,
	"contact car side contact");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32,
	last_side_contact_tick) == 0x5e0, "contact car contact tick");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32,
	airborne_friction_gate) == 0x5e4, "contact car friction gate");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32, flag_60c) == 0x60c,
	"contact car flag 60c");
_Static_assert(offsetof(TMNFVehicleContactCarLayout32,
	total_impulse_added) == 0x824, "contact car impulse sum");

_Static_assert(sizeof(TMNFVehicleContactTuningLayout32) == 0x3ac,
	"contact tuning game size");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	friction_force) == 0x058, "contact tuning friction");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	extra_friction_force) == 0x05c, "contact tuning extra friction");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	lateral_contact_slowdown_curve) == 0x068, "contact tuning slowdown curve");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	slope_adherence_min) == 0x0d4, "contact tuning slope min");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	slope_secondary_max) == 0x0e0, "contact tuning secondary slope max");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	angular_y_scale) == 0x0e8, "contact tuning angular y scale");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	angular_xz_scale) == 0x0ec, "contact tuning angular xz scale");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32, damper_max) == 0x11c,
	"contact tuning damper max");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	max_angular_speed) == 0x14c, "contact tuning max angular speed");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	max_linear_speed_delta) == 0x150, "contact tuning max linear delta");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	body_tangent_ratio) == 0x170, "contact tuning body tangent ratio");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	body_tangent_ratio_material4) == 0x174,
	"contact tuning material4 body tangent ratio");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	restitution_air_material4) == 0x178, "contact tuning air material4");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	restitution_ground) == 0x184, "contact tuning ground restitution");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	restitution_ground_material4) == 0x18c,
	"contact tuning ground material4");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	lateral_linear) == 0x1a8, "contact tuning lateral linear");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	lateral_quadratic) == 0x1ac, "contact tuning lateral quadratic");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	m4_max_friction_curve) == 0x1bc, "contact tuning model4 curve");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	lateral_ground_scale) == 0x1c0, "contact tuning ground scale");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	lateral_contact_duration_ticks) == 0x1e8,
	"contact tuning contact duration");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	wheel_contact_model) == 0x350, "contact tuning wheel model");
_Static_assert(offsetof(TMNFVehicleContactTuningLayout32,
	friction_model) == 0x354, "contact tuning friction model");

_Static_assert(sizeof(TMNFVehicleContactWheelLayout32) == 0x2fc,
	"contact wheel game size");
_Static_assert(offsetof(TMNFVehicleContactWheelLayout32,
	impulse_point) == 0x064, "contact wheel impulse point");
_Static_assert(offsetof(TMNFVehicleContactWheelLayout32,
	real_time) == 0x0b4, "contact wheel real-time state");
_Static_assert(offsetof(TMNFVehicleContactWheelLayout32,
	contact_relative_valid) == 0x15c, "contact wheel relative flag");
_Static_assert(offsetof(TMNFVehicleContactWheelLayout32,
	contact_relative_local_distance) == 0x160,
	"contact wheel relative distance");

_Static_assert(sizeof(TMNFVehicleGroundMaterialLayout32) == 0x24,
	"ground material accessed size");
_Static_assert(offsetof(TMNFVehicleGroundMaterialLayout32, blend) == 0x14,
	"ground material blend values");
_Static_assert(sizeof(TMNFVehicleBodyParamsLayout32) == 0x5c,
	"vehicle body params accessed size");
_Static_assert(offsetof(TMNFVehicleBodyParamsLayout32, mass) == 0x18,
	"vehicle body mass");
_Static_assert(offsetof(TMNFVehicleBodyParamsLayout32,
	inverse_inertia) == 0x1c, "vehicle body inverse inertia");
_Static_assert(offsetof(TMNFVehicleBodyParamsLayout32,
	center_of_mass) == 0x50, "vehicle body center");
_Static_assert(sizeof(TMNFVehicleContactTimerLayout32) == 0x20,
	"contact timer accessed size");
_Static_assert(offsetof(TMNFVehicleContactTimerLayout32, tick_time) == 0x1c,
	"contact timer tick");

/* 0x0093A4A0 */
TMNF_HD const uint32_t *CMwTimerAdapter_GetTickTime(
	const TMNFVehicleContactTimer *self);

/* 0x007BE390 */
TMNF_HD void CSceneVehicleCar_AddVehicleImpulse(
	TMNFVehicleContactContext *context, const GmVec3 *impulse,
	const GmVec3 *point);

/* 0x007C11D0 */
TMNF_HD void CSceneVehicleCar_WheelAbsorbContact(
	TMNFVehicleContactContext *context,
	TMNFVehicleContactWheelState *wheel,
	CHmsPhysicalContact *contact);

/* 0x007C3410 */
TMNF_HD void CSceneVehicleCar_AbsorbContact(
	TMNFVehicleContactContext *context,
	CHmsPhysicalContact *contact);

/* 0x007BEB40 */
TMNF_HD void CSceneVehicleCar_GetSlopeAdherence(
	const TMNFVehicleContactContext *context, const GmVec3 *normal,
	float *adherence, float *secondary);

/* 0x007BED10 */
TMNF_HD void CSceneVehicleCar_ApplyFrictionForces(
	TMNFVehicleContactContext *context, const GmVec3 *velocity);

/* 0x007BF080 */
TMNF_HD void CSceneVehicleCar_GetLateralFriction(
	const TMNFVehicleContactContext *context, const GmVec3 *velocity,
	const GmVec3 *lateral_axis,
	const CSceneVehicleMaterialBlendableVals *material, float load,
	int grounded, float *friction, int *sliding);

/* 0x007C2800 */
TMNF_HD void CSceneVehicleCar_ComputeVehicleGroundMaterialVals(
	const TMNFVehicleContactContext *context,
	CSceneVehicleMaterialBlendableVals *values, int *has_material);

/* 0x007BD1E0 */
TMNF_HD int CSceneVehicleCar_IsGroundContact(
	const TMNFVehicleContactContext *context);

/* 0x007BF5C0 */
TMNF_HD int CSceneVehicleCar_IsAllWheelGroundContactId(
	const TMNFVehicleContactContext *context, uint8_t material_id);

/* 0x007BF620 */
TMNF_HD int CSceneVehicleCar_IsGroundContactId(
	const TMNFVehicleContactContext *context, uint8_t material_id,
	GmVec3 *relative_axis, CHmsResponseBody **contact_body);

#ifdef __cplusplus
}
#endif

#endif
