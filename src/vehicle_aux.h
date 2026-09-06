#ifndef TMNF_VEHICLE_AUX_H
#define TMNF_VEHICLE_AUX_H

#include "tmnf_hd.h"
#include <stddef.h>
#include <stdint.h>

#include "collision.h"
#include "vehicle.h"
#include "vehicle_curve.h"

#ifdef __cplusplus
extern "C" {
#endif

struct TMNFVehicleContactContext;
struct TmnfTrackWater;

typedef enum {
	TMNF_TURBO_NONE = 0,
	TMNF_TURBO_NORMAL = 1,
	TMNF_TURBO_ROULETTE = 2,
} TMNFVehicleTurboType;

/*
 * Native views for runtime-owned state that is absent from vehicle.h.
 * Curves contain parallel position/value arrays.
 */
typedef struct {
	uint32_t count;
	const float *positions;
	const float *lower_bounds;   /* CFuncKeys_Compile of positions */
	const float *upper_bounds;
	const float *values;
	int32_t interpolation;
} TMNFVehicleAuxCurve;

typedef struct {
	float steering_speed_base;             /* game tuning +0x06c */
	float steering_speed_scale;            /* game tuning +0x070 */
	float steering_slew_rate;              /* game tuning +0x094 */
	float air_torque_linear;                /* game tuning +0x158 */
	float air_torque_quadratic;             /* game tuning +0x15c */
	float suspension_follow_rate;           /* game tuning +0x194 */
	uint32_t air_control_window_ticks;      /* game tuning +0x364 */
	float air_reversal_threshold;           /* game tuning +0x368 */
	const TMNFVehicleAuxCurve *air_vertical_curve; /* game tuning +0x36c */
	const TMNFVehicleAuxCurve *steering_angle_curve; /* game tuning +0x378 */
	/* 0x007C2910 ApplyWaterForces tuning. */
	float water_buoyancy;                    /* game tuning +0x204 */
	float water_entry_speed_threshold;      /* game tuning +0x208 */
	float water_entry_speed_minimum;        /* game tuning +0x20c */
	const CFuncKeysReal *water_impulse_vertical_curve;   /* +0x210 */
	const CFuncKeysReal *water_impulse_horizontal_curve; /* +0x214 */
	const CFuncKeysReal *water_friction_curve;           /* +0x218 */
	float water_angular_drag_linear;        /* game tuning +0x21c */
	float water_angular_drag_quadratic;     /* game tuning +0x220 */
	const struct TmnfTrackWater *water_map; /* CHmsZone+0x168 water map */
} CSceneVehicleCarTuningAux;

typedef struct {
	GmIso4 surface_source;                  /* game wheel +0x010 */
	GmIso4 surface_location;                /* game wheel +0x040 */
} CSceneVehicleCarWheelAux;

typedef void (*TMNFVehicleTurboSoundPlayFn)(void *runtime);
typedef void (*TMNFVehicleSurfaceSetLocationFn)(
	void *runtime, void *surface_tree, const GmIso4 *location);
typedef void (*TMNFVehicleFinishIntegrationFn)(
	void *runtime, CSceneVehicleCar *vehicle);

typedef struct {
	CSceneVehicleCar *vehicle;
	const CSceneVehicleCarTuningAux *tuning;
	CSceneVehicleCarWheelAux *wheels;
	uint32_t wheel_count;

	uint32_t integration_flags;             /* game car +0x2f4 */
	uint32_t turbo_epoch_tick;              /* game car +0x5d0 */
	int32_t air_control_immediate;           /* game car +0x5d4 */
	int32_t air_control_locked;              /* game car +0x5e4 */
	float steering_value;                   /* game car +0x5e8 */
	float turbo_progress;                   /* game car +0x5f0 */
	float turbo_factor;                     /* game car +0x5f4 */
	uint32_t turbo_start_tick;               /* game car +0x5f8 */
	uint32_t turbo_end_tick;                 /* game car +0x5fc */
	TMNFVehicleTurboType turbo_type;         /* game car +0x600 */
	uint32_t roulette_token;                 /* game car +0x604 */
	float roulette_value;                   /* game car +0x608 */
	uint32_t air_control_tick;               /* game car +0x614 */
	GmVec3 air_control_speed;                /* game car +0x618 */

	uint32_t roulette_modulus;               /* runtime DAT_00d06a74 */
	/* Game car +0x1dc..+0x1f4: the body box read by ApplyWaterForces. The
	 * same words are CSceneVehicleCarModel6State pivot_position/pivot_axis;
	 * both views are decoded from the snapshot and never written. */
	GmBoxAligned body_box;
	int32_t turbo_sound_attached;            /* game car +0x26c != NULL */
	void *runtime;
	TMNFVehicleTurboSoundPlayFn play_turbo_sound;
	TMNFVehicleSurfaceSetLocationFn set_surface_location;
	TMNFVehicleFinishIntegrationFn finish_integration;
} CSceneVehicleCarAuxContext;

/*
 * Exact 32-bit game layouts for the fields used by this port. Pointer-bearing
 * members are uint32_t so host pointer size cannot alter their offsets.
 */
typedef struct {
	uint32_t active;                         /* 0x000 */
	uint32_t steerable;                      /* 0x004 */
	float radius;                            /* 0x008 */
	uint32_t surface_tree;                   /* 0x00c */
	GmIso4 surface_source;                   /* 0x010 */
	GmIso4 surface_location;                 /* 0x040 */
	uint8_t reserved070[0x38];
	GmVec3 offset_from_vehicle;              /* 0x0a8 */
	CSceneVehicleCarWheelRealTimeState real_time; /* 0x0b4 */
	int32_t field15c;                        /* 0x15c */
	GmVec3 contact_relative_local_distance; /* 0x160 */
	CSceneVehicleCarWheelState previous_sync; /* 0x16c */
	CSceneVehicleCarWheelState sync;         /* 0x1d0 */
	CSceneVehicleCarWheelState field234;     /* 0x234 */
	CSceneVehicleCarWheelState async_state;  /* 0x298 */
} TMNFVehicleAuxWheelGame32;

typedef struct {
	uint8_t reserved000[0x06c];
	float steering_speed_base;              /* 0x06c */
	float steering_speed_scale;             /* 0x070 */
	uint8_t reserved074[0x020];
	float steering_slew_rate;               /* 0x094 */
	uint8_t reserved098[0x07c];
	float suspension_stiffness;             /* 0x114 */
	float suspension_damping;               /* 0x118 */
	uint8_t reserved11c[0x008];
	float suspension_rest_length;           /* 0x124 */
	float suspension_scale;                 /* 0x128 */
	uint8_t reserved12c[0x02c];
	float air_torque_linear;                 /* 0x158 */
	float air_torque_quadratic;              /* 0x15c */
	uint8_t reserved160[0x034];
	float suspension_follow_rate;           /* 0x194 */
	uint8_t reserved198[0x1b8];
	int32_t suspension_model;                /* 0x350 */
	int32_t engine_model;                    /* 0x354 */
	uint8_t reserved358[0x00c];
	uint32_t air_control_window_ticks;       /* 0x364 */
	float air_reversal_threshold;            /* 0x368 */
	uint32_t air_vertical_curve;             /* 0x36c */
	uint8_t reserved370[0x008];
	uint32_t steering_angle_curve;           /* 0x378 */
	uint8_t reserved37c[0x030];
} TMNFVehicleAuxTuningGame32;

typedef struct {
	uint8_t reserved000[0x028];
	uint32_t hms_item;                       /* 0x028 */
	uint8_t reserved02c[0x024];
	float input_gas;                         /* 0x050 */
	float input_brake;                       /* 0x054 */
	float input_steer;                       /* 0x058 */
	uint32_t reserved05c;
	uint32_t tuning;                         /* 0x060 */
	uint32_t tuning_container;               /* 0x064 */
	uint8_t reserved068[0x204];
	uint32_t turbo_sound;                    /* 0x26c */
	uint8_t reserved270[0x074];
	int32_t engine_mode;                     /* 0x2e4 */
	uint32_t wheel_data;                     /* 0x2e8 */
	uint32_t wheel_count;                    /* 0x2ec */
	uint32_t wheel_capacity;                 /* 0x2f0 */
	uint32_t integration_flags;              /* 0x2f4 */
	uint8_t reserved2f8[0x2a4];
	CSceneVehicleCarEngine engine;           /* 0x59c */
	uint8_t reserved5cc[0x004];
	uint32_t turbo_epoch_tick;               /* 0x5d0 */
	int32_t air_control_immediate;            /* 0x5d4 */
	uint8_t reserved5d8[0x00c];
	int32_t air_control_locked;               /* 0x5e4 */
	float steering_value;                    /* 0x5e8 */
	uint8_t reserved5ec[0x004];
	float turbo_progress;                    /* 0x5f0 */
	float turbo_factor;                      /* 0x5f4 */
	uint32_t turbo_start_tick;               /* 0x5f8 */
	uint32_t turbo_end_tick;                 /* 0x5fc */
	int32_t turbo_type;                      /* 0x600 */
	uint32_t roulette_token;                 /* 0x604 */
	float roulette_value;                    /* 0x608 */
	int32_t flag60c;                         /* 0x60c */
	uint8_t reserved610[0x004];
	uint32_t air_control_tick;               /* 0x614 */
	GmVec3 air_control_speed;                /* 0x618 */
	uint8_t reserved624[0x004];
	int32_t turbo_active;                    /* 0x628 */
	uint8_t reserved62c[0x070];
	int32_t drive_mode;                      /* 0x69c */
	int32_t force_wheel_speed;               /* 0x6a0 */
	uint8_t reserved6a4[0x068];
	GmVec3 current_local_speed;              /* 0x70c */
	uint8_t reserved718[0x024];
	int32_t block_wheel_speed;               /* 0x73c */
	uint8_t reserved740[0x004];
	int32_t engine_limit_flag;               /* 0x744 */
	int32_t gear_downshift_flag;             /* 0x748 */
	uint8_t reserved74c[0x0cc];
	GmVec3 total_force_added;                /* 0x818 */
	GmVec3 total_impulse_added;              /* 0x824 */
	uint8_t reserved830[0x048];
} TMNFVehicleAuxCarGame32;

#define TMNF_AUX_OFFSET(type, member, offset) \
	_Static_assert(offsetof(type, member) == (offset), #type "." #member)

_Static_assert(sizeof(TMNFVehicleAuxWheelGame32) == 0x2fc,
	"aux wheel game size");
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, active, 0x000);
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, steerable, 0x004);
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, radius, 0x008);
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, surface_tree, 0x00c);
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, surface_source, 0x010);
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, surface_location, 0x040);
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, offset_from_vehicle, 0x0a8);
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, real_time, 0x0b4);
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, field15c, 0x15c);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxWheelGame32, contact_relative_local_distance, 0x160);
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, previous_sync, 0x16c);
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, sync, 0x1d0);
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, field234, 0x234);
TMNF_AUX_OFFSET(TMNFVehicleAuxWheelGame32, async_state, 0x298);

_Static_assert(sizeof(TMNFVehicleAuxTuningGame32) == 0x3ac,
	"aux tuning game size");
TMNF_AUX_OFFSET(
	TMNFVehicleAuxTuningGame32, steering_speed_base, 0x06c);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxTuningGame32, steering_speed_scale, 0x070);
TMNF_AUX_OFFSET(TMNFVehicleAuxTuningGame32, steering_slew_rate, 0x094);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxTuningGame32, suspension_stiffness, 0x114);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxTuningGame32, suspension_damping, 0x118);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxTuningGame32, suspension_rest_length, 0x124);
TMNF_AUX_OFFSET(TMNFVehicleAuxTuningGame32, suspension_scale, 0x128);
TMNF_AUX_OFFSET(TMNFVehicleAuxTuningGame32, air_torque_linear, 0x158);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxTuningGame32, air_torque_quadratic, 0x15c);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxTuningGame32, suspension_follow_rate, 0x194);
TMNF_AUX_OFFSET(TMNFVehicleAuxTuningGame32, suspension_model, 0x350);
TMNF_AUX_OFFSET(TMNFVehicleAuxTuningGame32, engine_model, 0x354);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxTuningGame32, air_control_window_ticks, 0x364);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxTuningGame32, air_reversal_threshold, 0x368);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxTuningGame32, air_vertical_curve, 0x36c);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxTuningGame32, steering_angle_curve, 0x378);

_Static_assert(sizeof(TMNFVehicleAuxCarGame32) == 0x878,
	"aux car game size");
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, hms_item, 0x028);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, input_gas, 0x050);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, input_brake, 0x054);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, input_steer, 0x058);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, tuning, 0x060);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, tuning_container, 0x064);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, turbo_sound, 0x26c);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, engine_mode, 0x2e4);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, wheel_data, 0x2e8);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, wheel_count, 0x2ec);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, wheel_capacity, 0x2f0);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, integration_flags, 0x2f4);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, engine, 0x59c);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, turbo_epoch_tick, 0x5d0);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxCarGame32, air_control_immediate, 0x5d4);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, air_control_locked, 0x5e4);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, steering_value, 0x5e8);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, turbo_progress, 0x5f0);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, turbo_factor, 0x5f4);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, turbo_start_tick, 0x5f8);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, turbo_end_tick, 0x5fc);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, turbo_type, 0x600);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, roulette_token, 0x604);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, roulette_value, 0x608);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, flag60c, 0x60c);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, air_control_tick, 0x614);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, air_control_speed, 0x618);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, turbo_active, 0x628);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, drive_mode, 0x69c);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, force_wheel_speed, 0x6a0);
TMNF_AUX_OFFSET(
	TMNFVehicleAuxCarGame32, current_local_speed, 0x70c);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, block_wheel_speed, 0x73c);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, engine_limit_flag, 0x744);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, gear_downshift_flag, 0x748);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, total_force_added, 0x818);
TMNF_AUX_OFFSET(TMNFVehicleAuxCarGame32, total_impulse_added, 0x824);

#undef TMNF_AUX_OFFSET

/* 0x007BC820 */
TMNF_HD float CSceneVehicleCar_GetRouletteValue01(
	uint32_t value, uint32_t divisor);
/* 0x007BC890 */
TMNF_HD float CSceneVehicleCar_GetRouletteBoostFactorFromValue01(float value);
/* 0x007BC8B0 */
TMNF_HD void CSceneVehicleCar_UpdateTurbo(
	CSceneVehicleCarAuxContext *self, uint32_t tick);
/* 0x007BCF90 */
TMNF_HD void CSceneVehicleCar_EnableTurbo(
	CSceneVehicleCarAuxContext *self, uint32_t tick, uint32_t duration,
	float factor, TMNFVehicleTurboType type, uint32_t roulette_token);
/* 0x007BD3F0 */
TMNF_HD void CSceneVehicleCar_WheelIntegrate(
	CSceneVehicleCarAuxContext *self, uint32_t wheel_index, float dt);
/* 0x007BF1D0 */
TMNF_HD void CSceneVehicleCar_ComputeAirControl(
	CSceneVehicleCarAuxContext *self, const GmVec3 *angular_speed,
	uint32_t tick, int suppress_torque, int reset);
/* 0x007C2910 */
TMNF_HD int CSceneVehicleCar_ApplyWaterForces(
	CSceneVehicleCarAuxContext *self, const GmVec3 *existing_force);
/* 0x007C3900 */
TMNF_HD void CSceneVehicleCar_IntegrateVehicle(
	CSceneVehicleCarAuxContext *self, float dt);
/* 0x007C3C00 */
TMNF_HD void CSceneVehicleCar_CreateFakeContacts(
	CSceneVehicleCarAuxContext *self,
	struct TMNFVehicleContactContext *contact);

#ifdef __cplusplus
}
#endif

#endif
