/* Top-level 100 Hz TMNF physics orchestration.
 *
 * This is the clean native representation of CHmsZoneDynamic::PhysicsStep2
 * (0x00549C90) and ComputeCorpusForces (0x005481A0). Game pointer-bearing
 * objects are represented by typed native pointers; pure-data state structs
 * retain their exact game byte layouts.
 */
#ifndef TMNF_PHYSICS_H
#define TMNF_PHYSICS_H

#include "tmnf_hd.h"
#include <stddef.h>
#include <stdint.h>

#include "collision.h"
#include "collision_response.h"
#include "hms_dyna.h"
#include "route.h"
#include "vehicle.h"
#include "vehicle_compute.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	int32_t active;                 /* game CHmsForceFieldUniform+0x54 */
	GmVec3 value;                   /* game +0x5C */
} CHmsForceFieldUniform;

typedef struct {
	uint8_t pad00[0x54];
	int32_t active;
	uint8_t pad58[0x04];
	GmVec3 value;
} CHmsForceFieldUniformLayout32;

_Static_assert(
	sizeof(CHmsForceFieldUniformLayout32) == 0x68,
	"CHmsForceFieldUniform game size");
_Static_assert(
	offsetof(CHmsForceFieldUniformLayout32, active) == 0x54,
	"CHmsForceFieldUniform active");
_Static_assert(
	offsetof(CHmsForceFieldUniformLayout32, value) == 0x5C,
	"CHmsForceFieldUniform value");

typedef struct {
	CHmsCorpus *collision_corpus;
	CHmsDyna *dyna;
	CSceneVehicleCar *vehicle;
	TMNFVehicleComputeContext *vehicle_compute;
	uint32_t scene_flags;           /* game scene object +0x18 */
	uint64_t tick_time;
} TmnfPhysicsCorpus;

struct TmnfPhysicsWorld;

typedef struct TmnfPhysicsWorld {
	float linear_drag_scale;        /* game CHmsZoneDynamic+0x11C */
	float angular_drag_scale;       /* game CHmsZoneDynamic+0x120 */

	CHmsForceFieldUniform *force_fields;
	uint32_t force_field_count;

	TmnfPhysicsCorpus *corpora;
	uint32_t corpus_count;

	CHmsCollisionManager_SZone *collision_zone;
	CHmsCollisionBuffer collision_buffer;
	CHmsResponseZone response_zone;

	/* The route's waypoint triggers stand in for the game's checkpoint and
	 * finish corpora, static-tree members whose contact sink (0x0047CBA0
	 * CTrackManiaRaceTriggerAbsorbContact::AbsorbContact) fires from
	 * ComputeCollisionResponse on the contacts each detection pass found
	 * against the car's predicted iso (analysis/game_rules.md, "Trigger
	 * timing"). Every detection pass of the player corpus ORs
	 * TmnfRace_TriggerContactMask into trigger_contacts; PhysicsStep2 clears
	 * it first. NULL: no triggers (physics-only worlds). A link source
	 * (TmnfWorldLinkSources.route). */
	const TmnfRoute *route;
	uint64_t trigger_contacts;
} TmnfPhysicsWorld;

/* 0x0055F3B0 */
TMNF_HD int CHmsForceFieldUniform_GetValue(
	const CHmsForceFieldUniform *self, const GmVec3 *position, GmVec3 *out);

/* 0x005481A0 */
TMNF_HD void CHmsZoneDynamic_ComputeCorpusForces(
	TmnfPhysicsWorld *world, TmnfPhysicsCorpus *corpus, float dt);

/* 0x00549C90. tick_ms is normally 10 for TMNF's fixed 100 Hz simulation. */
TMNF_HD void CHmsZoneDynamic_PhysicsStep2(TmnfPhysicsWorld *world, uint32_t tick_ms);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_PHYSICS_H */
