#ifndef TMNF_WORLD_H
#define TMNF_WORLD_H

#include <stdint.h>

#include "tmnf_hd.h"
#include "physics.h"
#include "track.h"
#include "vec_env.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TmnfWorld TmnfWorld;

enum {
	/* Root children of the vehicle collision tree: four wheels, four body
	 * ellipsoids. Each carries one per-sphere contact buffer. */
	TMNF_WORLD_CONTACT_BUFFER_COUNT = 8,
	/* Allocation boundary of a world (World_Size is a multiple of it). */
	TMNF_WORLD_ALIGNMENT = 64,
	/* Upper bound of World_Size, checked in world.c; the device steps each
	 * world in a local buffer of this size. */
	TMNF_WORLD_MAX_BYTES = 4800,
	TMNF_WORLD_MERGE_CAPACITY = 8,
};

/*
 * Builds one world using caller-owned, same-track collision and vehicle
 * snapshots. The track handle may serve many worlds concurrently and must
 * outlive all of them. Invalid or mismatched snapshots terminate immediately.
 */
/* Whether the build includes the locally supplied physics image. */
int World_HasLocalAssets(void);

TmnfWorld *World_Create(
	const TmnfTrack *track,
	const char *vehicle_snapshot_path);
void World_Destroy(TmnfWorld *world);

/*
 * Everything a world points at outside its own struct. World_LinkPointers
 * rewrites every pointer inside the world from these sources and the world's
 * own address, leaving all decoded data untouched. The host calls it once at
 * creation with its own allocations; a device world is a byte copy of a host
 * world relinked with device-resident sources. Record arrays are the
 * fixed-capacity scratch behind the collision buffers.
 */
typedef struct {
	/* The world's immutable half (World_GetCold of the source, World_ColdSize
	 * bytes), a byte copy of which every world on a device shares. */
	const void *cold;
	const uint8_t *vehicle_blob;
	const TmnfTrack *track;
	const float *curve_bounds;      /* World_GetCurveBounds of the source */
	const uint8_t *fake_contact_mask;
	const float *water_impulse_positions;
	const float *water_impulse_vertical_values;
	const float *water_impulse_horizontal_values;
	/* The waypoint triggers the physics step tests (TmnfPhysicsWorld.route);
	 * NULL for a physics-only world. World_Create links NULL; the race
	 * layer's owner sets it (TmnfVecEnv_Init, replay_tick --route). */
	const TmnfRoute *route;
	CSceneVehicleCarWheelHistory *wheel_history; /* four records, separately owned */
	SHmsPhysicalCollision *collision_records;
	uint32_t collision_capacity;
	SHmsPhysicalCollision *contact_records[TMNF_WORLD_CONTACT_BUFFER_COUNT];
	uint32_t contact_capacities[TMNF_WORLD_CONTACT_BUFFER_COUNT];
	GmVec3 *replacements;
	uint32_t replacement_capacity;
} TmnfWorldLinkSources;

TMNF_HD void World_LinkPointers(
	TmnfWorld *world, const TmnfWorldLinkSources *sources);
/* Only the scratch record arrays (collision, contact and replacement
 * buffers): the rest of the sources must already be linked. A device world
 * assembled from a shared template is repointed at its own scratch this way. */
TMNF_HD void World_LinkScratch(
	TmnfWorld *world, const TmnfWorldLinkSources *sources);
/* The sources a host world is currently linked against. A byte copy of the
 * world relinked with these is indistinguishable from the original. */
void World_GetLinkSources(
	const TmnfWorld *world, TmnfWorldLinkSources *sources);
/* A world is two allocations: the mutable part (World_Size bytes, what
 * TmnfWorld* addresses) and the immutable part it points at (World_ColdSize
 * bytes, page-aligned): decoded tuning, curves, materials, collision shapes,
 * contexts. The physics never writes the cold part after World_Create
 * (tests/world_cold_readonly.c). */
TMNF_HD size_t World_Size(void);
TMNF_HD size_t World_ColdSize(void);
const void *World_GetCold(const TmnfWorld *world);
const uint8_t *World_GetVehicleBlob(const TmnfWorld *world, uint32_t *size);
const TmnfTrack *World_GetTrack(const TmnfWorld *world);
/* The curve key bound table (CFuncKeys_CompileInto of every curve); a device
 * world links a copy of it. */
const float *World_GetCurveBounds(const TmnfWorld *world, uint32_t *count);
const TmnfPhysicsWorld *World_GetPhysicsWorldConst(const TmnfWorld *world);

TMNF_HD TmnfPhysicsWorld *World_GetPhysicsWorld(TmnfWorld *world);
TMNF_HD CSceneVehicleCar *World_GetPlayerVehicle(TmnfWorld *world);
TMNF_HD const CHmsStateDyna *World_GetPlayerState(const TmnfWorld *world);
TMNF_HD const CFastBuffer_SHmsPhysicalCollision *World_GetLastCollisions(
	const TmnfWorld *world);
TMNF_HD const CHmsReplacementBuf *World_GetPlayerReplacements(
	const TmnfWorld *world);

TMNF_HD uint32_t World_GetTimerTick(const TmnfWorld *world);
void World_AdvanceTimer(TmnfWorld *world, uint32_t tick_ms);

/*
 * In-race respawn, 0x0047BF00 CTrackManiaRace::RespawnPlayerVehicle without
 * the seeded yaw jitter (the seed is zero for a player respawn): the car is
 * reset (0x007C0320 CSceneVehicleCar::VehicleReset), its rigid body is
 * zeroed (0x0053D340 CHmsItem::ResetDynamicState) and placed at `spawn`
 * (0x007B2E00 CSceneMobil::SetLocation). The race timer keeps running. See
 * analysis/respawn.md.
 */
TMNF_HD void World_Respawn(TmnfWorld *world, const GmIso4 *spawn);

TMNF_HD void World_GetPlayerObservation(
	const TmnfWorld *world, TmnfObservation *observation);

/*
 * Reconstructs the game-layout car and wheel blocks from the immutable input
 * image plus all mutable fields represented by the native contexts.
 */
TMNF_HD void World_WritePlayerGameState(
	const TmnfWorld *world,
	uint8_t car[TMNF_CSCENE_VEHICLE_CAR_GAME_SIZE],
	uint8_t wheels[
		TMNF_STADIUM_WHEEL_COUNT *
		TMNF_CSCENE_VEHICLE_CAR_WHEEL_GAME_SIZE]);

#ifdef __cplusplus
}
#endif

#endif
