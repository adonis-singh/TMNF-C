/* TMNF race progression over an immutable TmnfRoute snapshot.
 *
 * Rules follow the game as measured against six TMX world-record replays:
 *   - a checkpoint counts once per lap, in any order; the finish counts only
 *     when every checkpoint of the lap has been taken;
 *   - a trigger fires when a car collision ellipsoid touches the trigger
 *     volume, computed with the game's own narrowphase, not when the root
 *     bounding boxes overlap, and tested where the game tests it: in every
 *     collision detection pass of the physics step, against the predicted
 *     pre-response transform;
 *   - "off track" means every wheel in ground contact rests on the stadium
 *     ground plane (Grass material), which no record line ever touches;
 *   - progress is credited only inside the route corridor (horizontal offset
 *     within TMNF_RACE_CORRIDOR_WIDTH_FACTOR half-widths of the projected
 *     centerline point and height within the vertical band); outside it
 *     unwrapped_progress freezes until the car comes back.
 */
#ifndef TMNF_RACE_H
#define TMNF_RACE_H

#include "tmnf_hd.h"
#include <stdint.h>

#include "collision.h"
#include "route.h"
#include "surface_material.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
	TMNF_RACE_TICK_MS = 10,
	TMNF_RACE_PROJECTION_WINDOW_SEGMENTS = 32,
	TMNF_RACE_TELEPORT_DISTANCE_METERS = 32,
	TMNF_RACE_MAX_CHECKPOINTS = 63,
	TMNF_RACE_FINISH_CONTACT_BIT = 63,
	TMNF_MATERIAL_GRASS = TMNF_SURFACE_GRASS,
	TMNF_MATERIAL_WET_GRASS = TMNF_SURFACE_WET_GRASS,
};

/*
 * Route corridor. The car is inside when its horizontal (XZ) distance from
 * the projected centerline point is at most max(WIDTH_FACTOR * half_width,
 * WIDTH_FLOOR) and its height relative to that point lies in
 * [-BELOW, ABOVE]. Bounds come from every committed game capture: record lines reach
 * 24.6 m off the
 * centerline (E01 wall ride, 2.9 half-widths on an 8.3 m sample and 20
 * half-widths where the route width degenerates to 1.0 m), 17.6 m above
 * (E01) and 12.7 m below (E01, B04) while grounded; the A01 record's final
 * jump flies 39.2 m above the road it clears. A04's pool glide runs 22 to
 * 30 m below the raised route and is outside.
 */
#define TMNF_RACE_CORRIDOR_WIDTH_FACTOR 3.0f
#define TMNF_RACE_CORRIDOR_WIDTH_FLOOR_METERS 28.0f
#define TMNF_RACE_CORRIDOR_ABOVE_METERS 44.0f
#define TMNF_RACE_CORRIDOR_BELOW_METERS 16.0f

typedef struct {
	uint64_t visited_checkpoints;   /* bit i: checkpoint i taken this lap */
	uint64_t trigger_contacts;      /* low bits: checkpoints; finish i uses bit 63-i */
	uint32_t visited_count;
	uint32_t completed_laps;
	uint32_t passed_race_checkpoints;
	uint32_t elapsed_ticks;
	uint32_t finish_time_ms;
	uint32_t projection_segment;
	uint32_t centerline_segment;
	uint32_t off_track_ticks;
	uint32_t stuck_ticks;
	float arc_length;
	float lateral_offset;           /* 3D distance to the centerline */
	float half_width;
	float corridor_lateral;         /* XZ distance to the projected point */
	float corridor_vertical;        /* car y minus projected point y */
	float unwrapped_progress;
	float best_progress;
	float previous_progress;
	GmIso4 previous_car_transform;
	/* CTrackManiaPlayerInfo+0x274: where a respawn places the car. The last
	 * accepted checkpoint's spawn unless that block has no_respawn set
	 * (0x0047C330 OnCheckpoint keeps the previous one); the start spawn after
	 * a lap (0x00480820 OnFinishLine) and at reset (0x004831F0 ResetPlayer). */
	GmIso4 respawn_location;
	uint8_t finished;
	uint8_t outside_corridor;       /* progress frozen this tick */
	/* CTrackManiaPlayerInfo+0x2e8: a respawnable checkpoint was passed. Without
	 * it 0x00472700 SmallRespawn restarts the race instead of respawning. */
	uint8_t respawn_available;
	uint8_t reserved[1];
} TmnfRaceState;

typedef struct {
	uint8_t checkpoint_accepted;
	uint8_t checkpoint_repeated;
	uint8_t lap_completed;
	uint8_t finished;
	uint32_t checkpoint_index;
	uint32_t race_time_ms;
} TmnfRaceStepResult;

/*
 * The Stadium ground plane (StadiumGrass terrain: Grass, WetGrass). Every
 * Stadium block surface reports another id. Other collections have their own
 * terrain material (to be read from each environment's first track snapshot
 * before the race layer is used there).
 */
TMNF_HD static inline int TmnfRace_IsGroundPlaneMaterial(int32_t material_id)
{
	return material_id == TMNF_MATERIAL_GRASS ||
		material_id == TMNF_MATERIAL_WET_GRASS;
}

/*
 * Game-faithful trigger contact. The game registers a waypoint through the
 * collision manager: the waypoint mobil has collision enabled and a contact
 * sink (0x0047CBA0 CTrackManiaRaceTriggerAbsorbContact::AbsorbContact). The
 * test therefore mirrors 0x0053A8F0 CHmsCollisionManager_SZone::ComputeCollision
 * for the car tree against a trigger tree whose surface is the captured root
 * box: world-aligned root AABB overlap, per-child AABB overlap, then
 * 0x008EADC0 GmCollision_Ellipsoid_Mesh against the twelve box faces.
 */
TMNF_HD int TmnfRace_TriggerContact(
	const TmnfRouteTrigger *trigger,
	const CPlugTree *car_tree,
	const GmIso4 *car_world_transform);

/*
 * Every trigger of the route against the car tree at one transform: bit i
 * for checkpoint i, TMNF_RACE_FINISH_CONTACT_BIT for the finish. The
 * physics step calls this from each collision detection pass with the
 * predicted pre-response iso the detection itself used, which is when and
 * where the game raises the waypoint contact, and ORs the passes into
 * TmnfPhysicsWorld.trigger_contacts.
 */
TMNF_HD uint64_t TmnfRace_TriggerContactMask(
	const TmnfRoute *route,
	const CPlugTree *car_tree,
	const GmIso4 *car_world_transform);

TMNF_HD void TmnfRace_Reset(
	const TmnfRoute *route,
	TmnfRaceState *state,
	const GmIso4 *car_world_transform);

/*
 * Off-track counter. Counts ticks on which at least one wheel touches the
 * ground and every touching wheel is on the ground plane. Any wheel on a
 * track surface resets the counter. Airborne ticks hold it, so bouncing on
 * the grass cannot evade the rule. Returns 1 on the tick the count reaches
 * grace_ticks.
 */
TMNF_HD int TmnfRace_UpdateOffTrack(
	TmnfRaceState *state,
	uint32_t grace_ticks,
	uint32_t wheels_in_contact,
	uint32_t wheels_on_ground_plane);

/*
 * Advances race bookkeeping by one canonical 10 ms physics tick.
 * trigger_contacts is the tick's TmnfPhysicsWorld.trigger_contacts: the OR
 * of TmnfRace_TriggerContactMask over the step's detection passes. A trigger
 * event is a contact edge: contact this tick and none on the previous tick.
 * Checkpoints are counted once per lap in any order. A finish contact
 * completes a lap only when every checkpoint of the lap has been taken.
 * car_world_transform is the post-step state and only feeds the progress
 * projection, which is the environment's, not the game's. Projection uses
 * the previous dense segment plus or minus 32 segments. A displacement over
 * 32 metres in one tick is treated as a restore/teleport and reacquired
 * with one full dense-centerline search.
 */
TMNF_HD TmnfRaceStepResult TmnfRace_Step(
	const TmnfRoute *route,
	TmnfRaceState *state,
	uint64_t trigger_contacts,
	const GmIso4 *car_world_transform);

/*
 * The spawn a respawn press applies (0x00472700 CTrackManiaRace1P::SmallRespawn
 * -> 0x0047C0D0 RespawnPlayer -> 0x0047BF00 RespawnPlayerVehicle), or NULL
 * when no respawnable checkpoint has been passed: the game then restarts the
 * race (CGameRace::SetStatus(3)) rather than moving the car. The caller
 * applies it with World_Respawn before the tick's input mapping and physics
 * step.
 */
TMNF_HD const GmIso4 *TmnfRace_RespawnLocation(const TmnfRaceState *state);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_RACE_H */
