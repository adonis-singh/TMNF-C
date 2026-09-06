/* In-race respawn of the player car (analysis/respawn.md).
 *
 * 0x0047BF00 CTrackManiaRace::RespawnPlayerVehicle, reached from the Enter
 * press through 0x0047DCD0 CTrackManiaRace::OnInputEvent -> 0x00472700
 * CTrackManiaRace1P::SmallRespawn -> 0x0047C0D0 RespawnPlayer(seed 0):
 *   1. CSceneMobil vtable +0x130: 0x007C0320 CSceneVehicleCar::VehicleReset
 *      (inputs, springs, turbo, air control, events, engine, four
 *      0x007BD2A0 WheelReset, 0x007BC9A0 SEngine::Reset), whose base
 *      0x007CB6B0 CSceneVehicle::VehicleReset calls 0x0053D340
 *      CHmsItem::ResetDynamicState;
 *   2. 0x0053D340 CHmsItem::ResetDynamicState again -> 0x005474E0
 *      CHmsCorpus::Reset -> 0x00535CB0 CHmsDyna::Reset: velocities, forces
 *      and the replacement buffer of the live, committed and temp states;
 *   3. vtable +0x140: 0x007BC800 VehicleBlockSpeed2Set(0);
 *   4. vtable +0x88: 0x007B2E00 CSceneMobil::SetLocation(spawn) -> 0x0053D550
 *      CHmsItem::SetLocation -> 0x005477B0 CHmsCorpus::SetLocation ->
 *      0x005338F0 CHmsDyna::SetLocation: quaternion, rotation, position and
 *      the world inverse inertia of both dyna states.
 * A player respawn passes seed 0, so the seeded yaw jitter of
 * RespawnPlayerVehicle does not apply. The race timer keeps running; the
 * checkpoint state is the race layer's (TmnfRaceState.respawn_location).
 */
#ifndef TMNF_VEHICLE_RESPAWN_H
#define TMNF_VEHICLE_RESPAWN_H

#include "gm.h"
#include "physics.h"
#include "tmnf_hd.h"

#ifdef __cplusplus
extern "C" {
#endif

TMNF_HD void TmnfVehicle_Respawn(TmnfPhysicsCorpus *corpus, const GmIso4 *spawn);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_VEHICLE_RESPAWN_H */
