/* CHmsDyna: rigid-body integration and state management.
 * Object offsets recovered from the 2.11.26 decompilation. */
#ifndef TMNF_HMS_DYNA_H
#define TMNF_HMS_DYNA_H

#include "tmnf_hd.h"
#include <stdint.h>
#include "gm.h"
#include "hms_state.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Replacement buffer of GmVec3 penetration-correction vectors
 * (CHmsDyna+0x330). Game CFastBuffer layout is {count, data, capacity}; this is
 * a clean 64-bit-native equivalent (CHmsDyna is not byte-mirrored). */
typedef struct {
	uint32_t count;
	GmVec3  *data;
	uint32_t capacity;
} CHmsReplacementBuf;

/* Clean 64-bit-native layout (NOT byte-identical to the game object, which has
 * 4-byte pointers). The pure-float data structs CHmsStateDyna and
 * CHmsDynaParams ARE byte-identical on both architectures and carry the actual
 * simulated data. The replay harness reconstructs this struct from a captured
 * game THIS snapshot using these documented offsets:
 *   +0x0C0 clampAngular(int)   +0x0C4 maxAngularSpeed(float)
 *   +0x108 params ptr          +0x274 tempState (embedded CHmsStateDyna)
 *   +0x328 stateB ptr          +0x32C liveState ptr
 *   +0x330 replacementBuf (CFastBuffer) +0x33C dirtyFlag(int) +0x340 mode(int) */
typedef struct {
	int32_t         clampAngular;
	float           maxAngularSpeed;
	CHmsDynaParams *params;
	CHmsStateDyna   tempState;
	CHmsStateDyna  *stateB;
	CHmsStateDyna  *liveState;
	CHmsReplacementBuf replacementBuf;
	int32_t         dirtyFlag;
	int32_t         mode;
} CHmsDyna;

/* 0x00532D40 */ TMNF_HD void CHmsDyna_CopyStateToTemp(CHmsDyna *self);
/* 0x00532D60 */ TMNF_HD void CHmsDyna_CopyTempToState(CHmsDyna *self);
/* 0x00532D20 */ TMNF_HD void CHmsDyna_ValidateDynamicState(CHmsDyna *self);
/* 0x00533DD0 */ TMNF_HD void CHmsDyna_GetSpeed(CHmsDyna *self, const GmVec3 *at, GmVec3 *out);
/* 0x00533EE0 */ TMNF_HD void CHmsDyna_GetLinearSpeed(CHmsDyna *self, GmVec3 *out);
/* 0x00533F30 */ TMNF_HD void CHmsDyna_GetAngularSpeed(CHmsDyna *self, GmVec3 *out);
/* 0x00533A30 */ TMNF_HD void CHmsDyna_SetForce(CHmsDyna *self, const GmVec3 *f);
/* 0x00533A50 */ TMNF_HD void CHmsDyna_SetTorque(CHmsDyna *self, const GmVec3 *t);
/* 0x00533A70 */ TMNF_HD void CHmsDyna_AddForceAt(
	CHmsDyna *self, const GmVec3 *force, const GmVec3 *world_point);
/* 0x00533B80 */ TMNF_HD void CHmsDyna_AddForce(
	CHmsDyna *self, const GmVec3 *force);
/* 0x00533BB0 */ TMNF_HD void CHmsDyna_GetForce(
	CHmsDyna *self, GmVec3 *force);
/* 0x00533BE0 */ TMNF_HD void CHmsDyna_AddTorque(
	CHmsDyna *self, const GmVec3 *torque);
/* 0x00533EC0 */ TMNF_HD void CHmsDyna_SetLinearSpeed(
	CHmsDyna *self, const GmVec3 *speed);
/* 0x00533F10 */ TMNF_HD void CHmsDyna_SetAngularSpeed(
	CHmsDyna *self, const GmVec3 *speed);
/* 0x00533F60 */ TMNF_HD void CHmsDyna_AddLocalForceAt(
	CHmsDyna *self, const GmVec3 *force, const GmVec3 *local_point);
/* 0x00534030 */ TMNF_HD void CHmsDyna_SetLocalForce(
	CHmsDyna *self, const GmVec3 *force);
/* 0x005340B0 */ TMNF_HD void CHmsDyna_AddLocalForce(
	CHmsDyna *self, const GmVec3 *force);
/* 0x00534120 */ TMNF_HD void CHmsDyna_GetLocalForce(
	CHmsDyna *self, GmVec3 *force);
/* 0x00534140 */ TMNF_HD void CHmsDyna_SetLocalTorque(
	CHmsDyna *self, const GmVec3 *torque);
/* 0x005341C0 */ TMNF_HD void CHmsDyna_AddLocalTorque(
	CHmsDyna *self, const GmVec3 *torque);
/* 0x00534300 */ TMNF_HD void CHmsDyna_AddLocalImpulse(
	CHmsDyna *self, const GmVec3 *impulse);
/* 0x00534370 */ TMNF_HD void CHmsDyna_SetLocalLinearSpeed(
	CHmsDyna *self, const GmVec3 *speed);
/* 0x005343E0 */ TMNF_HD void CHmsDyna_GetLocalLinearSpeed(
	CHmsDyna *self, GmVec3 *speed);
/* 0x00534400 */ TMNF_HD void CHmsDyna_SetLocalAngularSpeed(
	CHmsDyna *self, const GmVec3 *speed);
/* 0x00534470 */ TMNF_HD void CHmsDyna_GetLocalAngularSpeed(
	CHmsDyna *self, GmVec3 *speed);
/* 0x005334E0 */ TMNF_HD void CHmsDyna_ApplyReplacement(CHmsDyna *self, const GmVec3 *d);
/* 0x00533C10 */ TMNF_HD void CHmsDyna_AddImpulseAt(CHmsDyna *self, const GmVec3 *J, const GmVec3 *at);
/* 0x00533D50 */ TMNF_HD void CHmsDyna_AddImpulse(CHmsDyna *self, const GmVec3 *J);
/* 0x00533510 */ TMNF_HD void CHmsDyna_IntegrateStep(CHmsDyna *self, const CHmsStateDyna *in,
	CHmsStateDyna *out, float dt);
/* 0x00535D00 */ TMNF_HD void CHmsDyna_DoPreCollisionDynamic(CHmsDyna *self, float dt);
/* 0x00536400 */ TMNF_HD void CHmsDyna_DoPostCollisionDynamic(CHmsDyna *self);
/* 0x00535D50 */ TMNF_HD void CHmsDyna_ComputeSynthetizedReplacement(CHmsDyna *self, GmVec3 *out);
/* 0x00535FC0 */ TMNF_HD void CHmsDyna_AddReplacement(CHmsDyna *self, const GmVec3 *d);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_HMS_DYNA_H */
