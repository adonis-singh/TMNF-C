/* Core physics state layouts, offsets recovered from the 2.11.26
 * decompilation (CHmsDyna::IntegrateStep, GetSpeed, SetForce, CopyStateToTemp).
 *
 * CHmsStateDyna is the 180-byte (0xB4, 45 dwords) integrated rigid-body state.
 * CopyStateToTemp/CopyTempToState move exactly 0x2d dwords. */
#ifndef TMNF_HMS_STATE_H
#define TMNF_HMS_STATE_H

#include <stddef.h>
#include "gm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	GmQuat  quat;            /* 0x00  orientation (x,y,z,w)                    */
	GmMat3  rot;             /* 0x10  rotation matrix (from quat)             */
	GmVec3  pos;             /* 0x34  position, integrated by linVel+linVelAdd*/
	GmVec3  linVel;          /* 0x40  linear velocity, integrated by force    */
	GmVec3  linVelAdded;     /* 0x4C  added linear speed accumulator (zeroed) */
	GmVec3  angVel;          /* 0x58  angular velocity                        */
	GmVec3  force;           /* 0x64  accumulated force                       */
	GmVec3  torque;          /* 0x70  accumulated torque                      */
	GmMat3  invInertiaWorld; /* 0x7C  R * Ibody^-1 * R^T                       */
	float   tail[5];         /* 0xA0  5 dwords, purpose TBD                    */
} CHmsStateDyna;             /* 0xB4 = 180 bytes                               */

_Static_assert(sizeof(CHmsStateDyna) == 180, "CHmsStateDyna must be 180 bytes");
_Static_assert(offsetof(CHmsStateDyna, quat) == 0x00, "quat");
_Static_assert(offsetof(CHmsStateDyna, rot) == 0x10, "rot");
_Static_assert(offsetof(CHmsStateDyna, pos) == 0x34, "pos");
_Static_assert(offsetof(CHmsStateDyna, linVel) == 0x40, "linVel");
_Static_assert(offsetof(CHmsStateDyna, linVelAdded) == 0x4C, "linVelAdded");
_Static_assert(offsetof(CHmsStateDyna, angVel) == 0x58, "angVel");
_Static_assert(offsetof(CHmsStateDyna, force) == 0x64, "force");
_Static_assert(offsetof(CHmsStateDyna, torque) == 0x70, "torque");
_Static_assert(offsetof(CHmsStateDyna, invInertiaWorld) == 0x7C, "invInertiaWorld");

/* CHmsDyna params block, pointed to by CHmsDyna+0x108.
 * mass at +0x00, body inverse-inertia GmMat3 at +0x04, centre-of-mass offset
 * GmVec3 at +0x38. Remaining fields TBD. */
typedef struct {
	float  mass;             /* 0x00  float[0]                                */
	GmMat3 invInertiaBody;   /* 0x04                                          */
	float  dragLinear;       /* 0x28  float[10], linear velocity damping      */
	float  dragAngular;      /* 0x2C  float[11], angular velocity damping     */
	float  substepLen;       /* 0x30  float[12], length scale for substepping */
	float  forceFieldScale;  /* 0x34  float[13], multiplies force-field * mass */
	GmVec3 comOffset;        /* 0x38  centre-of-mass offset                   */
} CHmsDynaParams;

_Static_assert(offsetof(CHmsDynaParams, invInertiaBody) == 0x04, "invInertiaBody");
_Static_assert(offsetof(CHmsDynaParams, dragLinear) == 0x28, "dragLinear");
_Static_assert(offsetof(CHmsDynaParams, dragAngular) == 0x2C, "dragAngular");
_Static_assert(offsetof(CHmsDynaParams, substepLen) == 0x30, "substepLen");
_Static_assert(offsetof(CHmsDynaParams, forceFieldScale) == 0x34, "forceFieldScale");
_Static_assert(offsetof(CHmsDynaParams, comOffset) == 0x38, "comOffset");

#ifdef __cplusplus
}
#endif

#endif /* TMNF_HMS_STATE_H */
