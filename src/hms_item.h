/* CHmsItem facade over one or more rigid-body corpora.
 *
 * The game stores its corpus buffer at CHmsItem+0x34; this clean native form
 * keeps typed arrays and delegates the public item operations to CHmsDyna.
 */
#ifndef TMNF_HMS_ITEM_H
#define TMNF_HMS_ITEM_H

#include "tmnf_hd.h"
#include <stdint.h>

#include "collision.h"
#include "hms_dyna.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	CHmsCorpus **corpora;
	void **zones;
	uint32_t corpus_count;
} CHmsItem;

/* 0x0053BA40 */ TMNF_HD void *CHmsItem_GetZone(
	const CHmsItem *self, uint32_t corpus_index);
/* 0x0053BB60 */ TMNF_HD void CHmsItem_GetLinearSpeed(
	const CHmsItem *self, GmVec3 *out);
/* 0x0053BBA0 */ TMNF_HD void CHmsItem_GetAngularSpeed(
	const CHmsItem *self, GmVec3 *out);
/* 0x0053BBE0 */ TMNF_HD void CHmsItem_GetForce(
	const CHmsItem *self, GmVec3 *out);
/* 0x0053CE10 */ TMNF_HD void CHmsItem_SetLinearSpeed(
	CHmsItem *self, const GmVec3 *speed);
/* 0x0053CE60 */ TMNF_HD void CHmsItem_SetAngularSpeed(
	CHmsItem *self, const GmVec3 *speed);
/* 0x0053CEB0 */ TMNF_HD void CHmsItem_AddForce(
	CHmsItem *self, const GmVec3 *force);
/* 0x0053CF00 */ TMNF_HD void CHmsItem_AddTorque(
	CHmsItem *self, const GmVec3 *torque);
/* 0x0053CF50 */ TMNF_HD void CHmsItem_AddForceAt(
	CHmsItem *self, const GmVec3 *force, const GmVec3 *local_point);
/* 0x0053CFA0 */ TMNF_HD void CHmsItem_SetForce(
	CHmsItem *self, const GmVec3 *force);
/* 0x0053CFF0 */ TMNF_HD void CHmsItem_SetTorque(
	CHmsItem *self, const GmVec3 *torque);
/* 0x0053D090 */ TMNF_HD void CHmsItem_AddImpulse(
	CHmsItem *self, const GmVec3 *impulse);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_HMS_ITEM_H */
