#include "hms_item.h"

TMNF_HD static CHmsDyna *corpus_dyna(const CHmsCorpus *corpus) {
	return (CHmsDyna *)corpus->dyna;
}

TMNF_HD static CHmsDyna *first_dyna(const CHmsItem *self) {
	if (self->corpus_count == 0) {
		return NULL;
	}
	return corpus_dyna(self->corpora[0]);
}

/* 0x0053BA40 */
TMNF_HD void *CHmsItem_GetZone(const CHmsItem *self, uint32_t corpus_index) {
	return self->zones[corpus_index];
}

/* 0x0053BB60 */
TMNF_HD void CHmsItem_GetLinearSpeed(const CHmsItem *self, GmVec3 *out) {
	CHmsDyna *dyna = first_dyna(self);
	if (dyna == NULL) {
		*out = (GmVec3){ 0.0f, 0.0f, 0.0f };
		return;
	}
	CHmsDyna_GetLocalLinearSpeed(dyna, out);
}

/* 0x0053BBA0 */
TMNF_HD void CHmsItem_GetAngularSpeed(const CHmsItem *self, GmVec3 *out) {
	CHmsDyna *dyna = first_dyna(self);
	if (dyna == NULL) {
		*out = (GmVec3){ 0.0f, 0.0f, 0.0f };
		return;
	}
	CHmsDyna_GetLocalAngularSpeed(dyna, out);
}

/* 0x0053BBE0 */
TMNF_HD void CHmsItem_GetForce(const CHmsItem *self, GmVec3 *out) {
	CHmsDyna *dyna = first_dyna(self);
	if (dyna == NULL) {
		*out = (GmVec3){ 0.0f, 0.0f, 0.0f };
		return;
	}
	CHmsDyna_GetLocalForce(dyna, out);
}

/* 0x0053CE10 */
TMNF_HD void CHmsItem_SetLinearSpeed(CHmsItem *self, const GmVec3 *speed) {
	for (uint32_t i = 0; i < self->corpus_count; ++i) {
		CHmsDyna *dyna = corpus_dyna(self->corpora[i]);
		if (dyna != NULL) {
			CHmsDyna_SetLocalLinearSpeed(dyna, speed);
		}
	}
}

/* 0x0053CE60 */
TMNF_HD void CHmsItem_SetAngularSpeed(CHmsItem *self, const GmVec3 *speed) {
	for (uint32_t i = 0; i < self->corpus_count; ++i) {
		CHmsDyna *dyna = corpus_dyna(self->corpora[i]);
		if (dyna != NULL) {
			CHmsDyna_SetLocalAngularSpeed(dyna, speed);
		}
	}
}

/* 0x0053CEB0 */
TMNF_HD void CHmsItem_AddForce(CHmsItem *self, const GmVec3 *force) {
	for (uint32_t i = 0; i < self->corpus_count; ++i) {
		CHmsDyna *dyna = corpus_dyna(self->corpora[i]);
		if (dyna != NULL) {
			CHmsDyna_AddLocalForce(dyna, force);
		}
	}
}

/* 0x0053CF00 */
TMNF_HD void CHmsItem_AddTorque(CHmsItem *self, const GmVec3 *torque) {
	for (uint32_t i = 0; i < self->corpus_count; ++i) {
		CHmsDyna *dyna = corpus_dyna(self->corpora[i]);
		if (dyna != NULL) {
			CHmsDyna_AddLocalTorque(dyna, torque);
		}
	}
}

/* 0x0053CF50 */
TMNF_HD void CHmsItem_AddForceAt(
	CHmsItem *self, const GmVec3 *force, const GmVec3 *local_point) {
	for (uint32_t i = 0; i < self->corpus_count; ++i) {
		CHmsDyna *dyna = corpus_dyna(self->corpora[i]);
		if (dyna != NULL) {
			CHmsDyna_AddLocalForceAt(dyna, force, local_point);
		}
	}
}

/* 0x0053CFA0 */
TMNF_HD void CHmsItem_SetForce(CHmsItem *self, const GmVec3 *force) {
	for (uint32_t i = 0; i < self->corpus_count; ++i) {
		CHmsDyna *dyna = corpus_dyna(self->corpora[i]);
		if (dyna != NULL) {
			CHmsDyna_SetLocalForce(dyna, force);
		}
	}
}

/* 0x0053CFF0 */
TMNF_HD void CHmsItem_SetTorque(CHmsItem *self, const GmVec3 *torque) {
	for (uint32_t i = 0; i < self->corpus_count; ++i) {
		CHmsDyna *dyna = corpus_dyna(self->corpora[i]);
		if (dyna != NULL) {
			CHmsDyna_SetLocalTorque(dyna, torque);
		}
	}
}

/* 0x0053D090 */
TMNF_HD void CHmsItem_AddImpulse(CHmsItem *self, const GmVec3 *impulse) {
	for (uint32_t i = 0; i < self->corpus_count; ++i) {
		CHmsDyna *dyna = corpus_dyna(self->corpora[i]);
		if (dyna != NULL) {
			CHmsDyna_AddLocalImpulse(dyna, impulse);
		}
	}
}
