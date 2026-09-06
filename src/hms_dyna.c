/* CHmsDyna integration core, transcribed from the 2.11.26 disassembly.
 * Golden traces show that Direct3D leaves the live game's x87 precision
 * control at 24 bits, so every x87 arithmetic instruction rounds to float.
 *
 * VALIDATED against golden traces where noted; buffer-dependent functions
 * (DoPre/DoPostCollisionDynamic, ComputeSynthetizedReplacement, AddReplacement)
 * live in a later pass once the CFastBuffer element ops land. */
#include "hms_dyna.h"
#include "tmnf_fp.h"

#include <stdlib.h>

/* FP constants (raw float32 bit patterns from analysis/fp_constants.csv). */
#define DAT_00cdb690 9.999999439624929e-11f  /* ~1e-10 magnitude epsilon */
#define DAT_00cdb67c 0.009999999776482582f   /* 0.01, max replacement/step */

TMNF_HD static inline float x87_length_sq(float x, float y, float z) {
	return x87_add(x87_add(x87_mul(y, y), x87_mul(x, x)), x87_mul(z, z));
}

/* 0x00532D40  temp = *live */
TMNF_HD void CHmsDyna_CopyStateToTemp(CHmsDyna *self) {
	self->tempState = *self->liveState;
}

/* 0x00532D60  *stateB = temp */
TMNF_HD void CHmsDyna_CopyTempToState(CHmsDyna *self) {
	*self->stateB = self->tempState;
}

/* 0x00532D20  *stateB = *live */
TMNF_HD void CHmsDyna_ValidateDynamicState(CHmsDyna *self) {
	*self->stateB = *self->liveState;
}

/* 0x00533EE0 */
TMNF_HD void CHmsDyna_GetLinearSpeed(CHmsDyna *self, GmVec3 *out) {
	*out = self->liveState->linVel;
}

/* 0x00533F30 */
TMNF_HD void CHmsDyna_GetAngularSpeed(CHmsDyna *self, GmVec3 *out) {
	*out = self->liveState->angVel;
}

/* 0x00533A30 */
TMNF_HD void CHmsDyna_SetForce(CHmsDyna *self, const GmVec3 *f) {
	self->liveState->force = *f;
}

/* 0x00533A50 */
TMNF_HD void CHmsDyna_SetTorque(CHmsDyna *self, const GmVec3 *t) {
	self->liveState->torque = *t;
}

/* 0x00533B80. UNVALIDATED. */
TMNF_HD void CHmsDyna_AddForce(CHmsDyna *self, const GmVec3 *force) {
	CHmsStateDyna *state = self->liveState;
	state->force.x = x87_add(state->force.x, force->x);
	state->force.y = x87_add(force->y, state->force.y);
	state->force.z = x87_add(force->z, state->force.z);
}

/* 0x00533BB0. */
TMNF_HD void CHmsDyna_GetForce(CHmsDyna *self, GmVec3 *force) {
	*force = self->liveState->force;
}

/* 0x00533BE0. UNVALIDATED. */
TMNF_HD void CHmsDyna_AddTorque(CHmsDyna *self, const GmVec3 *torque) {
	CHmsStateDyna *state = self->liveState;
	state->torque.x = x87_add(state->torque.x, torque->x);
	state->torque.y = x87_add(torque->y, state->torque.y);
	state->torque.z = x87_add(torque->z, state->torque.z);
}

/* 0x00533EC0. */
TMNF_HD void CHmsDyna_SetLinearSpeed(CHmsDyna *self, const GmVec3 *speed) {
	self->liveState->linVel = *speed;
}

/* 0x00533F10. */
TMNF_HD void CHmsDyna_SetAngularSpeed(CHmsDyna *self, const GmVec3 *speed) {
	self->liveState->angVel = *speed;
}

/* 0x00533A70. Adds a world-space force and its moment about the world-space
 * centre of mass. UNVALIDATED. */
TMNF_HD void CHmsDyna_AddForceAt(
	CHmsDyna *self, const GmVec3 *force, const GmVec3 *world_point) {
	CHmsStateDyna *state = self->liveState;
	CHmsDyna_AddForce(self, force);

	GmVec3 world_com;
	GmVec3_SetMult_Iso4(
		&world_com, &self->params->comOffset,
		(const GmIso4 *)&state->rot);
	GmVec3 lever = {
		x87_sub(world_point->x, world_com.x),
		x87_sub(world_point->y, world_com.y),
		x87_sub(world_point->z, world_com.z),
	};
	GmVec3 torque = {
		x87_sub(
			x87_mul(lever.y, force->z),
			x87_mul(lever.z, force->y)),
		x87_sub(
			x87_mul(force->x, lever.z),
			x87_mul(lever.x, force->z)),
		x87_sub(
			x87_mul(force->y, lever.x),
			x87_mul(lever.y, force->x)),
	};
	CHmsDyna_AddTorque(self, &torque);
}

TMNF_HD static GmVec3 local_to_world_vector(
	const CHmsDyna *self, const GmVec3 *local) {
	GmVec3 world;
	GmVec3_SetMult_Mat3(&world, local, &self->liveState->rot);
	return world;
}

/* 0x00533F60. UNVALIDATED. */
TMNF_HD void CHmsDyna_AddLocalForceAt(
	CHmsDyna *self, const GmVec3 *force, const GmVec3 *local_point) {
	GmVec3 world_force = local_to_world_vector(self, force);
	GmVec3 world_point;
	GmVec3_SetMult_Iso4(
		&world_point, local_point,
		(const GmIso4 *)&self->liveState->rot);
	CHmsDyna_AddForceAt(self, &world_force, &world_point);
}

/* 0x00534030. UNVALIDATED. */
TMNF_HD void CHmsDyna_SetLocalForce(CHmsDyna *self, const GmVec3 *force) {
	GmVec3 world = local_to_world_vector(self, force);
	CHmsDyna_SetForce(self, &world);
}

/* 0x005340B0. UNVALIDATED. */
TMNF_HD void CHmsDyna_AddLocalForce(CHmsDyna *self, const GmVec3 *force) {
	GmVec3 world = local_to_world_vector(self, force);
	CHmsDyna_AddForce(self, &world);
}

/* 0x00534120. UNVALIDATED. */
TMNF_HD void CHmsDyna_GetLocalForce(CHmsDyna *self, GmVec3 *force) {
	CHmsDyna_GetForce(self, force);
	GmVec3_MultTranspose(force, &self->liveState->rot);
}

/* 0x00534140. UNVALIDATED. */
TMNF_HD void CHmsDyna_SetLocalTorque(CHmsDyna *self, const GmVec3 *torque) {
	GmVec3 world = local_to_world_vector(self, torque);
	CHmsDyna_SetTorque(self, &world);
}

/* 0x005341C0. UNVALIDATED. */
TMNF_HD void CHmsDyna_AddLocalTorque(CHmsDyna *self, const GmVec3 *torque) {
	GmVec3 world = local_to_world_vector(self, torque);
	CHmsDyna_AddTorque(self, &world);
}

/* 0x00534300. UNVALIDATED. */
TMNF_HD void CHmsDyna_AddLocalImpulse(CHmsDyna *self, const GmVec3 *impulse) {
	GmVec3 world = local_to_world_vector(self, impulse);
	CHmsDyna_AddImpulse(self, &world);
}

/* 0x00534370. UNVALIDATED. */
TMNF_HD void CHmsDyna_SetLocalLinearSpeed(CHmsDyna *self, const GmVec3 *speed) {
	GmVec3 world = local_to_world_vector(self, speed);
	CHmsDyna_SetLinearSpeed(self, &world);
}

/* 0x005343E0. UNVALIDATED. */
TMNF_HD void CHmsDyna_GetLocalLinearSpeed(CHmsDyna *self, GmVec3 *speed) {
	CHmsDyna_GetLinearSpeed(self, speed);
	GmVec3_MultTranspose(speed, &self->liveState->rot);
}

/* 0x00534400. UNVALIDATED. */
TMNF_HD void CHmsDyna_SetLocalAngularSpeed(CHmsDyna *self, const GmVec3 *speed) {
	GmVec3 world = local_to_world_vector(self, speed);
	CHmsDyna_SetAngularSpeed(self, &world);
}

/* 0x00534470. UNVALIDATED. */
TMNF_HD void CHmsDyna_GetLocalAngularSpeed(CHmsDyna *self, GmVec3 *speed) {
	CHmsDyna_GetAngularSpeed(self, speed);
	GmVec3_MultTranspose(speed, &self->liveState->rot);
}

/* 0x005334E0  live.pos += d */
TMNF_HD void CHmsDyna_ApplyReplacement(CHmsDyna *self, const GmVec3 *d) {
	CHmsStateDyna *s = self->liveState;
	s->pos.x = F(s->pos.x) + F(d->x);
	s->pos.y = F(d->y) + F(s->pos.y);
	s->pos.z = F(d->z) + F(s->pos.z);
}

/* 0x00533DD0  velocity at world point `at` = linVel + angVel x (at - worldCOM) */
TMNF_HD void CHmsDyna_GetSpeed(CHmsDyna *self, const GmVec3 *at, GmVec3 *out) {
	CHmsStateDyna *s = self->liveState;
	if (self->mode == 2) {
		out->x = 0.0f;
		out->y = 0.0f;
		out->z = 0.0f;
		return;
	}
	*out = s->linVel;
	if (self->mode == 1) {
		GmVec3 comWorld;
		/* rot(0x10) contiguous with pos(0x34) is treated as a GmIso4. */
		GmVec3_SetMult_Iso4(&comWorld, &self->params->comOffset,
			(const GmIso4 *)&s->rot);
		double rx = F(at->x) - F(comWorld.x);
		double ry = F(at->y) - F(comWorld.y);
		double rz = F(at->z) - F(comWorld.z);
		out->x = F(out->x) + (F(s->angVel.y) * rz - ry * F(s->angVel.z));
		out->y = F(out->y) + (F(s->angVel.z) * rx - rz * F(s->angVel.x));
		out->z = F(out->z) + (F(s->angVel.x) * ry - rx * F(s->angVel.y));
	}
}

/* 0x00533D50  central linear impulse: linVel += J/mass */
TMNF_HD void CHmsDyna_AddImpulse(CHmsDyna *self, const GmVec3 *J) {
	if (self->mode == 2) {
		return;
	}
	if (self->dirtyFlag == 0) {
		self->dirtyFlag = 1;
	}
	float invMass = x87_rcp(self->params->mass);
	CHmsStateDyna *s = self->liveState;
	s->linVel.x =
		x87_add(x87_mul(invMass, J->x), s->linVel.x);
	s->linVel.y =
		x87_add(s->linVel.y, x87_mul(J->y, invMass));
	s->linVel.z =
		x87_add(x87_mul(invMass, J->z), s->linVel.z);
}

/* 0x00533C10  impulse J at world point `at`: linear + angular */
TMNF_HD void CHmsDyna_AddImpulseAt(CHmsDyna *self, const GmVec3 *J, const GmVec3 *at) {
	if (self->mode == 2) {
		return;
	}
	if (self->dirtyFlag == 0) {
		self->dirtyFlag = 1;
	}
	CHmsStateDyna *s = self->liveState;
	float invMass = x87_rcp(self->params->mass);
	s->linVel.x =
		x87_add(s->linVel.x, x87_mul(invMass, J->x));
	s->linVel.y =
		x87_add(s->linVel.y, x87_mul(J->y, invMass));
	s->linVel.z =
		x87_add(x87_mul(invMass, J->z), s->linVel.z);
	if (self->mode == 1) {
		GmVec3 comWorld;
		GmVec3_SetMult_Iso4(&comWorld, &self->params->comOffset,
			(const GmIso4 *)&s->rot);
		float rx = x87_sub(at->x, comWorld.x);
		float ry = x87_sub(at->y, comWorld.y);
		float rz = x87_sub(at->z, comWorld.z);
		GmVec3 tau;
		tau.x = x87_sub(
			x87_mul(ry, J->z), x87_mul(rz, J->y));
		tau.y = x87_sub(
			x87_mul(rz, J->x), x87_mul(rx, J->z));
		tau.z = x87_sub(
			x87_mul(rx, J->y), x87_mul(J->x, ry));
		GmVec3_Mult_Mat3(&tau, &s->invInertiaWorld);
		s->angVel.x = F(s->angVel.x) + F(tau.x);
		s->angVel.y = F(s->angVel.y) + F(tau.y);
		s->angVel.z = F(tau.z) + F(s->angVel.z);
	}
}

/* 0x00533510  one integration substep: out = integrate(in, dt) */
TMNF_HD void CHmsDyna_IntegrateStep(CHmsDyna *self, const CHmsStateDyna *in,
	CHmsStateDyna *out, float dt) {
	if (self->mode == 2) {
		*out = *in;
		return;
	}

	float invMass = x87_rcp(self->params->mass);
	float force_x = x87_mul(in->force.x, invMass);
	float force_y = x87_mul(in->force.y, invMass);
	float force_z = x87_mul(invMass, in->force.z);
	float pos_dx = x87_mul(in->linVel.x, dt);
	float pos_dy = x87_mul(in->linVel.y, dt);
	float pos_dz = x87_mul(in->linVel.z, dt);

	out->pos.x = x87_add(pos_dx, in->pos.x);
	out->pos.y = x87_add(in->pos.y, pos_dy);
	out->pos.z = x87_add(in->pos.z, pos_dz);
	float added_dx = x87_mul(in->linVelAdded.x, dt);
	float added_dy = x87_mul(in->linVelAdded.y, dt);
	float added_dz = x87_mul(in->linVelAdded.z, dt);
	out->pos.x = x87_add(out->pos.x, added_dx);
	out->pos.y = x87_add(out->pos.y, added_dy);
	out->pos.z = x87_add(out->pos.z, added_dz);
	out->linVelAdded.z = 0.0f;
	out->linVelAdded.y = 0.0f;
	out->linVelAdded.x = 0.0f;
	out->linVel.x = x87_add(in->linVel.x, x87_mul(force_x, dt));
	out->linVel.y = x87_add(in->linVel.y, x87_mul(force_y, dt));
	out->linVel.z = x87_add(in->linVel.z, x87_mul(dt, force_z));

	if (self->mode != 0) {
		GmVec3 angAcc; /* Iinv_world * torque */
		GmVec3_SetMult_Mat3(&angAcc, &in->torque, &in->invInertiaWorld);

		float wx = in->angVel.x, wy = in->angVel.y, wz = in->angVel.z;
		if (x87_length_sq(wx, wy, wz) <= DAT_00cdb690) {
			GmMat3_Set(&out->rot, &in->rot);
			GmVec4_Set(&out->quat, &in->quat);
		} else {
			float qx = in->quat.x, qy = in->quat.y;
			float qz = in->quat.z, qw = in->quat.w;
			float dqx = x87_mul(x87_sub(x87_sub(x87_mul(-wx, qy),
				x87_mul(wy, qz)), x87_mul(qw, wz)), 0.5f);
			float dqy = x87_mul(x87_sub(x87_add(x87_mul(qx, wx),
				x87_mul(wy, qw)), x87_mul(qz, wz)), 0.5f);
			float dqz = x87_mul(x87_add(x87_sub(x87_mul(wy, qx),
				x87_mul(qw, wx)), x87_mul(wz, qy)), 0.5f);
			float dqw = x87_mul(x87_add(x87_sub(x87_mul(qz, wx),
				x87_mul(wy, qy)), x87_mul(qx, wz)), 0.5f);
			GmVec4_Set(&out->quat, &in->quat);
			out->quat.x = x87_add(x87_mul(dqx, dt), out->quat.x);
			out->quat.y = x87_add(x87_mul(dqy, dt), out->quat.y);
			out->quat.z = x87_add(x87_mul(dqz, dt), out->quat.z);
			out->quat.w = x87_add(x87_mul(dt, dqw), out->quat.w);
			GmQuat_Normalize(&out->quat);
			GmMat3_SetFromQuat(&out->rot, out->quat.x, out->quat.y,
				out->quat.z, out->quat.w);
			GmVec3 comOld, comNew;
			GmVec3_SetMult_Mat3(&comOld, &self->params->comOffset, &in->rot);
			GmVec3_SetMult_Mat3(&comNew, &self->params->comOffset, &out->rot);
			float com_dx = x87_sub(comNew.x, comOld.x);
			float com_dy = x87_sub(comNew.y, comOld.y);
			float com_dz = x87_sub(comNew.z, comOld.z);
			out->pos.x = x87_sub(out->pos.x, com_dx);
			out->pos.y = x87_sub(out->pos.y, com_dy);
			out->pos.z = x87_sub(out->pos.z, com_dz);
		}
		out->angVel.x = x87_add(in->angVel.x, x87_mul(dt, angAcc.x));
		out->angVel.y = x87_add(in->angVel.y, x87_mul(angAcc.y, dt));
		out->angVel.z = x87_add(in->angVel.z, x87_mul(dt, angAcc.z));

		if (self->clampAngular != 0) {
			float mx = self->maxAngularSpeed;
			float nsq = x87_length_sq(out->angVel.x, out->angVel.y,
				out->angVel.z);
			if (x87_mul(mx, mx) < nsq) {
				float s = x87_sqrt(nsq);
				float ratio = x87_div(mx, s);
				out->angVel.x = x87_mul(out->angVel.x, ratio);
				out->angVel.y = x87_mul(ratio, out->angVel.y);
				out->angVel.z = x87_mul(ratio, out->angVel.z);
			}
		}

		GmMat3_SetTranspose(&out->invInertiaWorld, &out->rot);
		GmMat3_Mult(&out->invInertiaWorld, &self->params->invInertiaBody);
		GmMat3_Mult(&out->invInertiaWorld, &out->rot);
		return;
	}

	GmMat3_Set(&out->rot, &in->rot);
}

/* 0x00535D00  integrate live state one step, then clear the replacement buffer.
 * UNVALIDATED: no golden trace captured yet for this VA. */
TMNF_HD void CHmsDyna_DoPreCollisionDynamic(CHmsDyna *self, float dt) {
	CHmsStateDyna *live = self->liveState;
	CHmsStateDyna local = *live;
	CHmsDyna_IntegrateStep(self, &local, live, dt);
	/* rot/pos are read as an iso by vector loads for the rest of the tick. */
	GmIso4_Relayout((GmIso4 *)&live->rot);
	self->replacementBuf.count = 0;
}

/* 0x00535FC0  append a penetration-correction vector and mark dirty.
 * UNVALIDATED. */
TMNF_HD void CHmsDyna_AddReplacement(CHmsDyna *self, const GmVec3 *d) {
	if (self->dirtyFlag == 0) {
		self->dirtyFlag = 1;
	}
	CHmsReplacementBuf *b = &self->replacementBuf;
	if (b->count == b->capacity) {
#if defined(__CUDA_ARCH__)
		tmnf_fail("replacement buffer capacity exceeded");
#else
		uint32_t requested = b->count + 1;
		uint32_t grown = b->capacity + (b->capacity >> 1);
		uint32_t cap = requested > grown ? requested : grown;
		GmVec3 *data = (GmVec3 *)realloc(b->data, cap * sizeof(GmVec3));
		if (data == NULL) {
			tmnf_abort();
		}
		b->data = data;
		b->capacity = cap;
#endif
	}
	GmVec3 *slot = &b->data[b->count++];
	slot->x = d->x;
	slot->y = d->y;
	slot->z = d->z;
}

/* 0x00535D50  reduce accumulated replacement vectors into one clamped vector.
 * UNVALIDATED: transcribed from the decompiler order; refine against a golden
 * trace with tools/x87trace.py once 0x00535D50 is captured. */
TMNF_HD void CHmsDyna_ComputeSynthetizedReplacement(CHmsDyna *self, GmVec3 *out) {
	CHmsReplacementBuf *b = &self->replacementBuf;
	if (b->count == 0) {
		out->x = 0.0f;
		out->y = 0.0f;
		out->z = 0.0f;
		return;
	}
	float ax = b->data[0].x, ay = b->data[0].y, az = b->data[0].z;
	for (uint32_t i = 1; i < b->count; i++) {
		float px = b->data[i].x, py = b->data[i].y, pz = b->data[i].z;
		float dot = x87_add(x87_add(x87_mul(ax, px), x87_mul(ay, py)),
			x87_mul(az, pz));
		float mag = x87_length_sq(ax, ay, az);
		if (dot > 0.0f && mag > DAT_00cdb690) {
			if (mag < dot) {
				dot = mag;
			}
			dot = x87_div(dot, mag);
			ax = x87_sub(ax, x87_mul(dot, ax));
			ay = x87_sub(ay, x87_mul(dot, ay));
			az = x87_sub(az, x87_mul(dot, az));
		}
		ax = x87_add(px, ax);
		ay = x87_add(ay, py);
		az = x87_add(az, pz);
	}
	float mag = x87_length_sq(ax, ay, az);
	if (mag <= x87_mul(DAT_00cdb67c, DAT_00cdb67c)) {
		out->x = 0.0f;
		out->y = 0.0f;
		out->z = 0.0f;
		return;
	}
	float s = x87_sqrt(mag);
	float inv = x87_rcp(s);
	float nx = x87_mul(ax, inv);
	float ny = x87_mul(ay, inv);
	float nz = x87_mul(az, inv);
	out->x = x87_sub(ax, x87_mul(DAT_00cdb67c, nx));
	out->y = x87_sub(ay, x87_mul(DAT_00cdb67c, ny));
	out->z = x87_sub(az, x87_mul(DAT_00cdb67c, nz));
}

/* 0x00536400  apply the synthesized replacement to the live position.
 * UNVALIDATED. */
TMNF_HD void CHmsDyna_DoPostCollisionDynamic(CHmsDyna *self) {
	GmVec3 repl;
	CHmsDyna_ComputeSynthetizedReplacement(self, &repl);
	CHmsDyna_ApplyReplacement(self, &repl);
}
