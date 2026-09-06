/* Geometry leaf ops, transcribed from the 2.11.26 disassembly.
 * Golden traces show that Direct3D leaves the live game's x87 precision
 * control at 24 bits. Rounding therefore occurs after every x87 arithmetic
 * instruction, in addition to explicit float stores. */
#include "gm.h"
#include "tmnf_fp.h"

TMNF_HD void GmMat3_SetFromQuat(GmMat3 *self, float qx, float qy, float qz, float qw) {
	float two_y = x87_r24(F(qy) * 2.0);
	float two_z = x87_r24(F(qz) * 2.0);
	float two_w = x87_r24(F(qw) * 2.0);
	float xy2 = x87_mul(qx, two_y);
	float xz2 = x87_mul(qx, two_z);
	float xw2 = x87_mul(qx, two_w);
	float yy2 = x87_mul(two_y, qy);
	float yz2 = x87_mul(qy, two_z);
	float yw2 = x87_mul(qy, two_w);
	float zz2 = x87_mul(two_z, qz);
	float zw2 = x87_mul(qz, two_w);
	float ww2 = x87_mul(two_w, qw);

	self->m[0] = x87_sub(x87_sub(1.0f, zz2), ww2);
	self->m[3] = x87_add(yz2, xw2);
	self->m[6] = x87_sub(yw2, xz2);
	self->m[1] = x87_sub(yz2, xw2);
	self->m[4] = x87_sub(x87_sub(1.0f, yy2), ww2);
	self->m[7] = x87_add(zw2, xy2);
	self->m[2] = x87_add(yw2, xz2);
	self->m[5] = x87_sub(zw2, xy2);
	self->m[8] = x87_sub(x87_sub(1.0f, yy2), zz2);
}

TMNF_HD void GmMat3_SetMult(GmMat3 *self, const GmMat3 *A, const GmMat3 *B) {
	GmMat3_Set(self, A);
	GmMat3_Mult(self, B);
}

TMNF_HD void GmMat3_MultTranspose(GmMat3 *self, const GmMat3 *M) {
	float s[9];
	for (int i = 0; i < 9; i++) {
		s[i] = self->m[i];
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			self->m[i * 3 + j] = x87_dot3_left(
				M->m[i], s[j], M->m[3 + i], s[3 + j],
				M->m[6 + i], s[6 + j]);
		}
	}
}

TMNF_HD void GmQuat_SetFromMat3(GmQuat *self, const GmMat3 *M) {
	const int NEXT[3] = { 1, 2, 0 }; /* DAT_00d1a86c */
	const float *m = M->m;
	float *q = &self->x;
	float trace = x87_add(x87_add(m[0], m[4]), m[8]);
	if (0.0f < trace) {
		float s = x87_sqrt(x87_r24(F(trace) + 1.0));
		float f = x87_r24(0.5 / F(s));
		q[0] = x87_r24(F(s) * 0.5);
		q[1] = x87_mul(x87_sub(m[7], m[5]), f);
		q[2] = x87_mul(x87_sub(m[2], m[6]), f);
		q[3] = x87_mul(x87_sub(m[3], m[1]), f);
		return;
	}
	int i = m[0] < m[4] ? 1 : 0;
	if (m[i * 4] < m[8]) {
		i = 2;
	}
	int j = NEXT[i];
	int k = NEXT[j];
	float s = x87_sqrt(x87_r24(
		F(x87_sub(m[i * 4], x87_add(m[k * 4], m[j * 4]))) + 1.0));
	float f = x87_r24(0.5 / F(s));
	q[i + 1] = x87_r24(F(s) * 0.5);
	q[0] = x87_mul(x87_sub(m[3 * k + j], m[3 * j + k]), f);
	q[j + 1] = x87_mul(x87_add(m[3 * i + j], m[3 * j + i]), f);
	q[k + 1] = x87_mul(x87_add(m[3 * i + k], m[3 * k + i]), f);
}

TMNF_HD void GmQuat_Normalize(GmQuat *self) {
	float sumsq = x87_add(x87_add(x87_add(x87_mul(self->y, self->y),
		x87_mul(self->x, self->x)), x87_mul(self->z, self->z)),
		x87_mul(self->w, self->w));
	float s = x87_sqrt(sumsq);
	float inv = x87_rcp(s);
	self->x = x87_mul(self->x, inv);
	self->y = x87_mul(self->y, inv);
	self->z = x87_mul(self->z, inv);
	self->w = x87_mul(inv, self->w);
}

TMNF_HD static void normalize_vec3_if_nonzero(GmVec3 *value) {
	float length_sq = x87_add(
		x87_add(
			x87_mul(value->y, value->y),
			x87_mul(value->x, value->x)),
		x87_mul(value->z, value->z));
	if (length_sq > 9.999999439624929e-11f) {
		float length = x87_sqrt(length_sq);
		float inverse = x87_rcp(length);
		value->x = x87_mul(inverse, value->x);
		value->y = x87_mul(value->y, inverse);
		value->z = x87_mul(inverse, value->z);
	}
}

/*
 * 0x009C1C40 __CIacos. DAT_00D7AFF8 is zero in the live image, so the
 * dispatcher takes the x87 path at 0x009C1CB3..0x009C1CC1.
 */
TMNF_HD float GmFunc_Acos(float value) {
	float product = x87_mul(
		x87_add(1.0f, value),
		x87_sub(1.0f, value));
	float root = x87_sqrt(product);
	return x87_atan2(root, value);
}

/* 0x008E7DC0. Signed angle around the global Y axis. UNVALIDATED. */
TMNF_HD float GmVec3_GetAngle(const GmVec3 *a, const GmVec3 *b) {
	float dot = x87_add(
		x87_add(x87_mul(a->x, b->x), x87_mul(a->y, b->y)),
		x87_mul(a->z, b->z));
	float guarded_dot = x87_r24(F(dot) * 0.9999900000002526);
	float angle = GmFunc_Acos(guarded_dot);
	if (angle > 9.999999747378752e-06f) {
		GmVec3 an = *a;
		GmVec3 bn = *b;
		normalize_vec3_if_nonzero(&an);
		normalize_vec3_if_nonzero(&bn);

		float cross_z = x87_sub(
			x87_mul(bn.y, an.x), x87_mul(an.y, bn.x));
		float cross_y = x87_sub(
			x87_mul(an.z, bn.x), x87_mul(an.x, bn.z));
		float cross_x = x87_sub(
			x87_mul(bn.z, an.y), x87_mul(an.z, bn.y));
		float sign_value = x87_add(
			x87_add(x87_mul(cross_z, 0.0f), cross_y),
			x87_mul(cross_x, 0.0f));
		if (sign_value < 0.0f) {
			angle = -angle;
		}
	}
	return angle;
}

/* 0x008E7FD0. */
TMNF_HD int GmFunc_IsANumber(float value) {
	return !isnan(F(value));
}

/* 0x00457540. Uses the float sign bit directly, including signed zero/NaN. */
TMNF_HD float GmFunc_Sign(float value) {
	union {
		float f;
		unsigned int u;
	} bits = { .f = value };
	return (bits.u & 0x80000000u) != 0 ? -1.0f : 1.0f;
}

/*
 * 0x004575B0. The clamp precedes the 0x009C1D90 __CIasin dispatcher.
 * DAT_00D7AFF8 is zero in the live image, so the dispatcher takes the x87
 * path at 0x009C1E03..0x009C1E0F.
 */
TMNF_HD float GmFunc_AsinSafe(float value) {
	float product;
	float root;

	if (value < -0.999999f) {
		return -1.5707964f;
	}
	if (value > 0.999999f) {
		return 1.5707964f;
	}
	product = x87_mul(
		x87_add(1.0f, value),
		x87_sub(1.0f, value));
	root = x87_sqrt(product);
	return x87_atan2(value, root);
}

/* 0x00575440. */
TMNF_HD float GmFunc_Mod(float value, float lower, float upper) {
	if (lower < value && value < upper) {
		return value;
	}
	float period = x87_sub(upper, lower);
	float relative = x87_sub(value, lower);
	float result = x87_r24(fmod(F(relative), F(period)));
	if (result < 0.0f) {
		result = x87_add(result, period);
	}
	return x87_add(result, lower);
}
