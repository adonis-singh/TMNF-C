/* TMNF geometry primitives. Layouts mirror the game's memory exactly so that
 * struct pointers can alias captured state snapshots byte-for-byte. */
#ifndef TMNF_GM_H
#define TMNF_GM_H

#include <stddef.h>

#include "tmnf_fp.h"
#include "tmnf_hd.h"

#if TMNF_SSE
#include <emmintrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* 12 bytes. */
typedef struct { float x, y, z; } GmVec3;
/* 16 bytes. Quaternion stores (x,y,z,w) in this order. */
typedef struct { float x, y, z, w; } GmVec4;
typedef GmVec4 GmQuat;
/* 36 bytes, row-major: rows {m[0..2]},{m[3..5]},{m[6..8]}. */
typedef struct { float m[9]; } GmMat3;
/* 48 bytes: row-major 3x3 (m[0..8]) then translation t[0..2]. */
typedef struct { float m[9]; float t[3]; } GmIso4;

_Static_assert(sizeof(GmVec3) == 12, "GmVec3 size");
_Static_assert(sizeof(GmVec4) == 16, "GmVec4 size");
_Static_assert(sizeof(GmMat3) == 36, "GmMat3 size");
_Static_assert(sizeof(GmIso4) == 48, "GmIso4 size");

/* The leaf linear-algebra ops live here as static inline so that callers in
 * other translation units get them inlined (they sit on the collision hot
 * path). Transcribed from the 2.11.26 disassembly; golden traces show that
 * Direct3D leaves the live game's x87 precision control at 24 bits, so
 * rounding occurs after every arithmetic instruction. */
TMNF_HD static inline float x87_dot3_left(float a0, float b0, float a1, float b1,
	float a2, float b2) {
	return x87_add(x87_add(x87_mul(a0, b0), x87_mul(a1, b1)),
		x87_mul(a2, b2));
}

#if TMNF_SSE
/* Packed forms. Every lane performs the same IEEE binary32 multiply and add
 * as the scalar transcription, in the same association: each output
 * component is ((p_x + p_y) + p_z) where p_x/p_y are the two products the
 * scalar code adds first (their order is irrelevant since addition is
 * commutative) and p_z the one it adds last. Loads never read outside the
 * matrix: the third row comes from m[5..8] and is shifted down. */

/* Rows of a bare 3x3: r0 = [m0 m1 m2 m3], r1 = [m3 m4 m5 m6], r2 = [m6 m7 m8 m8]. */
static inline void gm_load_rows(
	const float *m, __m128 *r0, __m128 *r1, __m128 *r2) {
	*r0 = _mm_loadu_ps(m);
	*r1 = _mm_loadu_ps(m + 3);
	__m128 tail = _mm_loadu_ps(m + 5);
	*r2 = _mm_shuffle_ps(tail, tail, _MM_SHUFFLE(3, 3, 2, 1));
}

/* A GmIso4 is always moved as its three natural 16-byte chunks
 * A = [m0 m1 m2 m3], B = [m4 m5 m6 m7], C = [m8 t0 t1 t2], so that every
 * vector load of an isometry that was just written (by these stores or by a
 * 48-byte memcpy) is served by exactly one store and forwards. Loads that
 * straddle two stores (rows at m+3, m+5) stall for the store buffer to
 * drain, which costs more than the arithmetic saved. */
typedef struct { __m128 a, b, c; } GmIsoChunks;

static inline GmIsoChunks gm_iso_load(const GmIso4 *iso) {
	GmIsoChunks k;
	k.a = _mm_loadu_ps(iso->m);
	k.b = _mm_loadu_ps(iso->m + 4);
	k.c = _mm_loadu_ps(iso->m + 8);
	return k;
}

/* Rows: r0 = [m0 m1 m2 m3], r1 = [m3 m4 m5 m5], r2 = [m6 m7 m8 m8]. */
static inline void gm_iso_rows(
	GmIsoChunks k, __m128 *r0, __m128 *r1, __m128 *r2) {
	*r0 = k.a;
	__m128 u = _mm_shuffle_ps(k.a, k.b, _MM_SHUFFLE(1, 0, 3, 3));
	*r1 = _mm_shuffle_ps(u, u, _MM_SHUFFLE(3, 3, 2, 1));
	*r2 = _mm_shuffle_ps(k.b, k.c, _MM_SHUFFLE(0, 0, 3, 2));
}

/* Columns: c0 = [m0 m3 m6 m6], c1 = [m1 m4 m7 m7], c2 = [m2 m5 m8 m8]. */
static inline void gm_iso_cols(
	GmIsoChunks k, __m128 *c0, __m128 *c1, __m128 *c2) {
	*c0 = _mm_shuffle_ps(k.a, k.b, _MM_SHUFFLE(2, 2, 3, 0));
	__m128 u1 = _mm_shuffle_ps(k.a, k.b, _MM_SHUFFLE(3, 0, 1, 1));
	*c1 = _mm_shuffle_ps(u1, u1, _MM_SHUFFLE(3, 3, 2, 0));
	__m128 u2 = _mm_shuffle_ps(k.a, k.b, _MM_SHUFFLE(1, 1, 2, 2));
	*c2 = _mm_shuffle_ps(u2, k.c, _MM_SHUFFLE(0, 0, 2, 0));
}

/* Translation [t0 t1 t2 t2]. */
static inline __m128 gm_iso_translation(GmIsoChunks k) {
	return _mm_shuffle_ps(k.c, k.c, _MM_SHUFFLE(3, 3, 2, 1));
}


#define gm_lane(v, i) _mm_shuffle_ps((v), (v), _MM_SHUFFLE((i), (i), (i), (i)))

/* (a*x + b*y) + c*z with x, y, z already broadcast. */
static inline __m128 gm_combine3v(__m128 a, __m128 x, __m128 b, __m128 y,
	__m128 c, __m128 z) {
	return _mm_add_ps(_mm_add_ps(_mm_mul_ps(a, x), _mm_mul_ps(b, y)),
		_mm_mul_ps(c, z));
}

/* (a*x + b*y) + c*z with x, y, z broadcast. */
static inline __m128 gm_combine3(__m128 a, float x, __m128 b, float y,
	__m128 c, float z) {
	return _mm_add_ps(
		_mm_add_ps(_mm_mul_ps(a, _mm_set1_ps(x)),
			_mm_mul_ps(b, _mm_set1_ps(y))),
		_mm_mul_ps(c, _mm_set1_ps(z)));
}

/* Writes lanes 0..2 only. */
static inline void gm_store3(float *out, __m128 v) {
	_mm_storel_pi((__m64 *)out, v);
	_mm_store_ss(out + 2, _mm_movehl_ps(v, v));
}
#endif

/* 0x0045BBA0  out = A * v + A.translation  (A is GmIso4) */
TMNF_HD static inline void GmVec3_SetMult_Iso4(
	GmVec3 *out, const GmVec3 *v, const GmIso4 *A) {
	out->x = x87_add(x87_dot3_left(A->m[1], v->y, v->x, A->m[0],
		A->m[2], v->z), A->t[0]);
	out->y = x87_add(x87_dot3_left(A->m[3], v->x, A->m[4], v->y,
		A->m[5], v->z), A->t[1]);
	out->z = x87_add(x87_dot3_left(A->m[6], v->x, A->m[7], v->y,
		A->m[8], v->z), A->t[2]);
}

/* 0x004574A0  out = M * v  (M is GmMat3, no translation) */
TMNF_HD static inline void GmVec3_SetMult_Mat3(
	GmVec3 *out, const GmVec3 *v, const GmMat3 *M) {
	out->x = x87_dot3_left(M->m[1], v->y, v->x, M->m[0],
		M->m[2], v->z);
	out->y = x87_dot3_left(M->m[3], v->x, M->m[4], v->y,
		M->m[5], v->z);
	out->z = x87_dot3_left(M->m[6], v->x, M->m[7], v->y,
		M->m[8], v->z);
}

/* 0x0045BC60  this = A * this + A.translation */
TMNF_HD static inline void GmVec3_Mult_Iso4(GmVec3 *self, const GmIso4 *A) {
	float x = self->x, y = self->y, z = self->z;
	self->x = x87_add(x87_add(x87_mul(A->m[2], z),
		x87_add(x87_mul(A->m[0], x), x87_mul(A->m[1], y))), A->t[0]);
	self->y = x87_add(x87_dot3_left(A->m[3], x, A->m[4], y,
		A->m[5], z), A->t[1]);
	self->z = x87_add(x87_add(x87_mul(z, A->m[8]),
		x87_add(x87_mul(y, A->m[7]), x87_mul(x, A->m[6]))), A->t[2]);
}

/* 0x0045BD40  this = M^T * this */
TMNF_HD static inline void GmVec3_MultTranspose(GmVec3 *self, const GmMat3 *M) {
	float x = self->x, y = self->y, z = self->z;
#if TMNF_SSE
	/* out = (row0*x + row1*y) + row2*z; component i is ((m_i x + m_{i+3} y)
	 * + m_{i+6} z), matching the scalar association below. */
	__m128 r0, r1, r2;
	gm_load_rows(M->m, &r0, &r1, &r2);
	gm_store3(&self->x, gm_combine3(r0, x, r1, y, r2, z));
#else
	self->x = x87_add(x87_mul(M->m[6], z),
		x87_add(x87_mul(M->m[0], x), x87_mul(M->m[3], y)));
	self->y = x87_dot3_left(M->m[1], x, M->m[4], y, M->m[7], z);
	self->z = x87_add(x87_mul(z, M->m[8]),
		x87_add(x87_mul(y, M->m[5]), x87_mul(x, M->m[2])));
#endif
}

/* 0x0045BCE0  this = M * this  (M is GmMat3, no translation) */
TMNF_HD static inline void GmVec3_Mult_Mat3(GmVec3 *self, const GmMat3 *M) {
	float x = self->x, y = self->y, z = self->z;
	float out_y = x87_dot3_left(M->m[3], x, M->m[4], y, M->m[5], z);
	float out_z = x87_dot3_left(M->m[6], x, M->m[7], y, M->m[8], z);
	self->x = x87_dot3_left(M->m[1], y, M->m[0], x, M->m[2], z);
	self->y = out_y;
	self->z = out_z;
}

/* 0x008E0640  copy */
TMNF_HD static inline void GmMat3_Set(GmMat3 *self, const GmMat3 *src) {
	for (int i = 0; i < 9; i++) {
		self->m[i] = src->m[i];
	}
}

/* 0x008E0B10  build rotation matrix from quaternion (x,y,z,w) */
TMNF_HD void GmMat3_SetFromQuat(GmMat3 *self, float qx, float qy, float qz, float qw);

/* 0x008E0C60  this = src^T */
TMNF_HD static inline void GmMat3_SetTranspose(GmMat3 *self, const GmMat3 *src) {
	self->m[0] = src->m[0];
	self->m[4] = src->m[4];
	self->m[8] = src->m[8];
	self->m[1] = src->m[3];
	self->m[3] = src->m[1];
	self->m[2] = src->m[6];
	self->m[6] = src->m[2];
	self->m[5] = src->m[7];
	self->m[7] = src->m[5];
}

/* 0x008E09F0  this = this * B */
TMNF_HD static inline void GmMat3_Mult(GmMat3 *self, const GmMat3 *B) {
	/* Kept scalar: a bare GmMat3 has no 16-byte chunking that survives a
	 * round trip through memory, see GmIso4_Mult for the packed form. The
	 * copy is spelled out so the loop vectorizer does not turn it into
	 * 16-byte loads of a matrix that scalar code has just written. */
	const float a[9] = {
		self->m[0], self->m[1], self->m[2],
		self->m[3], self->m[4], self->m[5],
		self->m[6], self->m[7], self->m[8],
	};
	self->m[0] = x87_add(x87_mul(B->m[2], a[6]),
		x87_add(x87_mul(B->m[1], a[3]), x87_mul(B->m[0], a[0])));
	self->m[1] = x87_dot3_left(B->m[1], a[4], B->m[0], a[1],
		B->m[2], a[7]);
	self->m[2] = x87_dot3_left(B->m[0], a[2], B->m[1], a[5],
		B->m[2], a[8]);
	self->m[3] = x87_dot3_left(B->m[4], a[3], a[0], B->m[3],
		B->m[5], a[6]);
	self->m[4] = x87_dot3_left(B->m[4], a[4], B->m[3], a[1],
		B->m[5], a[7]);
	self->m[5] = x87_dot3_left(B->m[4], a[5], B->m[3], a[2],
		B->m[5], a[8]);
	self->m[6] = x87_add(x87_mul(a[6], B->m[8]),
		x87_add(x87_mul(a[0], B->m[6]), x87_mul(a[3], B->m[7])));
	self->m[7] = x87_dot3_left(a[1], B->m[6], a[4], B->m[7],
		B->m[8], a[7]);
	self->m[8] = x87_dot3_left(B->m[6], a[2], B->m[7], a[5],
		B->m[8], a[8]);
}

#if TMNF_SSE
/* Packs rows r0..r2 (lanes 0..2 used) and translation t into chunks. */
static inline GmIsoChunks gm_iso_pack(
	__m128 r0, __m128 r1, __m128 r2, __m128 t) {
	GmIsoChunks k;
	__m128 u = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(0, 0, 2, 2));
	k.a = _mm_shuffle_ps(r0, u, _MM_SHUFFLE(2, 0, 1, 0));
	k.b = _mm_shuffle_ps(r1, r2, _MM_SHUFFLE(1, 0, 2, 1));
	__m128 v = _mm_shuffle_ps(r2, t, _MM_SHUFFLE(0, 0, 2, 2));
	k.c = _mm_shuffle_ps(v, t, _MM_SHUFFLE(2, 1, 2, 0));
	return k;
}

/* diag(sx, sy, sz) with zero translation (GmIso4_SetNUScaleTrans). */
static inline GmIsoChunks gm_iso_scale(float sx, float sy, float sz) {
	GmIsoChunks k;
	k.a = _mm_set_ss(sx);   /* [m0 m1 m2 m3] */
	k.b = _mm_set_ss(sy);   /* [m4 m5 m6 m7] */
	k.c = _mm_set_ss(sz);   /* [m8 t0 t1 t2] */
	return k;
}

/* s * p, translation included: rotation row i = (P[3i]*S_row0 +
 * P[3i+1]*S_row1) + P[3i+2]*S_row2, the association GmMat3_Mult uses for
 * every entry (row-0 and row-1 products first, row-2 last); translation =
 * ((P_col0*tx + P_col1*ty) + P_col2*tz) + P_t as in GmVec3_Mult_Iso4. */
static inline GmIsoChunks gm_iso_mult(GmIsoChunks s, GmIsoChunks p) {
	__m128 s0, s1, s2, p0, p1, p2;
	gm_iso_rows(s, &s0, &s1, &s2);
	gm_iso_cols(p, &p0, &p1, &p2);
	__m128 o0 = gm_combine3v(s0, gm_lane(p0, 0), s1, gm_lane(p1, 0),
		s2, gm_lane(p2, 0));
	__m128 o1 = gm_combine3v(s0, gm_lane(p0, 1), s1, gm_lane(p1, 1),
		s2, gm_lane(p2, 1));
	__m128 o2 = gm_combine3v(s0, gm_lane(p0, 2), s1, gm_lane(p1, 2),
		s2, gm_lane(p2, 2));
	__m128 t = _mm_add_ps(
		gm_combine3v(p0, gm_lane(s.c, 1), p1, gm_lane(s.c, 2),
			p2, gm_lane(s.c, 3)),
		gm_iso_translation(p));
	return gm_iso_pack(o0, o1, o2, t);
}

/* Inverse: rotation = source^T (rows are the source columns); translation =
 * source^T * (-t) = (S_row0*(-tx) + S_row1*(-ty)) + S_row2*(-tz), the
 * association GmVec3_Mult_Mat3 gives per component. */
static inline GmIsoChunks gm_iso_inverse(GmIsoChunks k) {
	__m128 r0, r1, r2, c0, c1, c2;
	gm_iso_rows(k, &r0, &r1, &r2);
	gm_iso_cols(k, &c0, &c1, &c2);
	__m128 neg_t = _mm_xor_ps(gm_iso_translation(k), _mm_set1_ps(-0.0f));
	__m128 t = gm_combine3v(r0, gm_lane(neg_t, 0), r1, gm_lane(neg_t, 1),
		r2, gm_lane(neg_t, 2));
	return gm_iso_pack(c0, c1, c2, t);
}

static inline void gm_iso_store(GmIso4 *out, GmIsoChunks k) {
	_mm_storeu_ps(out->m, k.a);
	_mm_storeu_ps(out->m + 4, k.b);
	_mm_storeu_ps(out->m + 8, k.c);
}

/* Four scalar loads packed into one lane vector. Loads that are narrower than
 * the stores they read forward; 16-byte loads spanning scalar stores stall. */
static inline __m128 gm_gather4(const float *p) {
	return _mm_set_ps(p[3], p[2], p[1], p[0]);
}

static inline GmIsoChunks gm_iso_gather(const GmIso4 *iso) {
	GmIsoChunks k;
	k.a = gm_gather4(iso->m);
	k.b = gm_gather4(iso->m + 4);
	k.c = gm_gather4(iso->m + 8);
	return k;
}
#endif

/* Rewrites an iso that was produced by scalar stores with three 16-byte
 * stores, so later vector loads of it forward instead of stalling. */
TMNF_HD static inline void GmIso4_Relayout(GmIso4 *iso) {
#if TMNF_SSE
	gm_iso_store(iso, gm_iso_gather(iso));
#else
	(void)iso;
#endif
}

/* this = this * parent, translation included: rotation as GmMat3_Mult, then
 * translation = parent * translation + parent.translation (GmVec3_Mult_Iso4). */
TMNF_HD static inline void GmIso4_Mult(GmIso4 *self, const GmIso4 *parent) {
#if TMNF_SSE
	gm_iso_store(self, gm_iso_mult(gm_iso_load(self), gm_iso_load(parent)));
#else
	GmVec3 translation = { self->t[0], self->t[1], self->t[2] };
	GmMat3_Mult((GmMat3 *)self, (const GmMat3 *)parent);
	GmVec3_Mult_Iso4(&translation, parent);
	self->t[0] = translation.x;
	self->t[1] = translation.y;
	self->t[2] = translation.z;
#endif
}

/* 0x008E2570. UNVALIDATED. */
TMNF_HD static inline void GmIso4_SetInverse(GmIso4 *self, const GmIso4 *source) {
#if TMNF_SSE
	gm_iso_store(self, gm_iso_inverse(gm_iso_load(source)));
#else
	GmMat3_SetTranspose((GmMat3 *)self, (const GmMat3 *)source);
	GmVec3 translation = {
		-source->t[0], -source->t[1], -source->t[2],
	};
	GmVec3_Mult_Mat3(&translation, (const GmMat3 *)self);
	self->t[0] = translation.x;
	self->t[1] = translation.y;
	self->t[2] = translation.z;
#endif
}

/* 0x008E08A0  this = A, then Mult(B): the same products and sums. */
TMNF_HD void GmMat3_SetMult(GmMat3 *self, const GmMat3 *A, const GmMat3 *B);
/* 0x008E06B0  this[i][j] = sum_k M[k][i] * this[k][j] */
TMNF_HD void GmMat3_MultTranspose(GmMat3 *self, const GmMat3 *M);
/* 0x008E34C0  quaternion (scalar first, as the dyna state stores it) from a
 * rotation matrix; Shepperd's method with the game's axis successor table. */
TMNF_HD void GmQuat_SetFromMat3(GmQuat *self, const GmMat3 *M);

/* 0x00597780  copy */
TMNF_HD static inline void GmVec4_Set(GmVec4 *self, const GmVec4 *src) {
	self->x = src->x;
	self->y = src->y;
	self->z = src->z;
	self->w = src->w;
}
/* 0x008E3120  normalize quaternion in place */
TMNF_HD void GmQuat_Normalize(GmQuat *self);

/* 0x009C1C40 */ TMNF_HD float GmFunc_Acos(float value);
/* 0x008E7DC0 */ TMNF_HD float GmVec3_GetAngle(
	const GmVec3 *a, const GmVec3 *b);
/* 0x008E7FD0 */ TMNF_HD int GmFunc_IsANumber(float value);
/* 0x00457540 */ TMNF_HD float GmFunc_Sign(float value);
/* 0x004575B0 */ TMNF_HD float GmFunc_AsinSafe(float value);
/* 0x00575440 */ TMNF_HD float GmFunc_Mod(
	float value, float lower, float upper);

#ifdef __cplusplus
}
#endif

#endif /* TMNF_GM_H */
