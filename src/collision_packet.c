#include "collision_packet.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

#define TMNF_PACKET_TARGET __attribute__((target("avx2")))
#define TMNF_PACKET_NAME contacts_avx2
#define TMNF_PACKET_WIDTH 8
#define TMNF_PACKET_VECTOR __m256
#define TMNF_PACKET_INDEX __m256i
#define TMNF_PACKET_INDEX_LOAD(p) _mm256_loadu_si256((const __m256i *)(p))
#define TMNF_PACKET_GATHER(p, i) _mm256_i32gather_ps((p), (i), 4)
#define PADD _mm256_add_ps
#define PSUB _mm256_sub_ps
#define PMUL _mm256_mul_ps
#define PDIV _mm256_div_ps
#define PSQRT _mm256_sqrt_ps
#define PSET _mm256_set1_ps
#define PSTORE _mm256_storeu_ps
#define PGT(a, b) ((uint32_t)_mm256_movemask_ps(_mm256_cmp_ps((a), (b), _CMP_GT_OQ)))
#define PLE(a, b) ((uint32_t)_mm256_movemask_ps(_mm256_cmp_ps((a), (b), _CMP_LE_OQ)))
#define PBLEND(mask, a, b) \
	_mm256_blendv_ps((a), (b), \
	                 _mm256_castsi256_ps(_mm256_cmpeq_epi32( \
	                     _mm256_and_si256(_mm256_set1_epi32(mask), \
	                                      _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128)), \
	                     _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128))))
#define PNEG(a) _mm256_xor_ps((a), _mm256_set1_ps(-0.0f))
#include "collision_packet_impl.h"

#define TMNF_PACKET_TARGET __attribute__((target("avx512f,avx512dq")))
#define TMNF_PACKET_NAME contacts_avx512
#define TMNF_PACKET_WIDTH 16
#define TMNF_PACKET_VECTOR __m512
#define TMNF_PACKET_INDEX __m512i
#define TMNF_PACKET_INDEX_LOAD(p) _mm512_loadu_si512((const void *)(p))
#define TMNF_PACKET_GATHER(p, i) _mm512_i32gather_ps((i), (p), 4)
#define PADD _mm512_add_ps
#define PSUB _mm512_sub_ps
#define PMUL _mm512_mul_ps
#define PDIV _mm512_div_ps
#define PSQRT _mm512_sqrt_ps
#define PSET _mm512_set1_ps
#define PSTORE _mm512_storeu_ps
#define PGT(a, b) ((uint32_t)_mm512_cmp_ps_mask((a), (b), _CMP_GT_OQ))
#define PLE(a, b) ((uint32_t)_mm512_cmp_ps_mask((a), (b), _CMP_LE_OQ))
#define PBLEND(mask, a, b) _mm512_mask_blend_ps((__mmask16)(mask), (a), (b))
#define PNEG(a) _mm512_xor_ps((a), _mm512_set1_ps(-0.0f))
#include "collision_packet_impl.h"
#endif

uint32_t TmnfCollision_PacketWidth(void)
{
	static atomic_int selected = ATOMIC_VAR_INIT(-1);
	int width = atomic_load_explicit(&selected, memory_order_relaxed);
	if (width >= 0)
		return (uint32_t)width;
	width = 0;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
	if (__builtin_cpu_supports("avx2"))
		width = 8;
	if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq"))
		width = 16;
#endif
	/* Diagnostic override permits paired ISA measurements and scalar parity
	 * checks without rebuilding the rest of the engine. */
	const char *setting = getenv("TMNF_COLLISION_PACKET");
	if (setting != NULL && strcmp(setting, "auto") != 0) {
		if (strcmp(setting, "scalar") == 0)
			width = 0;
		else if (strcmp(setting, "avx2") == 0 && width >= 8)
			width = 8;
		else if (!(strcmp(setting, "avx512") == 0 && width == 16))
			tmnf_fail("unsupported TMNF_COLLISION_PACKET setting");
	}
	atomic_store_explicit(&selected, width, memory_order_relaxed);
	return (uint32_t)width;
}

uint32_t TmnfCollision_FaceContacts(const GmIso4 *inverse, const TmnfSphereFaceEdges *faces,
                                    const uint32_t *indices, uint32_t count, GmCollision *out)
{
	uint32_t width = TmnfCollision_PacketWidth();
	if (count == 0 || count > width)
		tmnf_fail("invalid face packet size");
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
	if (width == 16)
		return contacts_avx512(inverse, faces, indices, count, out);
	if (width == 8)
		return contacts_avx2(inverse, faces, indices, count, out);
#else
	(void)inverse;
	(void)faces;
	(void)indices;
	(void)out;
#endif
	tmnf_fail("face packet ISA unavailable");
}
