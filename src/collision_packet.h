#ifndef TMNF_COLLISION_PACKET_H
#define TMNF_COLLISION_PACKET_H

#include "collision.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Host ISA dispatch; emitted contact bits retain the original face order.
 * Output contacts are in scaled ellipsoid space; the caller sets materials
 * and applies the existing exact output transforms. */
uint32_t TmnfCollision_PacketWidth(void);
uint32_t TmnfCollision_FaceContacts(const GmIso4 *inverse, const TmnfSphereFaceEdges *faces,
                                    const uint32_t *indices, uint32_t count, GmCollision *out);

#ifdef __cplusplus
}
#endif
#endif
