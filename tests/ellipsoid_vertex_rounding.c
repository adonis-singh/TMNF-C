/* Reduced from real-game replay 7966129, collision call at 8330 ms.
 * The end vertex has squared distance 1 + 1 ulp. The game rounds its
 * square root to 1 before testing, so this contact must be emitted. */
#include <assert.h>
#include <string.h>

#include "collision.h"
#include "collision_packet.h"

int main(void)
{
	GmVec3 vertices[3] = {
		{-0x1.1b68p+1f, -0x1.8b21p-1f, -0x1.3a6dcp+3f},
		{-0x1.6f4p-5f, -0x1.ec4a8p-1f, -0x1.159cp-2f},
		{0x1.e72p-3f, -0x1.0bbecp+0f, -0x1.298p-2f},
	};
	GmSurfMeshFace face = {.vertex = {0, 1, 2}};
	GmSurfMeshNode node = {
		.skip_count = 1, .box = {.half_extent = {20, 20, 20}},
		.face_index = 0,
	};
	GmSurfMesh mesh = {
		.base = {.type = GM_SURF_MESH},
		.vertex_count = 3, .vertices = vertices,
		.face_count = 1, .faces = &face, .node_count = 1, .nodes = &node,
	};
	GmSurfEllipsoid ellipsoid = {
		.base = {.type = GM_SURF_ELLIPSOID}, .radii = {1, 1, 1},
	};
	GmIso4 identity = {.m = {1, 0, 0, 0, 1, 0, 0, 0, 1}};
	LocatedGmSurf moving = {.surf = &ellipsoid.base, .iso = &identity};
	LocatedGmSurf fixed = {.surf = &mesh.base, .iso = &identity};
	CHmsCollisionBuffer buffer;
	CHmsCollisionBuffer_Init(&buffer);
	assert(GmCollision_Ellipsoid_Mesh(&moving, &fixed, &buffer) == 1);
	assert(buffer.collisions.count == 1);
	const GmCollision *contact = CHmsCollisionBuffer_GetCollision(&buffer, 0);
	assert(memcmp(&contact->position, &vertices[1], sizeof(GmVec3)) == 0);
	assert(contact->flags == 0);
	GmCollision expected = *contact;

	/* A cyclic rotation preserves winding but selects the start-vertex
	 * branch, which tests the squared distance and rejects this point. */
	face.vertex[0] = 1; face.vertex[1] = 2; face.vertex[2] = 0;
	buffer.collisions.count = 0;
	assert(GmCollision_Ellipsoid_Mesh(&moving, &fixed, &buffer) == 0);
	assert(buffer.collisions.count == 0);

	if (TmnfCollision_PacketWidth()) {
		TmnfSphereFaceEdges cached[2] = {0};
		for (int variant = 0; variant < 2; ++variant)
			for (int v = 0; v < 4; ++v)
				for (int axis = 0; axis < 3; ++axis)
					cached[variant].xyz[axis][v] =
						((float *)&vertices[(v % 3 + variant) % 3])[axis];
		uint32_t indices[2] = {0, 1};
		GmCollision contacts[2] = {0};
		assert(TmnfCollision_FaceContacts(&identity, cached, indices, 2, contacts) == 1);
		assert(memcmp(&contacts[0].position, &expected.position, sizeof(GmVec3)) == 0);
		assert(memcmp(&contacts[0].normal, &expected.normal, sizeof(GmVec3)) == 0);
		assert(memcmp(&contacts[0].face_normal, &expected.face_normal, sizeof(GmVec3)) == 0);
		assert(contacts[0].flags == expected.flags);
		/* The ordinary path applies an identity output transform, which
		 * may change the sign of a zero separation component. */
		assert(contacts[0].separation.x == expected.separation.x);
		assert(contacts[0].separation.y == expected.separation.y);
		assert(contacts[0].separation.z == expected.separation.z);
	}
	CHmsCollisionBuffer_Destroy(&buffer);
	return 0;
}
