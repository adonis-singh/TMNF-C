/* Real-game replay 6924773, wheel contact at 44420 ms. Summing z+x+y
 * instead of y+x+z rounds the dot product one ulp above cos(pi/6),
 * incorrectly merging away a contact the game retains. */
#include <assert.h>
#include <string.h>

#include "collision.h"

static void check_contact(GmVec3 normal, uint32_t expected_count)
{
	SHmsSphereBufferContact source;
	CHmsCollisionBuffer destination;
	SHmsSphereBufferContact_Init(&source);
	CHmsCollisionBuffer_Init(&destination);
	SHmsPhysicalCollision *base = CFastBuffer_SHmsPhysicalCollision_AddNewElem(
		&source.base.collisions);
	memset(base, 0, sizeof(*base));
	base->collision.normal = (GmVec3){
		0.2033737301826477f, 0.9776391983032227f, -0.05348503217101097f};
	base->collision.face_normal = (GmVec3){1, 0, 0};
	base->collision.flags = 1;
	SHmsPhysicalCollision *feature = CFastBuffer_SHmsPhysicalCollision_AddNewElem(
		&source.base.collisions);
	memset(feature, 0, sizeof(*feature));
	feature->collision.normal = normal;
	feature->collision.face_normal = (GmVec3){0, 1, 0};
	SHmsSphereBufferContact_MergeAndAddToCollisions(&source, &destination);
	assert(destination.collisions.count == expected_count);
	if (expected_count == 2)
		assert(memcmp(&destination.collisions.data[1].collision.normal,
			&normal, sizeof(normal)) == 0);
	assert(source.base.collisions.count == 0);
	CHmsCollisionBuffer_Destroy(&source.base);
	CHmsCollisionBuffer_Destroy(&destination);
}

int main(void)
{
	check_contact((GmVec3){
		0.18157334625720978f, 0.8728424906730652f, 0.4529648721218109f}, 2);
	/* A contact clearly inside the angle threshold still merges. */
	check_contact((GmVec3){
		0.2033737301826477f, 0.9776391983032227f, -0.05348503217101097f}, 1);
	check_contact((GmVec3){0, 0, 1}, 2);
	return 0;
}
