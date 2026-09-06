#ifndef TMNF_COLLISION_RESPONSE_H
#define TMNF_COLLISION_RESPONSE_H

#include "tmnf_hd.h"
#include <stddef.h>
#include <stdint.h>

#include "collision.h"
#include "hms_dyna.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	float friction;
	float restitution;
} CPlugSurfaceMaterialData;

typedef struct {
	uint32_t category;
	uint32_t response_mode;
	uint32_t side_enabled[2];
} CHmsResponseMaterial;

typedef struct CHmsResponseBody CHmsResponseBody;

typedef struct {
	CHmsResponseBody *body;
	uint32_t tree_ref;
	uint16_t surface_material;
	uint16_t reserved0a;
	GmVec3 normal;
	GmVec3 position;
	GmVec3 relative_speed;
	GmVec3 replacement;
	uint32_t accepted;
	CHmsResponseBody *other_body;
	uint32_t other_tree_ref;
	uint16_t other_surface_material;
	uint16_t reserved_other;
} CHmsPhysicalContact;

typedef void (*CHmsAbsorbContactFn)(
	void *user, CHmsResponseBody *body, CHmsPhysicalContact *contact);

struct CHmsResponseBody {
	uint32_t corpus_ref;
	uint32_t classification_flags;
	uint32_t response_flags;
	float response_weight;
	GmIso4 iso;
	CHmsDyna *dyna;
	uint32_t has_contact_sink;
	CHmsAbsorbContactFn absorb_contact;
	void *contact_user;
};

typedef CHmsResponseBody *(*CHmsResolveResponseBodyFn)(
	void *user, uint32_t corpus_ref);
typedef const CHmsResponseMaterial *(*CHmsResolveResponseMaterialFn)(
	void *user, uint32_t material_ref);

typedef struct {
	CFastBuffer_SHmsPhysicalCollision *collisions;
	const CPlugSurfaceMaterialData *surface_materials;
	uint32_t surface_material_count;
	CHmsResolveResponseBodyFn resolve_body;
	CHmsResolveResponseMaterialFn resolve_material;
	void *resolver_user;
} CHmsResponseZone;

/* Exact 32-bit game layouts and accessed prefixes. */
typedef struct {
	float friction;
	float restitution;
} CPlugSurfaceMaterialDataLayout32;

typedef struct {
	uint32_t category;
	uint32_t reserved04;
	uint32_t response_mode;
	uint32_t side_enabled[2];
} CHmsResponseMaterialLayout32;

typedef struct {
	uint32_t corpus;
	uint32_t tree;
	uint16_t surface_material;
	uint16_t reserved0a;
	GmVec3 normal;
	GmVec3 position;
	GmVec3 relative_speed;
	GmVec3 replacement;
	uint32_t accepted;
	uint32_t other_corpus;
	uint32_t other_tree;
	uint16_t other_surface_material;
	uint16_t reserved4a;
} CHmsPhysicalContactLayout32;

typedef struct {
	uint8_t reserved00[0x18];
	float response_weight;
} CHmsResponseModelPrefixLayout32;

typedef struct {
	uint8_t reserved00[0x14];
	uint32_t response_model;
	uint32_t classification_flags;
	uint32_t response_flags;
	uint8_t reserved20[0x04];
	uint32_t contact_sink_ref;
} CHmsItemResponsePrefixLayout32;

typedef struct {
	uint8_t reserved00[0x08];
	uint32_t contact_sink;
} CHmsContactSinkRefLayout32;

typedef struct {
	uint32_t slot0;
	uint32_t slot1;
	uint32_t slot2;
	uint32_t absorb_contact;
} CHmsContactSinkVTablePrefixLayout32;

typedef struct {
	uint8_t reserved00[0x108];
	uint32_t params;
	uint8_t reserved10c[0x220];
	uint32_t live_state;
	uint8_t reserved330[0x10];
	int32_t mode;
} CHmsDynaResponsePrefixLayout32;

typedef struct {
	uint8_t reserved00[0x158];
	CHmsCollisionBufferLayout32 collision_buffer;
	uint32_t collision_manager;
} CHmsZoneDynamicResponsePrefixLayout32;

_Static_assert(sizeof(CPlugSurfaceMaterialDataLayout32) == 0x08,
	"CPlugSurfaceMaterialData size");
_Static_assert(offsetof(CPlugSurfaceMaterialDataLayout32, friction) == 0x00,
	"surface material friction");
_Static_assert(offsetof(CPlugSurfaceMaterialDataLayout32, restitution) == 0x04,
	"surface material restitution");
_Static_assert(sizeof(CHmsResponseMaterialLayout32) == 0x14,
	"response material accessed size");
_Static_assert(offsetof(CHmsResponseMaterialLayout32, category) == 0x00,
	"response material category");
_Static_assert(offsetof(CHmsResponseMaterialLayout32, response_mode) == 0x08,
	"response material mode");
_Static_assert(offsetof(CHmsResponseMaterialLayout32, side_enabled) == 0x0c,
	"response material side flags");
_Static_assert(sizeof(CHmsPhysicalContactLayout32) == 0x4c,
	"CHmsPhysicalContact size");
_Static_assert(offsetof(CHmsPhysicalContactLayout32, corpus) == 0x00,
	"contact corpus");
_Static_assert(offsetof(CHmsPhysicalContactLayout32, tree) == 0x04,
	"contact tree");
_Static_assert(offsetof(CHmsPhysicalContactLayout32, surface_material) == 0x08,
	"contact material");
_Static_assert(offsetof(CHmsPhysicalContactLayout32, normal) == 0x0c,
	"contact normal");
_Static_assert(offsetof(CHmsPhysicalContactLayout32, position) == 0x18,
	"contact position");
_Static_assert(offsetof(CHmsPhysicalContactLayout32, relative_speed) == 0x24,
	"contact speed");
_Static_assert(offsetof(CHmsPhysicalContactLayout32, replacement) == 0x30,
	"contact replacement");
_Static_assert(offsetof(CHmsPhysicalContactLayout32, accepted) == 0x3c,
	"contact accepted");
_Static_assert(offsetof(CHmsPhysicalContactLayout32, other_corpus) == 0x40,
	"contact other corpus");
_Static_assert(offsetof(CHmsPhysicalContactLayout32, other_tree) == 0x44,
	"contact other tree");
_Static_assert(offsetof(CHmsPhysicalContactLayout32, other_surface_material) == 0x48,
	"contact other material");
_Static_assert(sizeof(CHmsResponseModelPrefixLayout32) == 0x1c,
	"response model prefix size");
_Static_assert(offsetof(CHmsResponseModelPrefixLayout32, response_weight) == 0x18,
	"response model weight");
_Static_assert(sizeof(CHmsItemResponsePrefixLayout32) == 0x28,
	"item response prefix size");
_Static_assert(offsetof(CHmsItemResponsePrefixLayout32, response_model) == 0x14,
	"item response model");
_Static_assert(offsetof(CHmsItemResponsePrefixLayout32, classification_flags) == 0x18,
	"item classification flags");
_Static_assert(offsetof(CHmsItemResponsePrefixLayout32, response_flags) == 0x1c,
	"item response flags");
_Static_assert(offsetof(CHmsItemResponsePrefixLayout32, contact_sink_ref) == 0x24,
	"item contact sink ref");
_Static_assert(sizeof(CHmsContactSinkRefLayout32) == 0x0c,
	"contact sink ref prefix size");
_Static_assert(offsetof(CHmsContactSinkRefLayout32, contact_sink) == 0x08,
	"contact sink pointer");
_Static_assert(sizeof(CHmsContactSinkVTablePrefixLayout32) == 0x10,
	"contact sink vtable prefix size");
_Static_assert(offsetof(CHmsContactSinkVTablePrefixLayout32, absorb_contact) == 0x0c,
	"AbsorbContact vtable slot");
_Static_assert(sizeof(CHmsDynaResponsePrefixLayout32) == 0x344,
	"CHmsDyna response prefix size");
_Static_assert(offsetof(CHmsDynaResponsePrefixLayout32, params) == 0x108,
	"CHmsDyna params");
_Static_assert(offsetof(CHmsDynaResponsePrefixLayout32, live_state) == 0x32c,
	"CHmsDyna live state");
_Static_assert(offsetof(CHmsDynaResponsePrefixLayout32, mode) == 0x340,
	"CHmsDyna mode");
_Static_assert(sizeof(CHmsZoneDynamicResponsePrefixLayout32) == 0x16c,
	"CHmsZoneDynamic response prefix size");
_Static_assert(offsetof(CHmsZoneDynamicResponsePrefixLayout32, collision_buffer) == 0x158,
	"zone collision buffer");
_Static_assert(offsetof(CHmsZoneDynamicResponsePrefixLayout32, collision_manager) == 0x168,
	"zone collision manager");
_Static_assert(offsetof(SHmsPhysicalCollision, corpus1) == 0x00,
	"physical collision corpus1");
_Static_assert(offsetof(SHmsPhysicalCollision, tree1) == 0x04,
	"physical collision tree1");
_Static_assert(offsetof(SHmsPhysicalCollision, corpus2) == 0x08,
	"physical collision corpus2");
_Static_assert(offsetof(SHmsPhysicalCollision, tree2) == 0x0c,
	"physical collision tree2");

/* 0x00547E00 */
TMNF_HD void GmCollision_Neg(GmCollision *collision);

/* 0x0087D9A0 */
TMNF_HD float CPlugSurfaceMaterialData_GetRestitutionCoefWith(
	const CPlugSurfaceMaterialData *self,
	const CPlugSurfaceMaterialData *other);

/* 0x00548BF0 */
TMNF_HD void CHmsZoneDynamic_SolveImpulse(
	CHmsResponseZone *zone, SHmsPhysicalCollision *collision,
	CHmsPhysicalContact *contact1, CHmsPhysicalContact *contact2);

/* 0x005497C0 */
TMNF_HD void CHmsZoneDynamic_ComputeCollisionResponse(CHmsResponseZone *zone);

#ifdef __cplusplus
}
#endif

#endif
