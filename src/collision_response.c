#include "collision_response.h"

#include <stdlib.h>
#include <string.h>

#include "tmnf_fp.h"

#define RESPONSE_EPSILON 1.0e-5f
#define CENTRAL_IMPULSE_FLAG 0x1000u

TMNF_HD static CHmsResponseBody *resolve_body(
	CHmsResponseZone *zone, uint32_t corpus_ref) {
	if (zone->resolve_body == NULL) {
		tmnf_abort();
	}
	CHmsResponseBody *body =
		zone->resolve_body(zone->resolver_user, corpus_ref);
	if (body == NULL) {
		tmnf_abort();
	}
	return body;
}

TMNF_HD static const CHmsResponseMaterial *resolve_material(
	CHmsResponseZone *zone, uint32_t material_ref) {
	if (zone->resolve_material == NULL) {
		tmnf_abort();
	}
	const CHmsResponseMaterial *material =
		zone->resolve_material(zone->resolver_user, material_ref);
	if (material == NULL) {
		tmnf_abort();
	}
	return material;
}

TMNF_HD static const GmIso4 *body_iso(const CHmsResponseBody *body) {
	if (body->dyna == NULL) {
		return &body->iso;
	}
	if (body->dyna->liveState == NULL) {
		tmnf_abort();
	}
	return (const GmIso4 *)&body->dyna->liveState->rot;
}

/* 0x00533DD0, inlined here to preserve its PC=24 operation schedule. */
TMNF_HD static void get_speed(
	CHmsResponseBody *body, const GmVec3 *position, GmVec3 *speed) {
	if (body->dyna == NULL) {
		speed->x = 0.0f;
		speed->y = 0.0f;
		speed->z = 0.0f;
		return;
	}
	CHmsDyna *dyna = body->dyna;
	if (dyna->liveState == NULL || dyna->params == NULL) {
		tmnf_abort();
	}
	if (dyna->mode == 2) {
		speed->x = 0.0f;
		speed->y = 0.0f;
		speed->z = 0.0f;
		return;
	}
	const CHmsStateDyna *state = dyna->liveState;
	*speed = state->linVel;
	if (dyna->mode == 1) {
		GmVec3 center_of_mass;
		GmVec3_SetMult_Iso4(
			&center_of_mass, &dyna->params->comOffset,
			(const GmIso4 *)&state->rot);
		float radius_x =
			x87_sub(position->x, center_of_mass.x);
		float radius_y =
			x87_sub(position->y, center_of_mass.y);
		float radius_z =
			x87_sub(position->z, center_of_mass.z);
		float cross_x = x87_sub(
			x87_mul(radius_z, state->angVel.y),
			x87_mul(radius_y, state->angVel.z));
		float cross_y = x87_sub(
			x87_mul(radius_x, state->angVel.z),
			x87_mul(radius_z, state->angVel.x));
		float cross_z = x87_sub(
			x87_mul(radius_y, state->angVel.x),
			x87_mul(radius_x, state->angVel.y));
		speed->x = x87_add(speed->x, cross_x);
		speed->y = x87_add(speed->y, cross_y);
		speed->z = x87_add(speed->z, cross_z);
	}
}

TMNF_HD static uint32_t local_contact_mode(const CHmsResponseBody *body) {
	return (body->classification_flags >> 17) & 3u;
}

TMNF_HD static float length_squared(const GmVec3 *vector) {
	return x87_add(
		x87_add(x87_mul(vector->x, vector->x),
			x87_mul(vector->y, vector->y)),
		x87_mul(vector->z, vector->z));
}

TMNF_HD static float dot_collision_normal(
	const GmVec3 *normal, const GmVec3 *vector) {
	return x87_add(
		x87_mul(normal->z, vector->z),
		x87_add(x87_mul(normal->x, vector->x),
			x87_mul(normal->y, vector->y)));
}

TMNF_HD static GmVec3 cross(const GmVec3 *a, const GmVec3 *b) {
	GmVec3 result = {
		x87_sub(x87_mul(a->y, b->z), x87_mul(a->z, b->y)),
		x87_sub(x87_mul(a->z, b->x), x87_mul(a->x, b->z)),
		x87_sub(x87_mul(a->x, b->y), x87_mul(a->y, b->x)),
	};
	return result;
}

TMNF_HD static const CPlugSurfaceMaterialData *surface_material(
	const CHmsResponseZone *zone, uint16_t index) {
	if (index >= zone->surface_material_count ||
		zone->surface_materials == NULL) {
		tmnf_abort();
	}
	return &zone->surface_materials[index];
}

/* 0x00547E00  Negates collision orientation and swaps surface materials.
 * VALIDATED: 2,000 A01 records. */
TMNF_HD void GmCollision_Neg(GmCollision *collision) {
	collision->normal.x = -collision->normal.x;
	collision->normal.y = -collision->normal.y;
	collision->normal.z = -collision->normal.z;
	uint16_t material = collision->material1;
	collision->material1 = collision->material2;
	collision->separation.x = -collision->separation.x;
	collision->material2 = material;
	collision->separation.y = -collision->separation.y;
	collision->separation.z = -collision->separation.z;
	collision->face_normal.x = -collision->face_normal.x;
	collision->face_normal.y = -collision->face_normal.y;
	collision->face_normal.z = -collision->face_normal.z;
}

/* 0x0087D9A0  Combines two material restitution coefficients.
 * UNVALIDATED: no trace exists for this VA. */
TMNF_HD float CPlugSurfaceMaterialData_GetRestitutionCoefWith(
	const CPlugSurfaceMaterialData *self,
	const CPlugSurfaceMaterialData *other) {
	if (self->restitution <= 0.0f) {
		if (0.0f < other->restitution) {
			return self->restitution;
		}
		return x87_add(other->restitution, self->restitution);
	}
	if (0.0f < other->restitution) {
		return x87_mul(other->restitution, self->restitution);
	}
	return other->restitution;
}

TMNF_HD static CHmsPhysicalContact *make_contact(
	CHmsPhysicalContact *contact, CHmsResponseBody *body,
	uint32_t tree_ref, uint16_t surface_material_index,
	CHmsResponseBody *other_body, uint32_t other_tree_ref,
	uint16_t other_surface_material,
	const GmCollision *collision, uint32_t side_enabled) {
	if (side_enabled == 0 || body->has_contact_sink == 0) {
		return NULL;
	}
	if (body->absorb_contact == NULL) {
		tmnf_abort();
	}

	memset(contact, 0, sizeof(*contact));
	contact->body = body;
	contact->tree_ref = tree_ref;
	contact->surface_material = surface_material_index;
	contact->other_body = other_body;
	contact->other_tree_ref = other_tree_ref;
	contact->other_surface_material = other_surface_material;
	if (local_contact_mode(body) > 1) {
		const GmIso4 *iso = body_iso(body);
		contact->normal = collision->normal;
		GmVec3_MultTranspose(&contact->normal, (const GmMat3 *)iso);
		contact->position.x =
			x87_sub(collision->position.x, iso->t[0]);
		contact->position.y =
			x87_sub(collision->position.y, iso->t[1]);
		contact->position.z =
			x87_sub(collision->position.z, iso->t[2]);
		GmVec3_MultTranspose(&contact->position, (const GmMat3 *)iso);
	}
	return contact;
}

TMNF_HD static void absorb_contact(CHmsPhysicalContact *contact) {
	contact->body->absorb_contact(
		contact->body->contact_user, contact->body, contact);
}

TMNF_HD static void solve_body_impulse(
	CHmsResponseBody *body, const GmCollision *collision,
	const GmVec3 *speed, float friction_product, float restitution) {
	if (body->dyna == NULL) {
		return;
	}

	float normal_speed =
		dot_collision_normal(&collision->normal, speed);
	GmVec3 projection = {
		x87_mul(collision->normal.x, normal_speed),
		x87_mul(collision->normal.y, normal_speed),
		x87_mul(normal_speed, collision->normal.z),
	};
	GmVec3 tangent = {
		x87_sub(speed->x, projection.x),
		x87_sub(speed->y, projection.y),
		x87_sub(speed->z, projection.z),
	};

	float projection_length = x87_sqrt(length_squared(&projection));
	float tangent_length = x87_sqrt(length_squared(&tangent));
	float tangent_limit = x87_mul(projection_length, friction_product);
	if (tangent_limit < tangent_length) {
		float scale = x87_div(tangent_limit, tangent_length);
		tangent.x = x87_mul(scale, tangent.x);
		tangent.y = x87_mul(tangent.y, scale);
		tangent.z = x87_mul(scale, tangent.z);
	}

	GmVec3 impulse_direction = {
		-x87_add(tangent.x, projection.x),
		-x87_add(tangent.y, projection.y),
		-x87_add(tangent.z, projection.z),
	};
	float speed_length = x87_sqrt(length_squared(&impulse_direction));
	if (!(RESPONSE_EPSILON < speed_length)) {
		return;
	}

	float inverse_speed = x87_rcp(speed_length);
	impulse_direction.x = x87_mul(inverse_speed, impulse_direction.x);
	impulse_direction.y = x87_mul(impulse_direction.y, inverse_speed);
	impulse_direction.z = x87_mul(inverse_speed, impulse_direction.z);

	CHmsDyna *dyna = body->dyna;
	if (dyna->params == NULL) {
		tmnf_abort();
	}
	float numerator =
		x87_mul(x87_add(restitution, 1.0f), speed_length);
	float denominator = x87_rcp(dyna->params->mass);

	if (dyna->mode == 1 &&
		(body->response_flags & CENTRAL_IMPULSE_FLAG) == 0) {
		const CHmsStateDyna *state = dyna->liveState;
		GmVec3 center_of_mass;
		GmVec3_SetMult_Iso4(
			&center_of_mass, &dyna->params->comOffset,
			(const GmIso4 *)&state->rot);
		GmVec3 radius = {
			x87_sub(collision->position.x, center_of_mass.x),
			x87_sub(collision->position.y, center_of_mass.y),
			x87_sub(collision->position.z, center_of_mass.z),
		};
		GmVec3 radius_cross_normal =
			cross(&radius, &impulse_direction);
		GmVec3 inertia_cross;
		GmVec3_SetMult_Mat3(
			&inertia_cross, &radius_cross_normal,
			&state->invInertiaWorld);
		GmVec3 angular = cross(&inertia_cross, &radius);
		float rotational = x87_add(
			x87_add(
				x87_mul(angular.x, impulse_direction.x),
				x87_mul(angular.y, impulse_direction.y)),
			x87_mul(angular.z, impulse_direction.z));
		denominator = x87_add(rotational, denominator);
	}

	float scale = x87_div(numerator, denominator);
	GmVec3 impulse = {
		x87_mul(impulse_direction.x, scale),
		x87_mul(impulse_direction.y, scale),
		x87_mul(scale, impulse_direction.z),
	};
	if ((body->response_flags & CENTRAL_IMPULSE_FLAG) == 0) {
		CHmsDyna_AddImpulseAt(dyna, &impulse, &collision->position);
	} else {
		CHmsDyna_AddImpulse(dyna, &impulse);
	}
}

TMNF_HD static void compute_replacements(
	const CHmsResponseBody *body1, const CHmsResponseBody *body2,
	const GmVec3 *normal, GmVec3 *replacement1, GmVec3 *replacement2) {
	uint32_t rank1 = (body1->classification_flags >> 11) & 3u;
	uint32_t rank2 = (body2->classification_flags >> 11) & 3u;
	if (rank2 < rank1) {
		replacement1->x = -normal->x;
		replacement1->y = -normal->y;
		replacement1->z = -normal->z;
		replacement2->x = 0.0f;
		replacement2->y = 0.0f;
		replacement2->z = 0.0f;
		return;
	}
	if (rank1 < rank2) {
		*replacement1 = (GmVec3){ 0.0f, 0.0f, 0.0f };
		*replacement2 = *normal;
		return;
	}

	float inverse_total = x87_rcp(x87_add(body2->response_weight, body1->response_weight));
	float scale1 = x87_mul(-body2->response_weight, inverse_total);
	replacement1->x = x87_mul(normal->x, scale1);
	replacement1->y = x87_mul(scale1, normal->y);
	replacement1->z = x87_mul(scale1, normal->z);
	float scale2 = x87_mul(body1->response_weight, inverse_total);
	replacement2->x = x87_mul(normal->x, scale2);
	replacement2->y = x87_mul(scale2, normal->y);
	replacement2->z = x87_mul(scale2, normal->z);
}

TMNF_HD static void prepare_solve_contact(
	CHmsPhysicalContact *contact, const GmVec3 *replacement,
	GmVec3 *relative_speed, int first_body, GmVec3 *replacement_world) {
	contact->accepted = 1;
	const GmMat3 *rotation = (const GmMat3 *)body_iso(contact->body);
	contact->replacement.x = replacement->x;
	contact->replacement.y = replacement->y;
	contact->replacement.z = replacement->z;
	GmVec3_MultTranspose(&contact->replacement, rotation);
	if (first_body) {
		contact->relative_speed.x = -relative_speed->x;
		contact->relative_speed.y = -relative_speed->y;
		contact->relative_speed.z = -relative_speed->z;
	} else {
		contact->relative_speed = *relative_speed;
	}
	GmVec3_MultTranspose(&contact->relative_speed, rotation);
	absorb_contact(contact);
	GmVec3_Mult_Mat3(&contact->relative_speed, rotation);
	if (first_body) {
		relative_speed->x = -contact->relative_speed.x;
		relative_speed->y = -contact->relative_speed.y;
		relative_speed->z = -contact->relative_speed.z;
	}
	GmVec3_SetMult_Mat3(
		replacement_world, &contact->replacement, rotation);
}

/* 0x00548BF0  Resolves replacement, contact callbacks, friction, and impulse.
 * VALIDATED: 2,000 graph-complete A01 records. */
TMNF_HD void CHmsZoneDynamic_SolveImpulse(
	CHmsResponseZone *zone, SHmsPhysicalCollision *physical,
	CHmsPhysicalContact *contact1, CHmsPhysicalContact *contact2) {
	CHmsResponseBody *body1 = resolve_body(zone, physical->corpus1);
	CHmsResponseBody *body2 = resolve_body(zone, physical->corpus2);
	GmCollision *collision = &physical->collision;
	const CPlugSurfaceMaterialData *surface1 =
		surface_material(zone, collision->material1);
	const CPlugSurfaceMaterialData *surface2 =
		surface_material(zone, collision->material2);
	float restitution =
		CPlugSurfaceMaterialData_GetRestitutionCoefWith(surface1, surface2);

	GmVec3 replacement1;
	GmVec3 replacement2;
	compute_replacements(
		body1, body2, &collision->separation, &replacement1, &replacement2);

	GmVec3 speed1;
	GmVec3 speed2;
	get_speed(body1, &collision->position, &speed1);
	get_speed(body2, &collision->position, &speed2);
	GmVec3 relative_speed = {
		x87_sub(speed2.x, speed1.x),
		x87_sub(speed2.y, speed1.y),
		x87_sub(speed2.z, speed1.z),
	};

	if (contact1 != NULL) {
		prepare_solve_contact(
			contact1, &replacement1, &relative_speed, 1, &replacement1);
	}
	if (body1->dyna != NULL) {
		CHmsDyna_AddReplacement(body1->dyna, &replacement1);
	}
	if (contact2 != NULL) {
		prepare_solve_contact(
			contact2, &replacement2, &relative_speed, 0, &replacement2);
	}
	if (body2->dyna != NULL) {
		CHmsDyna_AddReplacement(body2->dyna, &replacement2);
	}

	if ((contact1 != NULL && contact1->accepted == 0) ||
		(contact2 != NULL && contact2->accepted == 0)) {
		return;
	}

	float friction_product =
		x87_mul(surface1->friction, surface2->friction);
	solve_body_impulse(
		body1, collision, &speed1, friction_product, restitution);
	GmCollision_Neg(collision);
	solve_body_impulse(
		body2, collision, &speed2, friction_product, restitution);
}

TMNF_HD static void fill_external_contact_speed(
	CHmsPhysicalContact *contact, const GmVec3 *relative_speed,
	int first_body) {
	if (local_contact_mode(contact->body) > 1) {
		if (first_body) {
			contact->relative_speed.x = -relative_speed->x;
			contact->relative_speed.y = -relative_speed->y;
			contact->relative_speed.z = -relative_speed->z;
		} else {
			contact->relative_speed = *relative_speed;
		}
		GmVec3_MultTranspose(
			&contact->relative_speed,
			(const GmMat3 *)body_iso(contact->body));
	}
	absorb_contact(contact);
}

/* 0x005497C0  Sorts and dispatches all detected physical collisions.
 * VALIDATED: 2,000 graph-complete A01 records. */
TMNF_HD void CHmsZoneDynamic_ComputeCollisionResponse(CHmsResponseZone *zone) {
	if (zone == NULL || zone->collisions == NULL) {
		tmnf_abort();
	}
	CFastBuffer_SHmsPhysicalCollision_QSort(
		zone->collisions, SHmsPhysicalCollision_Compare);

	uint32_t count = zone->collisions->count;
	for (uint32_t i = 0; i < count; i++) {
		SHmsPhysicalCollision *collision =
			CFastBuffer_SHmsPhysicalCollision_At(zone->collisions, i);
		CHmsResponseBody *body1 =
			resolve_body(zone, collision->corpus1);
		CHmsResponseBody *body2 =
			resolve_body(zone, collision->corpus2);
		const CHmsResponseMaterial *material =
			resolve_material(zone, collision->material);

		uint32_t side1 =
			((body1->classification_flags >> 13) & 0xfu) !=
			material->category;
		CHmsPhysicalContact contact1_storage;
		CHmsPhysicalContact contact2_storage;
		CHmsPhysicalContact *contact1 = make_contact(
			&contact1_storage, body1, collision->tree1,
			collision->collision.material1, body2, collision->tree2,
			collision->collision.material2, &collision->collision,
			material->side_enabled[side1]);
		CHmsPhysicalContact *contact2 = make_contact(
			&contact2_storage, body2, collision->tree2,
			collision->collision.material2, body1, collision->tree2,
			collision->collision.material1, &collision->collision,
			material->side_enabled[1u - side1]);

		if (material->response_mode != 0) {
			CHmsZoneDynamic_SolveImpulse(
				zone, collision, contact1, contact2);
			continue;
		}

		GmVec3 speed1;
		GmVec3 speed2;
		get_speed(body1, &collision->collision.position, &speed1);
		get_speed(body2, &collision->collision.position, &speed2);
		GmVec3 relative_speed = {
			x87_sub(speed2.x, speed1.x),
			x87_sub(speed2.y, speed1.y),
			x87_sub(speed2.z, speed1.z),
		};
		if (contact1 != NULL) {
			fill_external_contact_speed(contact1, &relative_speed, 1);
		}
		if (contact2 != NULL) {
			fill_external_contact_speed(contact2, &relative_speed, 0);
		}
	}
}
