#define PDOT(a, b) PADD(PADD(PMUL((a)[0], (b)[0]), PMUL((a)[1], (b)[1])), PMUL((a)[2], (b)[2]))
/* Included once per host ISA. FP contraction stays disabled by the build. */
TMNF_PACKET_TARGET static uint32_t TMNF_PACKET_NAME(const GmIso4 *inverse,
                                                    const TmnfSphereFaceEdges *faces,
                                                    const uint32_t *indices, uint32_t count,
                                                    GmCollision *out)
{
	int32_t offsets[TMNF_PACKET_WIDTH];
	for (uint32_t i = 0; i < TMNF_PACKET_WIDTH; ++i)
		offsets[i] =
		    (int32_t)(indices[i < count ? i : 0] * (sizeof(TmnfSphereFaceEdges) / sizeof(float)));
	TMNF_PACKET_INDEX index = TMNF_PACKET_INDEX_LOAD(offsets);
	TMNF_PACKET_VECTOR vertex[3][3];
	for (uint32_t v = 0; v < 3; ++v) {
		TMNF_PACKET_VECTOR x = TMNF_PACKET_GATHER(&faces[0].xyz[0][v], index);
		TMNF_PACKET_VECTOR y = TMNF_PACKET_GATHER(&faces[0].xyz[1][v], index);
		TMNF_PACKET_VECTOR z = TMNF_PACKET_GATHER(&faces[0].xyz[2][v], index);
		for (uint32_t axis = 0; axis < 3; ++axis) {
			vertex[v][axis] = PADD(PADD(PADD(PMUL(PSET(inverse->m[axis * 3]), x),
			                                 PMUL(PSET(inverse->m[axis * 3 + 1]), y)),
			                            PMUL(PSET(inverse->m[axis * 3 + 2]), z)),
			                       PSET(inverse->t[axis]));
		}
	}
	TMNF_PACKET_VECTOR a[3], b[3], normal[3];
	for (uint32_t axis = 0; axis < 3; ++axis) {
		a[axis] = PSUB(vertex[1][axis], vertex[0][axis]);
		b[axis] = PSUB(vertex[2][axis], vertex[0][axis]);
	}
	normal[0] = PSUB(PMUL(a[1], b[2]), PMUL(a[2], b[1]));
	normal[1] = PSUB(PMUL(b[0], a[2]), PMUL(a[0], b[2]));
	normal[2] = PSUB(PMUL(b[1], a[0]), PMUL(a[1], b[0]));
	TMNF_PACKET_VECTOR squared = PADD(PADD(PMUL(normal[0], normal[0]), PMUL(normal[1], normal[1])),
	                                  PMUL(normal[2], normal[2]));
	uint32_t valid = PGT(squared, PSET(0x1.b7cdfcp-34f));
	TMNF_PACKET_VECTOR reciprocal = PDIV(PSET(1.0f), PSQRT(squared));
	normal[0] = PMUL(reciprocal, normal[0]);
	normal[1] = PMUL(normal[1], reciprocal);
	normal[2] = PMUL(reciprocal, normal[2]);
	TMNF_PACKET_VECTOR distance =
	    PADD(PADD(PMUL(PNEG(vertex[0][0]), normal[0]), PMUL(PNEG(vertex[0][1]), normal[1])),
	         PMUL(PNEG(vertex[0][2]), normal[2]));
	valid &= ~(PGT(distance, PSET(1.0f)) | PGT(PSET(0.0f), distance));
	valid &= (1u << count) - 1u;
	if (valid == 0)
		return 0;

	TMNF_PACKET_VECTOR zero = PSET(0.0f), one = PSET(1.0f);
	TMNF_PACKET_VECTOR projected[3], position[3], contact_normal[3], separation[3];
	for (uint32_t axis = 0; axis < 3; ++axis) {
		projected[axis] = PADD(PMUL(normal[axis], PNEG(distance)), zero);
		position[axis] = projected[axis];
		contact_normal[axis] = normal[axis];
		separation[axis] = PMUL(PSUB(distance, one), normal[axis]);
	}
	TMNF_PACKET_VECTOR radius = PSQRT(PSUB(one, PMUL(distance, distance)));
	uint32_t active = valid, emitted = 0;
	for (uint32_t edge_index = 0; edge_index < 3 && active; ++edge_index) {
		uint32_t next = (edge_index + 1) % 3;
		TMNF_PACKET_VECTOR edge[3], side[3], from_start[3], from_end[3];
		for (uint32_t axis = 0; axis < 3; ++axis)
			edge[axis] = PSUB(vertex[next][axis], vertex[edge_index][axis]);
		TMNF_PACKET_VECTOR edge_squared = PDOT(edge, edge);
		uint32_t normalize = PGT(edge_squared, PSET(0x1.b7cdfcp-34f));
		TMNF_PACKET_VECTOR inv = PDIV(one, PSQRT(edge_squared));
		for (uint32_t axis = 0; axis < 3; ++axis)
			edge[axis] = PBLEND(normalize, edge[axis], PMUL(inv, edge[axis]));
		side[0] = PSUB(PMUL(edge[1], normal[2]), PMUL(edge[2], normal[1]));
		side[1] = PSUB(PMUL(edge[2], normal[0]), PMUL(normal[2], edge[0]));
		side[2] = PSUB(PMUL(normal[1], edge[0]), PMUL(edge[1], normal[0]));
		for (uint32_t axis = 0; axis < 3; ++axis)
			from_start[axis] = PSUB(projected[axis], vertex[edge_index][axis]);
		TMNF_PACKET_VECTOR side_distance = PDOT(from_start, side);
		active &= ~PGT(side_distance, radius);
		uint32_t positive = active & PGT(side_distance, zero);
		if (!positive)
			continue;
		/* Only the first positive side may emit or reject a given face. */
		active &= ~positive;
		TMNF_PACKET_VECTOR along_start = PDOT(from_start, edge);
		for (uint32_t axis = 0; axis < 3; ++axis)
			from_end[axis] = PSUB(projected[axis], vertex[next][axis]);
		TMNF_PACKET_VECTOR along_end = PDOT(from_end, edge);
		uint32_t after_start = PLE(zero, along_start);
		uint32_t is_edge = after_start & PLE(along_end, zero);
		uint32_t is_end = after_start & ~is_edge;
		TMNF_PACKET_VECTOR feature[3];
		for (uint32_t axis = 0; axis < 3; ++axis) {
			TMNF_PACKET_VECTOR closest =
			    PADD(PMUL(PNEG(side_distance), side[axis]), projected[axis]);
			feature[axis] = PBLEND(is_end, vertex[edge_index][axis], vertex[next][axis]);
			feature[axis] = PBLEND(is_edge, feature[axis], closest);
		}
		TMNF_PACKET_VECTOR feature_squared = PDOT(feature, feature);
		uint32_t reject =
		    (is_edge & PLE(feature_squared, PSET(0x1.4f8b58p-17f))) |
		    (~is_edge & (PGT(feature_squared, one) | PLE(feature_squared, PSET(0x1.b7cdfcp-34f))));
		uint32_t emit = positive & ~reject;
		if (!emit)
			continue;
		TMNF_PACKET_VECTOR feature_distance = PSQRT(feature_squared);
		/* Retain the game's double square root for the end vertex. */
		feature_distance = PBLEND(is_end, feature_distance, PSQRT(feature_distance));
		TMNF_PACKET_VECTOR inverse_distance = PDIV(one, feature_distance);
		TMNF_PACKET_VECTOR scale = PMUL(PSUB(feature_distance, one), inverse_distance);
		TMNF_PACKET_VECTOR radial[3];
		for (uint32_t axis = 0; axis < 3; ++axis) {
			position[axis] = PBLEND(emit, position[axis], feature[axis]);
			contact_normal[axis] =
			    PBLEND(emit, contact_normal[axis], PMUL(inverse_distance, PNEG(feature[axis])));
			radial[axis] = PMUL(scale, PNEG(feature[axis]));
		}
		TMNF_PACKET_VECTOR plane_scale = PDOT(normal, radial);
		for (uint32_t axis = 0; axis < 3; ++axis)
			separation[axis] = PBLEND(emit, separation[axis], PMUL(plane_scale, normal[axis]));
		emitted |= emit;
	}
	uint32_t face_contacts = active & PGT(distance, zero);
	valid = emitted | face_contacts;
	if (!valid)
		return 0;
	float values[12][TMNF_PACKET_WIDTH];
	for (uint32_t axis = 0; axis < 3; ++axis) {
		PSTORE(values[axis], position[axis]);
		PSTORE(values[3 + axis], contact_normal[axis]);
		PSTORE(values[6 + axis], separation[axis]);
		PSTORE(values[9 + axis], normal[axis]);
	}
	for (uint32_t i = 0; i < count; ++i) {
		if ((valid & (1u << i)) == 0)
			continue;
		out[i].position = (GmVec3){values[0][i], values[1][i], values[2][i]};
		out[i].normal = (GmVec3){values[3][i], values[4][i], values[5][i]};
		out[i].separation = (GmVec3){values[6][i], values[7][i], values[8][i]};
		out[i].face_normal = (GmVec3){values[9][i], values[10][i], values[11][i]};
		out[i].material1 = out[i].material2 = 0;
		out[i].flags = (face_contacts >> i) & 1u;
	}
	return valid;
}

#undef TMNF_PACKET_TARGET
#undef TMNF_PACKET_NAME
#undef TMNF_PACKET_WIDTH
#undef TMNF_PACKET_VECTOR
#undef TMNF_PACKET_INDEX
#undef TMNF_PACKET_INDEX_LOAD
#undef TMNF_PACKET_GATHER
#undef PADD
#undef PSUB
#undef PMUL
#undef PDIV
#undef PSQRT
#undef PSET
#undef PSTORE
#undef PGT
#undef PNEG

#undef PLE
#undef PBLEND
#undef PDOT
