#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "track.h"

static const uint8_t A01_SHA256[32] = {
	0xf0, 0xa8, 0x70, 0x80, 0x9b, 0xe9, 0x9d, 0xa2,
	0xcb, 0x36, 0xad, 0x5d, 0xf4, 0x3a, 0x2c, 0xf6,
	0x3d, 0x8f, 0x74, 0xfe, 0x4a, 0xc3, 0x47, 0x0e,
	0xca, 0xc6, 0x8b, 0x9e, 0x97, 0x62, 0x5d, 0xc3,
};

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s A01_SNAPSHOT\n", argv[0]);
		return 2;
	}
	TmnfTrack *track = TmnfTrack_Load(argv[1], A01_SHA256);
	for (uint32_t i = 0; i < track->surface_count; ++i) {
		if (track->surfaces[i].geom->type != GM_SURF_MESH) {
			fprintf(stderr, "surface %u has type %u\n",
				i, track->surfaces[i].geom->type);
			return 1;
		}
	}
	CHmsCollisionManager_SGroup static_group = { 0 };
	TmnfTrack_BindStaticGroup(track, &static_group);
	if (static_group.is_static != 1 ||
		static_group.static_entry_count != track->entry_count ||
		static_group.static_entries != track->entries) {
		fprintf(stderr, "static-group binding differs\n");
		return 1;
	}

	/* Run the same skip-count traversal as the static broad phase around the
	 * first active leaf. This checks the real snapshot's flattened octree,
	 * AABB bytes, inactive-root handling, and tree flags together. */
	uint32_t first_active = UINT32_MAX;
	for (uint32_t i = 0; i < track->entry_count; ++i) {
		if ((track->entries[i].tree_flags & 0x80u) != 0) {
			first_active = i;
			break;
		}
	}
	if (first_active == UINT32_MAX) {
		fprintf(stderr, "snapshot has no active static entry\n");
		return 1;
	}
	const GmBoxAligned *query = &track->entries[first_active].box;
	uint32_t visited_active = 0;
	for (uint32_t i = 0; i < track->entry_count;) {
		const HmsStaticCollisionEntry *entry = &track->entries[i];
		if (!GmBoxAligned_TestInter(query, &entry->box)) {
			i += entry->skip_count;
			continue;
		}
		if ((entry->tree_flags & 0x80u) != 0) {
			++visited_active;
		}
		++i;
	}
	if (visited_active == 0) {
		fprintf(stderr, "static broad phase found no active entry\n");
		return 1;
	}
	const TmnfTrackHeader *header = track->header;
	printf(
		"entries=%u surfaces=%u meshes=%u vertices=%u faces=%u "
		"nodes=%u material_ids=%u materials=%u pairs=%u "
		"probe_hits=%u bytes=%" PRIu64 "\n",
		header->sections[TMNF_TRACK_ENTRIES].count,
		header->sections[TMNF_TRACK_SURFACES].count,
		header->sections[TMNF_TRACK_MESHES].count,
		header->sections[TMNF_TRACK_VERTICES].count,
		header->sections[TMNF_TRACK_FACES].count,
		header->sections[TMNF_TRACK_NODES].count,
		header->sections[TMNF_TRACK_MATERIAL_IDS].count,
		header->sections[TMNF_TRACK_MATERIAL_DATA].count,
		header->sections[TMNF_TRACK_COLLISION_PAIRS].count,
		visited_active,
		header->file_size);
	TmnfTrack_Unload(track);
	return 0;
}
