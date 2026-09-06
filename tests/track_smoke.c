#include <stdint.h>
#include <stdio.h>

#include "track.h"

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s SNAPSHOT\n", argv[0]);
		return 2;
	}
	uint8_t expected_track_sha256[32];
	for (uint32_t i = 0; i < 32; ++i) {
		expected_track_sha256[i] = (uint8_t)i;
	}

	TmnfTrack *track =
		TmnfTrack_Load(argv[1], expected_track_sha256);
	if (track->entry_count != 1 ||
		track->surface_count != 1 ||
		track->mesh_count != 1 ||
		track->materials[0].friction != 1.0f ||
		track->materials[0].restitution != 0.5f ||
		track->entries[0].surface != &track->surfaces[0] ||
		track->surfaces[0].geom != &track->meshes[0].base ||
		((const GmSurfMesh *)track->surfaces[0].geom)->vertex_count != 3) {
		fprintf(stderr, "fixed-up snapshot contents differ\n");
		return 1;
	}
	TmnfTrack_Unload(track);
	return 0;
}
