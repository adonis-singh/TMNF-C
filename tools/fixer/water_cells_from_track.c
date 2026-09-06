/* Print the 32x32 block cells touched by material-13 faces of a track snapshot,
 * plus the y range of those faces, to compare with the live GmMap2 dump. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "track.h"
#include "gm.h"

static void hex(const char *text, uint8_t out[32])
{
	for (int i = 0; i < 32; ++i) {
		unsigned v;
		sscanf(text + 2 * i, "%2x", &v);
		out[i] = (uint8_t)v;
	}
}

int main(int argc, char **argv)
{
	if (argc != 3)
		return 2;
	uint8_t sha[32];
	hex(argv[2], sha);
	TmnfTrack *track = TmnfTrack_Load(argv[1], sha);
	static uint8_t cells[32][32];
	float ymin = 1e9f, ymax = -1e9f;
	uint32_t faces13 = 0, surfaces13 = 0;
	for (uint32_t e = 0; e < track->entry_count; ++e) {
		const HmsStaticCollisionEntry *entry = &track->entries[e];
		const CPlugSurface *surface = entry->surface;
		if (surface == NULL || surface->geom->type != GM_SURF_MESH)
			continue;
		const GmSurfMesh *mesh = (const GmSurfMesh *)surface->geom;
		int any = 0;
		for (uint32_t f = 0; f < mesh->face_count; ++f) {
			const GmSurfMeshFace *face = &mesh->faces[f];
			if (face->material_index >= surface->material_count)
				abort();
			if (surface->material_ids[face->material_index] != 13)
				continue;
			any = 1;
			++faces13;
			for (int k = 0; k < 3; ++k) {
				GmVec3 v = mesh->vertices[face->vertex[k]];
				GmVec3 w;
				GmVec3_SetMult_Iso4(&w, &v, &entry->iso);
				if (w.y < ymin) ymin = w.y;
				if (w.y > ymax) ymax = w.y;
			}
			GmVec3 c = { 0, 0, 0 };
			for (int k = 0; k < 3; ++k) {
				GmVec3 v = mesh->vertices[face->vertex[k]];
				GmVec3 w;
				GmVec3_SetMult_Iso4(&w, &v, &entry->iso);
				c.x += w.x / 3; c.z += w.z / 3;
			}
			int cx = (int)floorf(c.x / 32.0f), cz = (int)floorf(c.z / 32.0f);
			if (cx >= 0 && cx < 32 && cz >= 0 && cz < 32)
				cells[cz][cx] = 1;
		}
		surfaces13 += any;
	}
	printf("material13 faces=%u surfaces=%u y=[%g,%g]\n", faces13, surfaces13, ymin, ymax);
	for (int z = 0; z < 32; ++z) {
		int row = 0;
		for (int x = 0; x < 32; ++x) row |= cells[z][x];
		if (!row) continue;
		printf("%2d ", z);
		for (int x = 0; x < 32; ++x) putchar(cells[z][x] ? '#' : '.');
		putchar('\n');
	}
	TmnfTrack_Unload(track);
	return 0;
}
