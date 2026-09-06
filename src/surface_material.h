#ifndef TMNF_SURFACE_MATERIAL_H
#define TMNF_SURFACE_MATERIAL_H

#include <stdint.h>
#include <stdlib.h>

/*
 * CPlugSurface physical material ids (EPlugSurfaceMaterialId). The 31 names
 * are the reflection name table at 0x00D05500 (31 char* entries, .data). The
 * per-id friction/restitution pairs the collision response reads live at
 * 0x00D6EEC0 (runtime-initialised .data, 31 x {float, float}); the track
 * snapshot's material-data section is a copy of that table and the loader
 * requires 31 entries. The ids are shared by every environment: a
 * collection only changes which ids its block surfaces use and how the
 * vehicle's material manager (CSceneVehicleCar +0x68, one
 * CSceneVehicleMaterialBlendableVals per entry, indexed through the 31-entry
 * ground-id table at car +0x6c/+0x70) maps them.
 */
typedef enum {
	TMNF_SURFACE_CONCRETE = 0,
	TMNF_SURFACE_PAVEMENT = 1,
	TMNF_SURFACE_GRASS = 2,
	TMNF_SURFACE_ICE = 3,
	TMNF_SURFACE_METAL = 4,
	TMNF_SURFACE_SAND = 5,
	TMNF_SURFACE_DIRT = 6,
	TMNF_SURFACE_TURBO = 7,
	TMNF_SURFACE_DIRT_ROAD = 8,
	TMNF_SURFACE_RUBBER = 9,
	TMNF_SURFACE_SLIDING_RUBBER = 10,
	TMNF_SURFACE_TEST = 11,
	TMNF_SURFACE_ROCK = 12,
	TMNF_SURFACE_WATER = 13,
	TMNF_SURFACE_WOOD = 14,
	TMNF_SURFACE_DANGER = 15,
	TMNF_SURFACE_ASPHALT = 16,
	TMNF_SURFACE_WET_DIRT_ROAD = 17,
	TMNF_SURFACE_WET_ASPHALT = 18,
	TMNF_SURFACE_WET_PAVEMENT = 19,
	TMNF_SURFACE_WET_GRASS = 20,
	TMNF_SURFACE_SNOW = 21,
	TMNF_SURFACE_RESONANT_METAL = 22,
	TMNF_SURFACE_GOLF_BALL = 23,
	TMNF_SURFACE_GOLF_WALL = 24,
	TMNF_SURFACE_GOLF_GROUND = 25,
	TMNF_SURFACE_TURBO2 = 26,
	TMNF_SURFACE_BUMPER = 27,
	TMNF_SURFACE_NOT_COLLIDABLE = 28,
	TMNF_SURFACE_FREE_WHEELING = 29,
	TMNF_SURFACE_TURBO_ROULETTE = 30,
	TMNF_SURFACE_MATERIAL_COUNT = 31
} TmnfSurfaceMaterial;

/* Upper bound on material-manager entries a vehicle snapshot may carry.
 * Stadium has 13; the other collections are read from their snapshots. */
#define TMNF_MAX_GROUND_MATERIALS 32u

static inline const char *TmnfSurfaceMaterial_Name(uint32_t material_id)
{
	static const char *const names[TMNF_SURFACE_MATERIAL_COUNT] = {
		"Concrete", "Pavement", "Grass", "Ice", "Metal", "Sand", "Dirt",
		"Turbo", "DirtRoad", "Rubber", "SlidingRubber", "Test", "Rock",
		"Water", "Wood", "Danger", "Asphalt", "WetDirtRoad", "WetAsphalt",
		"WetPavement", "WetGrass", "Snow", "ResonantMetal", "GolfBall",
		"GolfWall", "GolfGround", "Turbo2", "Bumper", "NotCollidable",
		"FreeWheeling", "TurboRoulette",
	};
	if (material_id >= TMNF_SURFACE_MATERIAL_COUNT)
		abort();
	return names[material_id];
}

#endif
