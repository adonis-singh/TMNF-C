/* Device compilation unit for src/collision.c. */
#include "../../collision.c"
#include "../../tmnf_warp.h"

/* Static queries retain each car's original order. At most one query per
 * active lane is prepared at a time; node scans resume in bounded chunks.
 * Face work is pooled across the warp, then only emitted contacts are kept
 * for ordered replay. Large meshes need more rounds, not larger scratch.
 * The contact store has the same per-query bound as the destination child
 * buffer: a valid query cannot emit more contacts than that buffer holds. */
enum {
	WARP_QUERY_CAP = 1024,
	WARP_LANE_QUERY_CAP = 64,
	WARP_LANE_JOB_CAP = 64,
	WARP_ACTIVE_QUERIES = 256,
	WARP_SCAN_CHUNK = 256,
	WARP_JOB_CAP = 32 * WARP_LANE_JOB_CAP,
};

enum { WARP_QUERY_ELLIPSOID_MESH = 0, WARP_QUERY_OTHER = 1 };

typedef struct {
	/* Recorded by the owner (phase A). tree is in the owner's world; the
	 * entry and grid are in the track, shared by every lane. */
	CPlugTree *tree;
	const HmsStaticCollisionEntry *entry;
	const TmnfStaticGrid *grid;
	uint32_t entry_index;
	uint32_t grid_accel;        /* static_grid_scan path: the entry's accel */
	uint32_t kind;
	GmIso4 world;               /* tree's world iso */
	GmSurfEllipsoid ellipsoid;  /* copy of tree->surface->geom */
	uint32_t producer;
	uint32_t contact_count;
} WarpQuery;

typedef struct {
	GmCollision collision;
	uint32_t emitted;
	uint32_t reserved;
} WarpFaceResult;

typedef struct {
	WarpQuery queries[WARP_QUERY_CAP];          /* in record order */
	EllipsoidMeshQuery active_queries[WARP_ACTIVE_QUERIES];
	WarpFaceResult results[WARP_JOB_CAP];
	uint2 lane_jobs[32][WARP_LANE_JOB_CAP]; /* query slot, node index */
	uint16_t lane_slots[32][WARP_LANE_QUERY_CAP]; /* a lane's queries, in order */
} WarpCollisionScratch;

typedef struct {
	uint32_t query_count;       /* queries recorded so far */
	uint32_t overflow;          /* exact scalar fallback for exceptional scans */
	uint32_t job_total;         /* face tests recorded this round */
	uint32_t counts[32];        /* per lane, this round */
	uint32_t starts[32];        /* exclusive prefix of counts */
} WarpJobShared;

__shared__ WarpJobShared tmnf_warp_jobs[TMNF_WARPS_PER_BLOCK_MAX];

size_t tmnf_dev_warp_workspace_bytes(uint32_t contact_capacity)
{
	return sizeof(WarpCollisionScratch) +
		(size_t)WARP_ACTIVE_QUERIES * contact_capacity * sizeof(GmCollision);
}

__device__ static GmCollision *query_contacts(
	WarpCollisionScratch *ws, uint32_t producer)
{
	GmCollision *records = (GmCollision *)((uint8_t *)ws + sizeof(*ws));
	return records + (size_t)producer * tmnf_dev_warp_contact_capacity();
}

/* ---- phase A --------------------------------------------------------------- */

__device__ static void record_query(
	CHmsCollisionManager_SZone *self, WarpCollisionScratch *ws,
	WarpJobShared *sh, uint32_t lane, uint32_t *count, CPlugTree *tree,
	const GmIso4 *world, uint32_t entry_index, uint32_t grid_accel)
{
	const HmsStaticCollisionEntry *entry =
		&self->static_group->static_entries[entry_index];
	if (entry->surface == NULL)
		return;  /* compute_surface_pair: nothing */
	if (*count == WARP_LANE_QUERY_CAP) {
		atomicExch(&sh->overflow, 1u);
		return;
	}
	uint32_t slot = atomicAdd(&sh->query_count, 1u);
	if (slot >= WARP_QUERY_CAP) {
		atomicExch(&sh->overflow, 1u);
		return;
	}
	ws->lane_slots[lane][(*count)++] = (uint16_t)slot;
	WarpQuery *wq = &ws->queries[slot];
	wq->tree = tree;
	wq->entry = entry;
	wq->grid = self->static_group->static_grid;
	wq->entry_index = entry_index;
	wq->grid_accel = grid_accel;
	wq->world = *world;
	const GmSurf *geom = tree->surface->geom;
	if (geom->type == GM_SURF_ELLIPSOID &&
		entry->surface->geom->type == GM_SURF_MESH) {
		wq->kind = WARP_QUERY_ELLIPSOID_MESH;
		wq->ellipsoid = *(const GmSurfEllipsoid *)geom;
	} else {
		wq->kind = WARP_QUERY_OTHER;
	}
}

/* static_tree_detect with the pair computations replaced by records. */
__device__ static void record_static_tree(
	CHmsCollisionManager_SZone *self, WarpCollisionScratch *ws,
	WarpJobShared *sh, uint32_t lane, uint32_t *count, const GmIso4 *iso,
	CPlugTree *tree)
{
	if ((tree->flags & 0x80u) == 0)
		return;
	GmIso4 world;
	tree_world_iso(&world, tree, iso);
	for (uint32_t i = 0; i < tree->child_count; i++)
		record_static_tree(self, ws, sh, lane, count, &world,
			tree->children[i]);
	if (tree->surface == NULL || self->static_group == NULL)
		return;

	GmBoxAligned world_box;
	GmBoxQuery world_query =
		GmBoxAligned_SetMultQuery(&world_box, &tree->box, iso);
	const TmnfStaticGrid *grid = self->static_group->static_grid;
	if (grid != NULL) {
		uint32_t cell;
		if (static_grid_locate(grid, &world_box, tree->surface, &cell)) {
			/* static_grid_scan */
			const TmnfStaticCellNode *nodes =
				grid->nodes + grid->cell_offsets[cell];
			uint32_t n = grid->cell_counts[cell];
			uint32_t i = 0;
			while (i < n) {
				const TmnfStaticCellNode *node = &nodes[i];
				if (!GmBoxQuery_TestInter(&world_query, &node->box)) {
					i += node->skip;
					continue;
				}
				i++;
				if (node->entry_index >= TMNF_CELL_NODE_EMPTY)
					continue;
				record_query(self, ws, sh, lane, count, tree, &world,
					node->entry_index, 1);
			}
			return;
		}
	}

	uint32_t i = 0;
	while (i < self->static_group->static_entry_count) {
		HmsStaticCollisionEntry *entry =
			&self->static_group->static_entries[i];
		if (!GmBoxQuery_TestInter(&world_query, &entry->box)) {
			i += entry->skip_count;
			continue;
		}
		if (entry->surface != NULL && (entry->tree_flags & 0x80u) != 0)
			record_query(self, ws, sh, lane, count, tree, &world, i, 0);
		i++;
	}
}

/* ---- phase B --------------------------------------------------------------- */

__device__ static TmnfMeshQueryAccel entry_accel(
	const TmnfStaticGrid *grid, uint32_t entry_index)
{
	TmnfMeshQueryAccel accel;
	accel.inverse_iso = &grid->entry_inverse_isos[entry_index];
	accel.mesh_grid = grid->entry_mesh_grids[entry_index];
	return accel;
}

__device__ static void begin_query(WarpQuery *wq, EllipsoidMeshQuery *q)
{
	const HmsStaticCollisionEntry *entry = wq->entry;
	LocatedGmSurf ellipsoid = {
		(GmSurf *)&wq->ellipsoid.base, &wq->world, 1, NULL,
	};
	TmnfMeshQueryAccel accel;
	if (wq->grid_accel)
		accel = entry_accel(wq->grid, wq->entry_index);
	LocatedGmSurf mesh = {
		entry->surface->geom, &entry->iso, 1,
		wq->grid_accel ? &accel : NULL,
	};
	ellipsoid_mesh_begin(q, &ellipsoid, &mesh);
	/* The CPU computes these on the first contact; they depend on the
	 * query only, so every lane testing a face of it sees them ready. */
	prepare_ellipsoid_output_isos(
		&q->output_isos, q->ellipsoid, &q->inverse_relative, q->mesh_iso);
}

__device__ static void run_face_job(
	WarpCollisionScratch *ws, uint32_t producer, uint32_t node_index,
	WarpFaceResult *result)
{
	EllipsoidMeshQuery *q = &ws->active_queries[producer];
	uint32_t skip;
	const GmSurfMeshNode *node = ellipsoid_mesh_node(q, node_index, &skip);
	SHmsPhysicalCollision record;
	CHmsCollisionBuffer buffer;
	buffer.collisions.count = 0;
	buffer.collisions.data = &record;
	buffer.collisions.capacity = 1;
	result->emitted = (uint32_t)ellipsoid_mesh_face(q, node, &buffer);
	if (result->emitted)
		result->collision = record.collision;
}

/* Replays the lane's queries recorded before `limit`, in order, from the
 * first not yet replayed. */
__device__ static void replay_query(
	CHmsCollisionManager_SZone *self, WarpCollisionScratch *ws,
	const WarpQuery *wq);

__device__ static void replay_prefix(
	CHmsCollisionManager_SZone *self, WarpCollisionScratch *ws,
	uint32_t lane, uint32_t count,
	uint32_t *replayed, uint32_t limit)
{
	while (*replayed < count) {
		uint32_t slot = ws->lane_slots[lane][*replayed];
		if (slot >= limit)
			break;
		replay_query(self, ws, &ws->queries[slot]);
		(*replayed)++;
	}
}

/* Job g of the round: the k-th of lane p's list, by the prefix in sh->starts. */
__device__ static uint32_t job_at(const WarpJobShared *sh, uint32_t g)
{
	uint32_t p = 0;
	for (uint32_t step = 16; step > 0; step >>= 1) {
		if (p + step < 32 && sh->starts[p + step] <= g)
			p += step;
	}
	return (uint32_t)p << 16 | (g - sh->starts[p]);
}

/* A lane produces an ordered series of queries, resuming node traversal
 * between bounded chunks. All live lanes execute the resulting face jobs.
 * Only emitted contacts survive the chunk. Query tiles follow recording
 * order and each query has one producer, so compaction and replay are stable. */
__device__ static void pool_queries(
	CHmsCollisionManager_SZone *self, WarpCollisionScratch *ws,
	WarpJobShared *sh, uint32_t mask, uint32_t count)
{
	uint32_t lane = threadIdx.x % 32;
	uint32_t rank = __popc(mask & ((1u << lane) - 1u));
	uint32_t width = __popc(mask);
	uint32_t low = (uint32_t)(__ffs((int)mask) - 1);
	uint32_t total = sh->query_count;
	uint32_t replayed = 0;
	uint32_t capacity = tmnf_dev_warp_contact_capacity();
	uint2 *my_list = ws->lane_jobs[lane];

	for (uint32_t first = 0; first < total; first += WARP_ACTIVE_QUERIES) {
		uint32_t n = total - first;
		if (n > WARP_ACTIVE_QUERIES)
			n = WARP_ACTIVE_QUERIES;
		for (uint32_t i = rank; i < n; i += width) {
			ws->queries[first + i].producer = i;
			ws->queries[first + i].contact_count = 0;
		}
		uint32_t current = rank;
		uint32_t node_index = 0;
		uint32_t node_count = 0;
		int prepared = 0;
		EllipsoidMeshQuery q;
		__syncwarp(mask);
		while (__any_sync(mask, current < n)) {
			uint32_t my_jobs = 0;
			uint32_t visited = 0;
			while (current < n && my_jobs < WARP_LANE_JOB_CAP &&
				visited < WARP_SCAN_CHUNK) {
				WarpQuery *cur = &ws->queries[first + current];
				if (!prepared) {
					if (cur->kind != WARP_QUERY_ELLIPSOID_MESH) {
						current += width;
						continue;
					}
					begin_query(cur, &q);
					ws->active_queries[current] = q;
					node_count = q.node_count;
					node_index = 0;
					prepared = 1;
				}
				while (node_index < node_count && my_jobs < WARP_LANE_JOB_CAP &&
					visited++ < WARP_SCAN_CHUNK) {
					uint32_t skip;
					const GmSurfMeshNode *node = ellipsoid_mesh_node(&q, node_index, &skip);
					if (!GmBoxQuery_TestInter(&q.mesh_query, &node->box)) {
						node_index += skip;
						continue;
					}
					if (node->face_index != UINT32_MAX)
						my_list[my_jobs++] = make_uint2(current, node_index);
					node_index++;
				}
				if (node_index == node_count) {
					current += width;
					prepared = 0;
				}
			}
			if (mask == 0xffffffffu) {
				uint32_t prefix = my_jobs;
				for (uint32_t delta = 1; delta < 32; delta <<= 1) {
					uint32_t before = __shfl_up_sync(mask, prefix, delta);
					if (lane >= delta)
						prefix += before;
				}
				sh->starts[lane] = prefix - my_jobs;
				if (lane == 31)
					sh->job_total = prefix;
			} else {
				sh->counts[lane] = my_jobs;
				__syncwarp(mask);
				if (lane == low) {
					uint32_t sum = 0;
					for (uint32_t l = 0; l < 32; ++l) {
						sh->starts[l] = sum;
						if ((mask >> l) & 1u)
							sum += sh->counts[l];
					}
					sh->job_total = sum;
				}
			}
			__syncwarp(mask);
			uint32_t jobs = sh->job_total;
			for (uint32_t g0 = 0; g0 < jobs; g0 += width) {
				uint32_t g = g0 + rank;
				if (g < jobs) {
					uint32_t where = job_at(sh, g);
					uint2 job = ws->lane_jobs[where >> 16][where & 0xffffu];
					run_face_job(ws, job.x, job.y, &ws->results[g]);
				}
			}
			__syncwarp(mask);
			const WarpFaceResult *results = &ws->results[sh->starts[lane]];
			for (uint32_t j = 0; j < my_jobs; ++j) {
				if (results[j].emitted) {
					uint32_t owner = my_list[j].x;
					WarpQuery *cur = &ws->queries[first + owner];
					if (cur->contact_count == capacity)
						tmnf_fail("warp collision: query exceeds child contact capacity");
					query_contacts(ws, owner)[cur->contact_count++] = results[j].collision;
				}
			}
			__syncwarp(mask);
		}
		replay_prefix(self, ws, lane, count, &replayed, first + n);
		__syncwarp(mask);
	}
	if (replayed != count)
		tmnf_fail("warp collision: queries left unreplayed");
}

/* ---- phase C --------------------------------------------------------------- */

/* compute_surface_pair for one recorded query. */
__device__ static void replay_query(
	CHmsCollisionManager_SZone *self, WarpCollisionScratch *ws,
	const WarpQuery *wq)
{
	const HmsStaticCollisionEntry *entry = wq->entry;
	TmnfMeshQueryAccel accel;
	const TmnfMeshQueryAccel *accel_ptr = NULL;
	if (wq->grid_accel) {
		accel = entry_accel(wq->grid, wq->entry_index);
		accel_ptr = &accel;
	}
	CPlugTree *tree = wq->tree;
	if (wq->kind != WARP_QUERY_ELLIPSOID_MESH) {
		(void)compute_surface_pair(
			self,
			tree, &wq->world,
			NULL, &entry->iso, accel_ptr,
			tree->surface, entry->surface,
			self->current_corpus1, entry->corpus_ref,
			tree->object_ref, entry->tree_ref);
		return;
	}

	SHmsSphereBufferContact *mergeable = tree_contact_buffer(tree);
	CHmsCollisionBuffer *buffer = &mergeable->base;
	uint32_t start = CHmsCollisionBuffer_GetCount(buffer);
	const GmCollision *contacts = query_contacts(ws, wq->producer);
	for (uint32_t j = 0; j < wq->contact_count; ++j)
		*CHmsCollisionBuffer_AddCollision(buffer) = contacts[j];
	uint32_t end = CHmsCollisionBuffer_GetCount(buffer);
	if (end == start)
		return;

	/* compute_surface_collision's material remap */
	const CPlugSurface *surface1 = tree->surface;
	const CPlugSurface *surface2 = entry->surface;
	for (uint32_t i = start; i < end; i++) {
		GmCollision *collision = CHmsCollisionBuffer_GetCollision(buffer, i);
		if (collision->material1 >= surface1->material_count ||
			collision->material2 >= surface2->material_count) {
			tmnf_abort();
		}
		collision->material1 = surface1->material_ids[collision->material1];
		collision->material2 = surface2->material_ids[collision->material2];
	}

	if (mergeable->active == 0) {
		mergeable->active = 1;
		append_merge_buffer(self, mergeable);
	}
	for (uint32_t i = start; i < end; i++) {
		SHmsPhysicalCollision *collision =
			CFastBuffer_SHmsPhysicalCollision_At(&buffer->collisions, i);
		collision->corpus1 = self->current_corpus1;
		collision->tree1 = tree->object_ref;
		collision->corpus2 = entry->corpus_ref;
		collision->tree2 = entry->tree_ref;
		collision->material = self->current_material;
	}
}

/* ---- the collective call --------------------------------------------------- */

__device__ void tmnf_dev_static_tree_detect(
	CHmsCollisionManager_SZone *self, const GmIso4 *iso, CPlugTree *tree,
	int active)
{
	uint32_t mask = tmnf_dev_warp_mask();
	uint32_t lane = threadIdx.x % 32;
	WarpCollisionScratch *ws =
		(WarpCollisionScratch *)tmnf_dev_warp_workspace();
	WarpJobShared *sh = &tmnf_warp_jobs[threadIdx.x / 32];

	if (lane == (uint32_t)(__ffs((int)mask) - 1)) {
		sh->query_count = 0;
		sh->overflow = 0;
		sh->job_total = 0;
	}
	__syncwarp(mask);
	uint32_t count = 0;
	if (active)
		record_static_tree(self, ws, sh, lane, &count, iso, tree);
	__syncwarp(mask);
	/* Recording has no physics side effects. If its bounded queue fills,
	 * replay the original scan directly, preserving every query and contact. */
	if (sh->overflow) {
		if (active)
			static_tree_detect(self, iso, tree);
	} else {
		pool_queries(self, ws, sh, mask, count);
	}
	__syncwarp(mask);
}
