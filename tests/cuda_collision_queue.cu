/* Directed capacity tests. The only real contact is last, beyond either
 * the query queue or many face chunks, so dropping work cannot pass. */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "collision.h"
#include "tmnf_warp.h"

__device__ void tmnf_dev_static_tree_detect(CHmsCollisionManager_SZone *, const GmIso4 *,
                                            CPlugTree *, int);
static void check(cudaError_t e)
{
	if (e != cudaSuccess) {
		fprintf(stderr, "queue test: %s\n", cudaGetErrorString(e));
		exit(1);
	}
}
struct Scene {
	GmVec3 vertices[3];
	GmSurfMesh mesh;
	GmSurfEllipsoid ellipsoid;
	CPlugSurface surface[2];
	uint8_t materials[2];
};
struct Output {
	uint32_t count;
	SHmsPhysicalCollision collision;
};

__global__ static void probe(Scene *scene, HmsStaticCollisionEntry *entries, uint32_t queries,
                             uint32_t count, int sparse, uint8_t *workspace, size_t stride,
                             Output *out)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t mask = __ballot_sync(__activemask(), index < count);
	if (index >= count)
		return;
	tmnf_dev_warp_begin(mask, workspace + (index / 32) * stride, 4);
	GmIso4 identity = {};
	identity.m[0] = identity.m[4] = identity.m[8] = 1;
	SHmsPhysicalCollision records[4] = {};
	SHmsSphereBufferContact contacts = {};
	contacts.base.collisions.data = records;
	contacts.base.collisions.capacity = 4;
	CPlugTree tree = {};
	tree.flags = 0x80;
	tree.box.half_extent = {1, 1, 1};
	tree.local_iso = identity;
	tree.surface = &scene->surface[0];
	tree.contact_buffer = &contacts;
	tree.object_ref = 123;
	CollisionRuntime runtime;
	CollisionRuntime_Init(&runtime);
	CHmsCollisionManager_SGroup group = {};
	group.static_entries = entries;
	group.static_entry_count = queries;
	SHmsSphereBufferContact *merge[1];
	CHmsCollisionManager_SZone zone = {};
	zone.runtime = &runtime;
	zone.static_group = &group;
	zone.merge_buffers = merge;
	zone.merge_buffer_capacity = 1;
	zone.current_corpus1 = index;
	zone.current_material = 789;
	int active = !sparse || index % 3 == 0;
	tmnf_dev_static_tree_detect(&zone, &identity, &tree, active);
	out[index].count = contacts.base.collisions.count;
	if (out[index].count)
		out[index].collision = records[0];
}

int main()
{
	check(cudaDeviceSetLimit(cudaLimitStackSize, 40960));
	for (const auto &test :
	     {std::vector<uint32_t>{1, 65, 1, 0}, {32, 40, 1, 0}, {33, 1, 10000, 0}, {63, 40, 1, 1}}) {
		uint32_t count = test[0], queries = test[1], faces = test[2];
		Scene host = {};
		host.vertices[0] = {-2, -2, -0.5f};
		host.vertices[1] = {2, -2, -0.5f};
		host.vertices[2] = {0, 2, -0.5f};
		host.mesh.base.type = GM_SURF_MESH;
		host.mesh.vertex_count = 3;
		host.mesh.face_count = faces;
		host.mesh.node_count = faces;
		host.ellipsoid.base.type = GM_SURF_ELLIPSOID;
		host.ellipsoid.radii = {1, 1, 1};
		host.materials[0] = 5;
		host.materials[1] = 7;
		std::vector<GmSurfMeshFace> face(faces);
		std::vector<GmSurfMeshNode> node(faces);
		for (uint32_t i = 0; i < faces; ++i) {
			face[i].vertex[0] = 0;
			face[i].vertex[1] = i + 1 == faces ? 1 : 0;
			face[i].vertex[2] = i + 1 == faces ? 2 : 0;
			face[i].normal = {0, 0, 1};
			node[i].face_index = i;
			node[i].skip_count = 1;
			node[i].box.center = {0, 0, -0.5f};
			node[i].box.half_extent = {2, 2, 0};
		}
		Scene *device;
		GmSurfMeshFace *device_faces;
		GmSurfMeshNode *device_nodes;
		HmsStaticCollisionEntry *device_entries;
		Output *output;
		uint8_t *workspace;
		check(cudaMalloc(&device, sizeof(host)));
		check(cudaMalloc(&device_faces, faces * sizeof(face[0])));
		check(cudaMalloc(&device_nodes, faces * sizeof(node[0])));
		check(cudaMalloc(&device_entries, queries * sizeof(HmsStaticCollisionEntry)));
		check(cudaMalloc(&output, count * sizeof(Output)));
		size_t stride = tmnf_dev_warp_workspace_bytes(4);
		check(cudaMalloc(&workspace, ((count + 31) / 32) * stride));
		host.mesh.vertices = device->vertices;
		host.mesh.faces = device_faces;
		host.mesh.nodes = device_nodes;
		host.surface[0].geom = &device->ellipsoid.base;
		host.surface[1].geom = &device->mesh.base;
		for (int k = 0; k < 2; ++k) {
			host.surface[k].material_ids = &device->materials[k];
			host.surface[k].material_count = 1;
		}
		std::vector<HmsStaticCollisionEntry> entries(queries);
		for (uint32_t i = 0; i < queries; ++i) {
			auto &e = entries[i];
			e.skip_count = 1;
			e.box.half_extent = {1000, 1000, 1000};
			e.iso.m[0] = e.iso.m[4] = e.iso.m[8] = 1;
			e.iso.t[2] = i + 1 == queries ? 0 : -100;
			e.surface = &device->surface[1];
			e.tree_flags = 0x80;
			e.tree_ref = 456;
			e.corpus_ref = 987;
		}
		check(cudaMemcpy(device, &host, sizeof(host), cudaMemcpyHostToDevice));
		check(
		    cudaMemcpy(device_faces, face.data(), faces * sizeof(face[0]), cudaMemcpyHostToDevice));
		check(
		    cudaMemcpy(device_nodes, node.data(), faces * sizeof(node[0]), cudaMemcpyHostToDevice));
		check(cudaMemcpy(device_entries, entries.data(), queries * sizeof(entries[0]),
		                 cudaMemcpyHostToDevice));
		probe<<<(count + 127) / 128, 128>>>(device, device_entries, queries, count, test[3],
		                                    workspace, stride, output);
		check(cudaGetLastError());
		check(cudaDeviceSynchronize());
		std::vector<Output> result(count);
		check(cudaMemcpy(result.data(), output, count * sizeof(Output), cudaMemcpyDeviceToHost));
		host.mesh.vertices = host.vertices;
		host.mesh.faces = face.data();
		host.mesh.nodes = node.data();
		GmIso4 identity = {};
		identity.m[0] = identity.m[4] = identity.m[8] = 1;
		LocatedGmSurf first = {&host.ellipsoid.base, &identity, 1, NULL},
		              second = {&host.mesh.base, &identity, 1, NULL};
		SHmsPhysicalCollision record = {};
		CHmsCollisionBuffer buffer = {};
		buffer.collisions.data = &record;
		buffer.collisions.capacity = 1;
		if (!GmCollision_Ellipsoid_Mesh(&first, &second, &buffer) || buffer.collisions.count != 1)
			return 1;
		record.collision.material1 = 5;
		record.collision.material2 = 7;
		for (uint32_t i = 0; i < count; ++i) {
			bool active = !test[3] || i % 3 == 0;
			const auto &r = result[i];
			if (r.count != (uint32_t)active ||
			    (active &&
			     (memcmp(&r.collision.collision, &record.collision, sizeof(GmCollision)) ||
			      r.collision.corpus1 != i || r.collision.corpus2 != 987 ||
			      r.collision.tree1 != 123 || r.collision.tree2 != 456 ||
			      r.collision.material != 789))) {
				fprintf(stderr,
				        "queue test mismatch: %u envs %u queries %u faces env %u count %u\n", count,
				        queries, faces, i, r.count);
				return 1;
			}
		}
		printf("queue: %u envs, %u queries, %u faces, sparse %u: exact\n", count, queries, faces,
		       test[3]);
		check(cudaFree(workspace));
		check(cudaFree(output));
		check(cudaFree(device_entries));
		check(cudaFree(device_nodes));
		check(cudaFree(device_faces));
		check(cudaFree(device));
	}
}
