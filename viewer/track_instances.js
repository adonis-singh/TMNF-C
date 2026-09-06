/* Static game blocks share geometry and materials. Batch opaque copies in
 * spatial tiles so draw calls scale with assets, while culling remains local.
 * Transparent meshes keep individual objects for Three's depth sorting. */
(() => {
	"use strict";
	globalThis.buildTrackInstances = (templates, placements) => {
		const track = new THREE.Group();
		track.name = "TMNF game block visuals";
		const batches = new Map();
		const placementMatrix = new THREE.Matrix4();
		const transform = new THREE.Matrix4();
		let sourceMeshes = 0;
		for (const template of templates.values())
			template.updateMatrixWorld(true);
		for (const placement of placements) {
			placementMatrix.makeRotationY(placement.rotationY);
			placementMatrix.setPosition(...placement.position);
			templates.get(placement.asset).traverseVisible((source) => {
				if (!source.isMesh)
					return;
				sourceMeshes++;
				transform.multiplyMatrices(placementMatrix, source.matrixWorld);
				const materials = Array.isArray(source.material) ? source.material : [source.material];
				if (materials.some(material => material.transparent) || transform.determinant() < 0) {
					const mesh = source.clone(false);
					mesh.matrix.copy(transform);
					mesh.matrixAutoUpdate = false;
					track.add(mesh);
					return;
				}
				// Each source primitive has a fixed material and local transform.
				// Tiles avoid drawing an entire campaign's repeated road pieces
				// when only one corner is in view.
				const tile = `${Math.floor(placement.position[0] / 128)},${Math.floor(placement.position[2] / 128)}`;
				const key = `${source.uuid}:${tile}`;
				if (!batches.has(key))
					batches.set(key, { source, matrices: [] });
				batches.get(key).matrices.push(transform.clone());
			});
		}
		let instancedMeshes = 0;
		let instances = 0;
		for (const { source, matrices } of batches.values()) {
			if (matrices.length === 1) {
				const mesh = source.clone(false);
				mesh.matrix.copy(matrices[0]);
				mesh.matrixAutoUpdate = false;
				track.add(mesh);
				continue;
			}
			const mesh = new THREE.InstancedMesh(source.geometry, source.material, matrices.length);
			mesh.name = source.name;
			mesh.castShadow = source.castShadow;
			mesh.receiveShadow = source.receiveShadow;
			mesh.renderOrder = source.renderOrder;
			mesh.layers.mask = source.layers.mask;
			mesh.matrixAutoUpdate = false;
			matrices.forEach((matrix, index) => mesh.setMatrixAt(index, matrix));
			mesh.instanceMatrix.needsUpdate = true;
			mesh.computeBoundingSphere();
			track.add(mesh);
			instancedMeshes++;
			instances += matrices.length;
		}
		track.userData.batching = { sourceMeshes, drawObjects: track.children.length, instancedMeshes, instances };
		return track;
	};
})();
