// Compare batched geometry with the original cloned hierarchy, without WebGL.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const context = vm.createContext({ console });
vm.runInContext(fs.readFileSync(path.join(__dirname, "vendor/three.min.js"), "utf8"), context);
vm.runInContext(fs.readFileSync(path.join(__dirname, "track_instances.js"), "utf8"), context);
const THREE = context.THREE;
const template = new THREE.Group();
const parent = new THREE.Group();
parent.position.set(1, 2, 3);
parent.rotation.y = 0.3;
template.add(parent);
const geometry = new THREE.BoxGeometry(2, 3, 4);
const opaque = new THREE.MeshBasicMaterial();
const glass = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.5 });
const solid = new THREE.Mesh(geometry, opaque);
parent.add(solid);
const transparent = new THREE.Mesh(geometry, glass);
transparent.position.x = 4;
parent.add(transparent);
const mirrored = new THREE.Mesh(geometry, opaque);
mirrored.scale.x = -1;
mirrored.position.z = 8;
parent.add(mirrored);
const hidden = new THREE.Group();
hidden.visible = false;
hidden.add(new THREE.Mesh(geometry, opaque));
parent.add(hidden);
const placements = [
    { asset: "block", position: [0, 0, 0], rotationY: 0 },
    { asset: "block", position: [32, 0, 0], rotationY: Math.PI / 2 },
    { asset: "block", position: [512, 0, 0], rotationY: Math.PI },
];
const reference = new THREE.Group();
for (const placement of placements) {
    const clone = template.clone(true);
    clone.position.set(...placement.position);
    clone.rotation.y = placement.rotationY;
    reference.add(clone);
}
const batched = context.buildTrackInstances(new Map([["block", template]]), placements);
function vertices(root) {
    root.updateMatrixWorld(true);
    const rows = [];
    root.traverseVisible(mesh => {
        if (!mesh.isMesh) return;
        const local = new THREE.Matrix4();
        for (let i = 0; i < (mesh.isInstancedMesh ? mesh.count : 1); ++i) {
            if (mesh.isInstancedMesh) mesh.getMatrixAt(i, local);
            else local.identity();
            const matrix = new THREE.Matrix4().multiplyMatrices(mesh.matrixWorld, local);
            const positions = mesh.geometry.getAttribute("position");
            for (let j = 0; j < positions.count; ++j) {
                const point = new THREE.Vector3().fromBufferAttribute(positions, j).applyMatrix4(matrix);
                rows.push([mesh.material.uuid, ...point.toArray()]);
            }
        }
    });
    return rows.sort((a, b) => a[0].localeCompare(b[0]) || a[1] - b[1] || a[2] - b[2] || a[3] - b[3]);
}
const expected = vertices(reference), actual = vertices(batched);
assert.equal(actual.length, expected.length);
for (let i = 0; i < actual.length; ++i) {
    assert.equal(actual[i][0], expected[i][0]);
    for (let axis = 1; axis <= 3; ++axis)
        assert.ok(Math.abs(actual[i][axis] - expected[i][axis]) < 1e-4, `vertex ${i}, axis ${axis}`);
}
assert.equal(batched.userData.batching.instances, 2);
assert.equal(batched.children.filter(mesh => mesh.material === glass).length, 3);
assert.ok(batched.children.filter(mesh => mesh.isInstancedMesh).every(mesh => !mesh.material.transparent));
console.log(`track batching: ${actual.length} transformed vertices preserved; glass, mirrors, visibility and tile culling verified`);
