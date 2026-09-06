#!/usr/bin/env python3
"""Extract the TMNF visuals used by registered tracks into viewer-ready glTF."""

import argparse
import json
import math
import re
import shutil
import struct
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TOOL_PROJECT = ROOT / "tools/extract_game_assets/GbxTool/GbxTool.csproj"
NUGET_CONFIG = ROOT / "tools/extract_game_assets/NuGet.config"
CACHE = ROOT / "third_party/game_assets_cache"
DOTNET = ROOT / "third_party/dotnet/dotnet"
MODEL_ALIASES = {
    "StadiumRoadMainStartLine": "StadiumRoadMainStart",
    "StadiumRoadMainFinishLine": "StadiumRoadMainFinish",
    "StadiumRoadMainStartFinishLine": "StadiumRoadMainMultiLap",
}
DIRECT_SOLIDS = {
    ("StadiumCheckpointRingV", False, 0):
        "Stadium/Media/Solid/Platform/Checkpoint/RingVAir.Solid.Gbx",
    ("StadiumPlatformCheckpoint", False, 0):
        "Stadium/Media/Solid/Platform/Checkpoint/Air.Solid.Gbx",
    ("StadiumRoadMainStartLine", False, 0):
        "Stadium/Media/Solid/Road/Main/Start/Air.Solid.Gbx",
    ("StadiumRoadMainStartLine", True, 0):
        "Stadium/Media/Solid/Road/Main/Start/Ground.Solid.Gbx",
    ("StadiumRoadMainFinishLine", False, 0):
        "Stadium/Media/Solid/Road/Main/Finish/Air.Solid.Gbx",
    ("StadiumRoadMainFinishLine", True, 0):
        "Stadium/Media/Solid/Road/Main/Finish/Ground.Solid.Gbx",
    ("StadiumRoadMainTurbo", True, 0):
        "Stadium/Media/Solid/Road/Main/Turbo/Ground.Solid.Gbx",
    ("StadiumRoadMainTurbo", False, 0):
        "Stadium/Media/Solid/Road/Main/Turbo/Air.Solid.Gbx",
    ("StadiumRoadMainTurboDown", False, 0):
        "Stadium/Media/Solid/Road/Main/Turbo/DownAir.Solid.Gbx",
    ("StadiumRoadMainTurboUp", False, 0):
        "Stadium/Media/Solid/Road/Main/Turbo/UpAir.Solid.Gbx",
    ("StadiumRoadMainCheckpoint", False, 0):
        "Stadium/Media/Solid/Road/Main/Checkpoint/Air.Solid.Gbx",
    ("StadiumRoadMainCheckpoint", True, 0):
        "Stadium/Media/Solid/Road/Main/Checkpoint/Ground.Solid.Gbx",
    ("StadiumPlatformTurbo", False, 0):
        "Stadium/Media/Solid/Platform/Turbo/Air.Solid.Gbx",
    ("StadiumRoadMainStartFinishLine", False, 0):
        "Stadium/Media/Solid/Road/Main/MultiLap/Air.Solid.Gbx",
}
EXTRACT_SUFFIXES = [
    "Solid.Gbx",
    "TMEDClassic.Gbx",
    "TMEDRoad.Gbx",
    "TMEDClip.Gbx",
    "TMEDFlat.Gbx",
    "TMEDFrontier.Gbx",
    "TMEDRectAsym.Gbx",
    "Material.Gbx",
    "Texture.Gbx",
    "Shader.Gbx",
]
TEXTURE_SLOT_PRIORITY = [
    "Diffuse",
    "FenceA",
    "Blend1",
    "GrassPC0",
    "Water",
    "Advert",
    "Texture",
]
NON_COLOR_SLOTS = {
    "normal", "occlusion", "specular", "reflectsoft", "fresnel",
    "lighting", "prelightgen", "blend2", "blendi", "stripe", "mask", "clouds",
    "bump", "dudv0", "dudv1",
}
NON_COLOR_SLOT_TOKENS = {
    "cube", "env", "fresnel", "normal", "occlusion", "reflect", "specular",
}
CAR_SOLID = "Stadium/Vehicles/Media/Solid/StadiumCar.Solid.Gbx"
CAR_SKIN = "Skins/Vehicles/StadiumCar/FRA.zip"
CAR_SKIN_ENTRY = "Diffuse.dds"
# StadiumCar materials. The game binds each material's `Diffuse` slot twice,
# once as `Diffuse` and once as `Diffuse_Gloss`, so the DDS alpha channel is a
# gloss mask rather than transparency. The player skin replaces the
# `StadiumCarSkin` diffuse at runtime with the skin zip's `Diffuse.dds`.
# `metallic` approximates the material's cube-map specular term.
CAR_MATERIALS = {
    "StadiumCarSkin": {"source": "skin", "metallic": 0.0},
    "StadiumCarDetails": {"source": "Diffuse", "metallic": 0.45},
    "StadiumCarPilot": {"source": "Diffuse", "metallic": 0.0},
    # Glass has no diffuse image; the game renders it from environment cube
    # maps and fresnel ramps. Dark tinted constant instead.
    "StadiumCarGlass": {
        "baseColorFactor": [0.06, 0.08, 0.10, 0.62],
        "metallic": 0.9,
        "roughness": 0.08,
    },
}


def fail(message):
    raise SystemExit(f"extract_game_assets: {message}")


def run(*command):
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def asset_key(model, ground, variant):
    return f"{model}|{'ground' if ground else 'air'}|{variant}"


def asset_slug(model, ground, variant):
    return f"{safe_name(model)}-{'ground' if ground else 'air'}-v{variant}"


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def find_game_image(game_data, extracted_path):
    path = Path(extracted_path)
    try:
        relative = path.relative_to(CACHE / "Stadium")
    except ValueError:
        fail(f"texture reference escaped Stadium cache: {path}")
    candidate = game_data / relative
    if candidate.is_file():
        return candidate
    matches = [
        match for match in game_data.rglob(path.name)
        if match.is_file()
    ]
    if len(matches) != 1:
        fail(f"cannot resolve texture image {path.name}: {len(matches)} matches")
    return matches[0]


def choose_color_texture(material_textures):
    textures = []
    for item in material_textures or []:
        if not item.get("image"):
            continue
        slot = item["slot"].casefold()
        image = Path(item["image"]).name.casefold()
        if slot in NON_COLOR_SLOTS:
            continue
        if image.endswith(".bik"):
            # Bink video textures (animated signs) have no still image.
            continue
        if any(token in slot for token in NON_COLOR_SLOT_TOKENS):
            continue
        if any(token in image for token in NON_COLOR_SLOT_TOKENS):
            continue
        textures.append(item)
    by_slot = {item["slot"]: item for item in textures}
    for slot in TEXTURE_SLOT_PRIORITY:
        if slot in by_slot:
            return by_slot[slot]
    if len(textures) == 1:
        return textures[0]
    if len(textures) > 1:
        fail(f"ambiguous color texture slots: {[item['slot'] for item in textures]}")
    return None


class TextureStore:
    def __init__(self, game_data, output):
        self.game_data = game_data
        self.output = output
        self.converted = {}
        self.gloss_pairs = {}

    def convert(self, extracted_path):
        source = find_game_image(self.game_data, extracted_path)
        key = source.resolve()
        if key in self.converted:
            return self.converted[key]
        target_name = safe_name(source.stem) + ".png"
        target = self.output / target_name
        if target.exists() and target.resolve() not in self.converted.values():
            fail(f"texture output collision: {target_name}")
        with Image.open(source) as image:
            image.save(target, optimize=True)
        self.converted[key] = target
        return target

    def write_gloss_pair(self, source, stem):
        """Split a diffuse+gloss DDS into RGB base color and glTF
        metallicRoughness PNGs. Returns (base_color, metallic_roughness,
        size)."""
        key = source.resolve()
        if key in self.gloss_pairs:
            return self.gloss_pairs[key]
        base_target = self.output / f"{stem}.png"
        roughness_target = self.output / f"{stem}-mr.png"
        for target in (base_target, roughness_target):
            if target.exists():
                fail(f"texture output collision: {target.name}")
        with Image.open(source) as image:
            rgba = image.convert("RGBA")
        if rgba.getextrema()[3] == (255, 255):
            fail(f"{source.name} has no gloss alpha; not a diffuse+gloss map")
        rgba.convert("RGB").save(base_target, optimize=True)
        gloss = rgba.getchannel("A")
        # glTF metallicRoughness: G = roughness, B = metallic. Roughness is
        # the inverse of the gloss mask; metallic stays a per-material factor.
        roughness = gloss.point(lambda value: 255 - value)
        Image.merge("RGB", (
            Image.new("L", rgba.size, 0),
            roughness,
            Image.new("L", rgba.size, 255),
        )).save(roughness_target, optimize=True)
        result = (base_target, roughness_target, rgba.size)
        self.gloss_pairs[key] = result
        return result


class CarMaterials:
    """Resolve StadiumCar materials to the default skin and gloss maps."""

    def __init__(self, game_data, skin_relative, work, texture_store):
        self.texture_store = texture_store
        self.skin_relative = skin_relative
        skin_zip = game_data / skin_relative
        if not skin_zip.is_file():
            fail(f"missing StadiumCar skin {skin_zip}")
        skin_dir = work / "car-skin"
        skin_dir.mkdir()
        with zipfile.ZipFile(skin_zip) as archive:
            names = {name.casefold(): name for name in archive.namelist()}
            entry = names.get(CAR_SKIN_ENTRY.casefold())
            if entry is None:
                fail(f"{skin_zip} has no {CAR_SKIN_ENTRY}")
            archive.extract(entry, skin_dir)
        self.skin_stem = (
            f"stadiumcar-skin-{safe_name(Path(skin_relative).stem)}")
        self.skin_image = skin_dir / entry
        self.report = {}

    def resolve(self, name, material_textures):
        spec = CAR_MATERIALS.get(name)
        if spec is None:
            fail(f"unmapped StadiumCar material {name}")
        material = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [1, 1, 1, 1],
                "metallicFactor": spec["metallic"],
                "roughnessFactor": 1,
            },
            "doubleSided": True,
        }
        if "baseColorFactor" in spec:
            material["pbrMetallicRoughness"].update({
                "baseColorFactor": spec["baseColorFactor"],
                "roughnessFactor": spec["roughness"],
            })
            material["alphaMode"] = "BLEND"
            self.report[name] = {
                "baseColor": None,
                "baseColorFactor": spec["baseColorFactor"],
                "note": "constant; game shader uses env cube maps",
            }
            return material, None, None
        if spec["source"] == "skin":
            source = self.skin_image
            stem = self.skin_stem
            origin = f"{self.skin_relative}:{CAR_SKIN_ENTRY}"
        else:
            slots = {
                item["slot"]: item["image"] for item in material_textures or []
                if item.get("image")
            }
            if spec["source"] not in slots:
                fail(f"{name} has no {spec['source']} slot")
            source = find_game_image(
                self.texture_store.game_data, slots[spec["source"]])
            stem = safe_name(source.stem)
            origin = str(source.relative_to(self.texture_store.game_data))
        base, roughness, size = self.texture_store.write_gloss_pair(
            source, stem)
        self.report[name] = {
            "baseColor": base.name,
            "metallicRoughness": roughness.name,
            "source": origin,
            "size": list(size),
            "metallicFactor": spec["metallic"],
        }
        return material, base, roughness


class GltfWriter:
    def __init__(self, output_path, texture_store, car_materials=None):
        self.output_path = output_path
        self.texture_store = texture_store
        self.car_materials = car_materials
        self.binary = bytearray()
        self.material_indices = {}
        self.texture_indices = {}
        self.vertex_count = 0
        self.triangle_count = 0
        self.doc = {
            "asset": {
                "version": "2.0",
                "generator": "TMNF-C game asset pipeline",
            },
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "textures": [],
            "images": [],
            "samplers": [{
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": 10497,
                "wrapT": 10497,
            }],
            "accessors": [],
            "bufferViews": [],
            "buffers": [],
        }

    def add_view(self, data, target):
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        offset = len(self.binary)
        self.binary.extend(data)
        index = len(self.doc["bufferViews"])
        self.doc["bufferViews"].append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data),
            "target": target,
        })
        return index

    def add_accessor(self, values, width, gltf_type, target,
                     component_type=5126):
        if component_type == 5126:
            data = struct.pack(f"<{len(values)}f", *values)
        elif component_type == 5125:
            data = struct.pack(f"<{len(values)}I", *values)
        else:
            fail(f"unsupported component type {component_type}")
        accessor = {
            "bufferView": self.add_view(data, target),
            "componentType": component_type,
            "count": len(values) // width,
            "type": gltf_type,
        }
        if gltf_type in {"VEC3", "SCALAR"}:
            rows = list(zip(*(iter(values),) * width))
            accessor["min"] = [
                min(row[index] for row in rows) for index in range(width)
            ]
            accessor["max"] = [
                max(row[index] for row in rows) for index in range(width)
            ]
        index = len(self.doc["accessors"])
        self.doc["accessors"].append(accessor)
        return index

    def texture_ref(self, target):
        texture_key = target.name
        if texture_key not in self.texture_indices:
            image_index = len(self.doc["images"])
            self.doc["images"].append({
                "uri": f"../textures/{target.name}",
            })
            texture_index = len(self.doc["textures"])
            self.doc["textures"].append({
                "sampler": 0,
                "source": image_index,
            })
            self.texture_indices[texture_key] = texture_index
        return {"index": self.texture_indices[texture_key]}

    def car_material(self, name, material_textures):
        if name in self.material_indices:
            return self.material_indices[name]
        material, base, roughness = self.car_materials.resolve(
            name, material_textures)
        if base is not None:
            pbr = material["pbrMetallicRoughness"]
            pbr["baseColorTexture"] = self.texture_ref(base)
            pbr["metallicRoughnessTexture"] = self.texture_ref(roughness)
        index = len(self.doc["materials"])
        self.doc["materials"].append(material)
        self.material_indices[name] = index
        return index

    def material(self, name, material_textures):
        if self.car_materials is not None:
            if not name:
                fail("StadiumCar mesh without a material")
            return self.car_material(name, material_textures)
        name = name or "TMNF_Default"
        selected = choose_color_texture(material_textures)
        image_path = selected["image"] if selected else None
        key = (name, image_path)
        if key in self.material_indices:
            return self.material_indices[key]

        alpha = any(token in name for token in (
            "Glass", "Alpha", "Fence", "Glow", "Water"))
        base_color = [1, 1, 1, 1]
        if image_path is None:
            if "Water" in name:
                base_color = [0.18, 0.42, 0.54, 0.62]
            elif "Glass" in name:
                base_color = [0.22, 0.32, 0.38, 0.48]
            else:
                base_color = [0.7, 0.7, 0.7, 1]
        pbr = {
            "baseColorFactor": base_color,
            "metallicFactor": 0,
            "roughnessFactor": 1,
        }
        if image_path is not None:
            pbr["baseColorTexture"] = self.texture_ref(
                self.texture_store.convert(image_path))
        material = {
            "name": name,
            "pbrMetallicRoughness": pbr,
            "doubleSided": True,
        }
        if alpha:
            material["alphaMode"] = "BLEND"
        index = len(self.doc["materials"])
        self.doc["materials"].append(material)
        self.material_indices[key] = index
        return index

    def mesh(self, source, name, material_name, material_textures):
        positions = source["positions"]
        indices = source["indices"]
        if not positions or len(positions) % 3:
            fail(f"{name}: invalid positions")
        if not indices or len(indices) % 3:
            fail(f"{name}: invalid triangle indices")
        vertex_count = len(positions) // 3
        if max(indices) >= vertex_count:
            fail(f"{name}: index exceeds vertex count")
        attributes = {
            "POSITION": self.add_accessor(
                positions, 3, "VEC3", 34962),
        }
        normals = source["normals"]
        if normals:
            if len(normals) != len(positions):
                fail(f"{name}: normal count mismatch")
            attributes["NORMAL"] = self.add_accessor(
                normals, 3, "VEC3", 34962)
        texcoords = source["texCoords"]
        if texcoords:
            if len(texcoords) != vertex_count * 2:
                fail(f"{name}: UV count mismatch")
            attributes["TEXCOORD_0"] = self.add_accessor(
                texcoords, 2, "VEC2", 34962)
        mesh_index = len(self.doc["meshes"])
        self.doc["meshes"].append({
            "name": name,
            "primitives": [{
                "attributes": attributes,
                "indices": self.add_accessor(
                    indices, 1, "SCALAR", 34963, 5125),
                "material": self.material(
                    material_name, material_textures),
                "mode": 4,
            }],
        })
        self.vertex_count += vertex_count
        self.triangle_count += len(indices) // 3
        return mesh_index

    def node(self, source):
        name = source.get("name") or "[unnamed]"
        node = {"name": name}
        transform = source.get("transform")
        if transform:
            if len(transform) != 12:
                fail(f"{name}: invalid Iso4")
            node["matrix"] = [
                transform[0], transform[3], transform[6], 0,
                transform[1], transform[4], transform[7], 0,
                transform[2], transform[5], transform[8], 0,
                transform[9], transform[10], transform[11], 1,
            ]
        visual = source.get("visual")
        if visual and visual["positions"] and visual["indices"]:
            node["mesh"] = self.mesh(
                visual, name, source.get("material"),
                source.get("materialTextures"))
        index = len(self.doc["nodes"])
        self.doc["nodes"].append(node)
        children = tree_children(source)
        if children:
            node["children"] = [self.node(child) for child in children]
        return index

    def write(self, roots):
        root_indices = [self.node(root) for root in roots]
        self.doc["scenes"][0]["nodes"] = root_indices
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        self.doc["buffers"].append({
            "uri": self.output_path.with_suffix(".bin").name,
            "byteLength": len(self.binary),
        })
        self.output_path.with_suffix(".bin").write_bytes(self.binary)
        write_json(self.output_path, self.doc)
        return {
            "meshes": len(self.doc["meshes"]),
            "vertices": self.vertex_count,
            "triangles": self.triangle_count,
            "bytes": (
                self.output_path.stat().st_size
                + self.output_path.with_suffix(".bin").stat().st_size
            ),
        }


def selected_trees(block_info, ground, variant):
    rows = block_info["ground" if ground else "air"]
    if variant >= len(rows):
        return []
    trees = []
    for mobil in rows[variant]:
        if mobil["tree"]:
            trees.append(mobil["tree"])
        trees.extend(
            link["tree"] for link in mobil.get("objectLinks") or []
            if link["tree"])
    return trees


def tree_children(source):
    children = list(source.get("children") or [])
    levels = source.get("visualMip") or []
    if levels:
        finite = [
            level for level in levels if math.isfinite(level["distance"])
        ]
        selected = min(
            finite or levels, key=lambda level: level["distance"])
        children.append(selected["tree"])
    return children


def visual_bounds(roots):
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]

    def visit(source, parent_rotation, parent_translation):
        transform = source.get("transform")
        if transform:
            local_rotation = [
                transform[0:3],
                transform[3:6],
                transform[6:9],
            ]
            local_translation = transform[9:12]
        else:
            local_rotation = [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
            local_translation = [0, 0, 0]
        rotation = [
            [
                sum(parent_rotation[row][inner] * local_rotation[inner][column]
                    for inner in range(3))
                for column in range(3)
            ]
            for row in range(3)
        ]
        translation = [
            sum(parent_rotation[row][inner] * local_translation[inner]
                for inner in range(3)) + parent_translation[row]
            for row in range(3)
        ]
        visual = source.get("visual")
        if visual:
            positions = visual.get("positions") or []
            for index in range(0, len(positions), 3):
                point = positions[index:index + 3]
                world = [
                    sum(rotation[row][column] * point[column]
                        for column in range(3)) + translation[row]
                    for row in range(3)
                ]
                for axis in range(3):
                    minimum[axis] = min(minimum[axis], world[axis])
                    maximum[axis] = max(maximum[axis], world[axis])
        for child in tree_children(source):
            visit(child, rotation, translation)

    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for root in roots:
        visit(root, identity, [0, 0, 0])
    if not all(math.isfinite(value) for value in minimum + maximum):
        fail("visual tree has no finite vertices")
    return minimum, maximum


def nominal_footprint(roots):
    _, maximum = visual_bounds(roots)

    def extent(value):
        return 32 * max(1, math.ceil((value - 1.0) / 32))

    return extent(maximum[0]), extent(maximum[2])


def placement_transform(block, roots):
    width, depth = nominal_footprint(roots)
    offsets = [
        (0, 0),
        (depth, 0),
        (width, depth),
        (0, width),
    ]
    offset_x, offset_z = offsets[block["direction"]]
    return {
        "position": [
            block["coord"][0] * 32 + offset_x,
            block["coord"][1] * 8,
            block["coord"][2] * 32 + offset_z,
        ],
        "rotationY": -block["direction"] * math.pi / 2,
        "gridOffset": [offset_x, 0, offset_z],
        "nominalFootprint": [width, depth],
    }


def build_tool():
    if not DOTNET.is_file():
        fail("missing third_party/dotnet; run bootstrap_dotnet.sh")
    run(
        DOTNET, "build", TOOL_PROJECT, "-c", "Release",
        "--configfile", NUGET_CONFIG)
    return (
        ROOT
        / "tools/extract_game_assets/GbxTool/bin/Release/net10.0/GbxTool.dll"
    )


def parse_challenge(tool, source, output):
    run(DOTNET, tool, "challenge", source, output)
    return load_json(output)


def parse_solid(tool, source, output):
    run(DOTNET, tool, "solid", source, output)
    return load_json(output)["tree"]


def parse_block_info(tool, source, output):
    run(DOTNET, tool, "block-info", source, output)
    return load_json(output)


def verify_placement(scene_path, challenge, label):
    scene = load_json(scene_path)
    start = next(
        block for block in challenge["blocks"]
        if block["model"] in (
            "StadiumRoadMainStartLine",
            "StadiumRoadMainStartFinishLine",
        ))
    finishes = [
        block for block in challenge["blocks"]
        if block["model"] in (
            "StadiumRoadMainFinishLine",
            "StadiumRoadMainStartFinishLine",
        )]
    if not finishes:
        fail(f"{label} challenge has no finish block")
    first = scene["lap"]["ticks"][0][1:4]
    start_origin = [
        start["coord"][0] * 32,
        start["coord"][1] * 8,
        start["coord"][2] * 32,
    ]
    # A race may have several finish blocks (A12-Speed has four); the route
    # carries the one the official ghost crosses, so compare against the
    # nearest finish block.
    finish_entry = scene["route"]["finish"]["transform"]["translation"]
    finish_distance = min(
        math.dist(finish_entry, [
            finish["coord"][0] * 32,
            finish["coord"][1] * 8,
            finish["coord"][2] * 32,
        ])
        for finish in finishes)
    start_distance = math.dist(first, start_origin)
    if start_distance > 48 or finish_distance > 48:
        fail(
            f"{label} grid/collision mismatch: "
            f"start {start_distance:.3f}m, finish {finish_distance:.3f}m")
    return {
        "startDistanceMeters": round(start_distance, 6),
        "finishEntryDistanceMeters": round(finish_distance, 6),
    }


def parse_named_paths(values, option):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if (not separator or not re.fullmatch(r"[a-z][a-z0-9_-]*", name)
                or name in result):
            fail(f"{option} must be unique lowercase ID=PATH entries")
        result[name] = Path(path).resolve()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "viewer/assets/game")
    parser.add_argument(
        "--track", action="append", default=[], metavar="ID=CHALLENGE")
    parser.add_argument(
        "--scene-source", action="append", default=[], metavar="ID=SCENE")
    parser.add_argument(
        "--car-skin", default=CAR_SKIN, metavar="GAMEDATA_RELATIVE_ZIP",
        help="StadiumCar skin zip relative to GameData (default: the skin "
             "a fresh TMNF profile selects)")
    args = parser.parse_args()
    game_dir = args.game_dir.resolve()
    game_data = game_dir / "GameData"
    packs = game_dir / "Packs"
    campaign = game_data / "Tracks/Campaigns/Nations/White"
    if not packs.is_dir() or not campaign.is_dir():
        fail(f"{game_dir} is not a TMNF installation")

    tool = build_tool()
    if CACHE.exists():
        shutil.rmtree(CACHE)
    CACHE.mkdir(parents=True)
    work = CACHE / "work"
    work.mkdir()
    challenge_paths = {
        "a01": campaign / "A01-Race.Challenge.Gbx",
        "a10": campaign / "A10-Acrobatic.Challenge.Gbx",
    }
    challenge_paths.update(parse_named_paths(args.track, "--track"))
    scene_sources = {
        "a01": ROOT / "viewer/scenes/policy_lap.json",
        "a10": ROOT / "viewer/scenes/a10_turbo.json",
    }
    scene_sources.update(parse_named_paths(
        args.scene_source, "--scene-source"))
    if challenge_paths.keys() != scene_sources.keys():
        fail("--track and --scene-source IDs must match")
    for name, path in (*challenge_paths.items(), *scene_sources.items()):
        if not path.is_file():
            fail(f"missing {name} input: {path}")
    challenges = {
        name: parse_challenge(tool, path, work / f"{name}.json")
        for name, path in challenge_paths.items()
    }

    # Pak entries are stored under name hashes. GBX.NET only resolves names it
    # meets while scanning, so every block info the challenges may need is
    # offered as a candidate name under each block-info kind.
    candidates = work / "block-info-candidates.txt"
    kinds = [suffix[len("TMED"):-len(".Gbx")] for suffix in EXTRACT_SUFFIXES
             if suffix.startswith("TMED")]
    models = {
        MODEL_ALIASES.get(block["model"], block["model"])
        for challenge in challenges.values() for block in challenge["blocks"]
    }
    candidates.write_text("".join(
        f"{model}.TMED{kind}.Gbx\n" for model in sorted(models) for kind in kinds))
    run(DOTNET, tool, "extract", packs, CACHE, *EXTRACT_SUFFIXES,
        "--candidates", candidates)

    info_files = {}
    for path in (CACHE / "Stadium/Stadium/ConstructionBlockInfo").rglob(
            "*.Gbx"):
        model = path.name.split(".TMED", 1)[0]
        if model in info_files:
            fail(f"duplicate block info for {model}")
        info_files[model] = path

    requested = defaultdict(set)
    for challenge in challenges.values():
        for block in challenge["blocks"]:
            requested[block["model"]].add(
                (block["ground"], block["variant"]))

    block_infos = {}
    missing_infos = []
    for model in sorted(requested):
        info_model = MODEL_ALIASES.get(model, model)
        if info_model not in info_files:
            # The pak stores this block info under a name hash GBX.NET does
            # not resolve even when offered the name (StadiumFabricPillar
            # CornerOut). Its variants are recorded as unrendered, like the
            # visual-less grass clip, instead of aborting the whole track.
            missing_infos.append(model)
            continue
        block_infos[model] = parse_block_info(
            tool, info_files[info_model], work / f"{safe_name(model)}.json")
    if missing_infos:
        print("block infos not in the paks (rendered as nothing): "
              + ", ".join(missing_infos))

    destination = args.output.resolve()
    output = destination.with_name(destination.name + ".staging")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    models_dir = output / "models"
    textures_dir = output / "textures"
    scenes_dir = output / "scenes"
    models_dir.mkdir(parents=True, exist_ok=True)
    textures_dir.mkdir(exist_ok=True)
    scenes_dir.mkdir(exist_ok=True)
    texture_store = TextureStore(game_data, textures_dir)
    manifest_assets = {}
    visual_roots = {}
    failures = []

    for model in sorted(requested):
        for ground, variant in sorted(requested[model]):
            key = asset_key(model, ground, variant)
            if model not in block_infos:
                manifest_assets[key] = {"visual": None, "reason": "no block info"}
                failures.append(key)
                continue
            roots = selected_trees(block_infos[model], ground, variant)
            direct = DIRECT_SOLIDS.get((model, ground, variant))
            if not roots and direct:
                source = CACHE / "Stadium" / direct
                roots = [
                    parse_solid(
                        tool, source,
                        work / f"{asset_slug(model, ground, variant)}.json")
                ]
            if not roots:
                manifest_assets[key] = {"visual": None}
                failures.append(key)
                continue
            visual_roots[key] = roots
            slug = asset_slug(model, ground, variant)
            gltf_path = models_dir / f"{slug}.gltf"
            stats = GltfWriter(gltf_path, texture_store).write(roots)
            if stats["triangles"] == 0:
                fail(f"{key} produced no triangles")
            manifest_assets[key] = {
                "visual": f"models/{gltf_path.name}",
                **stats,
            }

    car_tree = parse_solid(tool, CACHE / CAR_SOLID, work / "stadium-car.json")
    high_lod = [
        child for child in car_tree["children"] if child["name"] == "1"
    ]
    if len(high_lod) != 1:
        fail(f"expected one StadiumCar high LOD, found {len(high_lod)}")
    car_path = models_dir / "stadium-car.gltf"
    car_materials = CarMaterials(
        game_data, args.car_skin, work, texture_store)
    car_stats = GltfWriter(
        car_path, texture_store, car_materials).write(high_lod)
    car_wheels = {"1FLWheel", "1FRWheel", "1RLWheel", "1RRWheel"}
    car_nodes = {node["name"] for node in load_json(car_path)["nodes"]}
    if not car_wheels <= car_nodes:
        fail(f"StadiumCar glTF lacks wheel nodes {car_wheels - car_nodes}")

    scene_manifest = {}
    for name, challenge in challenges.items():
        placements = []
        for block in challenge["blocks"]:
            key = asset_key(
                block["model"], block["ground"], block["variant"])
            if manifest_assets[key]["visual"] is None:
                continue
            transform = placement_transform(block, visual_roots[key])
            placements.append({
                "blockIndex": block["index"],
                "asset": key,
                "model": block["model"],
                "coord": block["coord"],
                "direction": block["direction"],
                "ground": block["ground"],
                "variant": block["variant"],
                "subVariant": block["subVariant"],
                **transform,
            })
        scene_path = scenes_dir / f"{name}.json"
        write_json(scene_path, {
            "format": "tmnf-game-visual-scene",
            "version": 2,
            "placements": placements,
        })
        scene_manifest[name] = {
            "url": f"scenes/{scene_path.name}",
            "placements": len(placements),
            "placementCheck": verify_placement(
                scene_sources[name], challenge, name),
        }

    write_json(output / "manifest.json", {
        "format": "tmnf-game-visual-manifest",
        "version": 2,
        "placementConvention": {
            "gridMeters": [32, 8, 32],
            "rotation": "negative-y-quarter-turns",
            "pivot": "nominal-footprint-min-corner",
        },
        "assets": manifest_assets,
        "car": {
            "visual": f"models/{car_path.name}",
            **car_stats,
            "skin": args.car_skin,
            "materials": car_materials.report,
        },
        "scenes": scene_manifest,
        "unrendered": failures,
    })
    backup = destination.with_name(destination.name + ".backup")
    if backup.exists():
        fail(f"stale asset backup exists: {backup}")
    if destination.exists():
        destination.rename(backup)
    output.rename(destination)
    if backup.exists():
        shutil.rmtree(backup)
    print(
        f"converted {len(manifest_assets) - len(failures)}/"
        f"{len(manifest_assets)} block variants, "
        f"{len(texture_store.converted)} block textures, "
        f"{len(texture_store.gloss_pairs)} car diffuse+gloss pairs "
        f"from {args.car_skin}")


if __name__ == "__main__":
    main()
