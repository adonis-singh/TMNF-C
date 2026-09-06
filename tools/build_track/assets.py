"""Game asset access for the offline track builder.

Assets are read from a cache directory that holds the decrypted GBX files
extracted from the game's NadeoPak archives plus one JSON dump per GBX
(produced by tools/build_track/GbxDump). `ensure_cache` builds it when it is
missing. All lookups go through `Assets`, which memoizes the parsed JSON.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import cached_property

from fp import box_from_bytes, iso_from_bytes

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# NadeoPak directory of the installed game: one <Environment>.pak per
# collection plus Game.pak/Resource.pak. TMNF ships Stadium only; a TMUF
# install has all seven environments in the same layout.
DEFAULT_PACKS = os.path.join(REPO, "oracle", "wineprefix", "drive_c", "TmNationsForever", "Packs")
# Decrypted assets + JSON dumps live under the checkout that runs the build
# (every path inside the cache is relative to it), or wherever
# TMNF_BUILD_TRACK_CACHE points.
CACHE_ROOT = os.environ.get("TMNF_BUILD_TRACK_CACHE") or os.path.join(REPO, "third_party", "build_track_cache")
GBXDUMP_DLL = os.path.join(HERE, "GbxDump", "bin", "Release", "net10.0", "GbxDump.dll")
DOTNET_INSTALL_HINT = ("install the .NET 10 SDK with third_party/dotnet-install.sh --install-dir "
                       "third_party/dotnet (or set TMNF_DOTNET / put dotnet on PATH)")


def find_dotnet() -> str | None:
    """The dotnet host GbxDump runs on: $TMNF_DOTNET, the repo-local SDK,
    PATH, then ~/.dotnet."""
    import shutil
    candidates = [os.environ.get("TMNF_DOTNET"),
                  os.path.join(REPO, "third_party", "dotnet", "dotnet"),
                  shutil.which("dotnet"),
                  os.path.expanduser("~/.dotnet/dotnet")]
    for path in candidates:
        if path and os.access(path, os.X_OK):
            return path
    return None
CLASS_PARENTS = os.path.join(HERE, "tmnf_class_parents.txt")
UNWRAP_CLASS_IDS = os.path.join(HERE, "tmnf_unwrap_class_ids.txt")

# Everything the collision build needs out of the paks.
ASSET_SUFFIXES = [
    ".Solid.Gbx", ".Material.Gbx", ".TMCollection.Gbx", ".TMZoneFlat.Gbx",
    ".TMZoneFrontier.Gbx", ".TMDecoration.Gbx", ".TMDecorationSize.Gbx",
    ".TMEDClassic.Gbx", ".TMEDClip.Gbx", ".TMEDFlat.Gbx", ".TMEDFrontier.Gbx",
    ".TMEDPylon.Gbx", ".TMEDRectAsym.Gbx", ".TMEDRoad.Gbx", ".Scene3d.Gbx",
    ".TMTerrainModifier.Gbx", ".DecoSolid.Gbx", ".Mobil.Gbx", ".TMDecorationMood.Gbx",
    ".GameSkin.gbx",
]

# CGameCtnBlockInfo+0x7c, the EBlockType the placement code switches on.
BLOCK_TYPES = {
    "CGameCtnBlockInfoFlat": 0,
    "CGameCtnBlockInfoFrontier": 1,
    "CGameCtnBlockInfoClassic": 2,
    "CGameCtnBlockInfoRoad": 3,
    "CGameCtnBlockInfoClip": 4,
    "CGameCtnBlockInfoRectAsym": 5,
    "CGameCtnBlockInfoSlope": 6,
    "CGameCtnBlockInfoPylon": 7,
}


class BuildError(Exception):
    pass


def gbxdump(*args: str) -> None:
    dotnet = find_dotnet()
    if dotnet is None:
        raise BuildError(f"dotnet not found: {DOTNET_INSTALL_HINT}")
    if not os.path.exists(GBXDUMP_DLL):
        subprocess.run(
            [dotnet, "build", "-c", "Release", os.path.join(HERE, "GbxDump")],
            check=True, stdout=sys.stderr)
    env = dict(os.environ, TMNF_CLASS_PARENTS=CLASS_PARENTS, TMNF_UNWRAP_CLASS_IDS=UNWRAP_CLASS_IDS,
               DOTNET_CLI_TELEMETRY_OPTOUT="1")
    subprocess.run([dotnet, GBXDUMP_DLL, *args], check=True, env=env, stdout=sys.stderr)


def cache_dir_for(packs: str) -> str:
    """One cache per game install, named after the install directory."""
    packs = os.path.abspath(packs)
    return os.path.join(CACHE_ROOT, os.path.basename(os.path.dirname(packs)))


# Bump when GbxDump's JSON shape changes; stale caches are rebuilt.
CACHE_VERSION = "3"


def cache_complete(cache: str) -> bool:
    marker = os.path.join(cache, "json", ".complete")
    return os.path.exists(marker) and open(marker).read() == CACHE_VERSION


def ensure_cache(cache: str, packs: str) -> None:
    """Extract and dump the game assets once. The marker file records that the
    tree is complete and which dump format it holds; anything else is rebuilt
    from scratch."""
    if cache_complete(cache):
        return
    if not os.path.isdir(packs):
        raise BuildError(f"pak directory not found: {packs}")
    os.makedirs(cache, exist_ok=True)
    candidates = os.path.join(cache, "candidates.txt")
    if not os.path.exists(candidates):
        open(candidates, "w").close()
    print(f"build_track: extracting paks from {packs} into {cache}", file=sys.stderr)
    gbxdump("extract", packs, cache, candidates, *ASSET_SUFFIXES)
    print("build_track: dumping GBX files to JSON", file=sys.stderr)
    gbxdump("dump-tree", cache, os.path.join(cache, "json"), *ASSET_SUFFIXES)
    with open(os.path.join(cache, "json", ".complete"), "w") as fh:
        fh.write(CACHE_VERSION)


@dataclass(frozen=True)
class Surface:
    """A CPlugSurface as loaded from a solid: the GmSurfMesh geometry, the
    physics material id of every material slot and the geom's bounding box."""
    box: tuple
    materials: bytes
    vertices: bytes
    faces: bytes
    nodes: bytes
    surf_type: str
    # Identity of the GmSurfMesh. A fid-parameterised load only re-creates
    # the nodes that depend on a remapped fid: the CPlugSurface (its material
    # list) is fresh, the CPlugSurfaceGeom and its mesh are shared with the
    # plain load.
    mesh_key: int


@dataclass(frozen=True)
class Remap:
    """CGameSkin::AddRemappingParams (0x006FCAA0): every fid the skin names
    resolves to <pak>/<folder>/<name>.Material.Gbx while the block's mobil is
    loaded. Instances are hashable so they can key the tree/surface caches."""
    folder: str
    names: frozenset
    label: str


class Tree:
    """Runtime CPlugTree. Instances are per mobil clone: placement mutates
    flags and location the way the game does."""
    __slots__ = ("name", "flags", "loc", "surface", "children", "has_visual")

    def __init__(self, name, flags, loc, surface, children, has_visual):
        self.name = name
        self.flags = flags
        self.loc = loc
        self.surface = surface
        self.children = children
        self.has_visual = has_visual

    def clone(self) -> "Tree":
        return Tree(self.name, self.flags, self.loc, self.surface,
                    [c.clone() for c in self.children], self.has_visual)

    def hide_invalid(self) -> int:
        """CPlugTree::HideInvalidTrees (0x00848E40)."""
        if not (self.flags & 8):
            return 0
        valid = 0
        for child in self.children:
            if child.hide_invalid():
                valid = 1
        if self.has_visual:
            valid = 1  # IsVisualValid: every shipped visual is valid
        if not valid:
            self.flags &= ~8
        return (self.flags >> 3) & 1


class Assets:
    def __init__(self, packs: str = DEFAULT_PACKS, cache: str | None = None):
        self.packs = os.path.abspath(packs)
        self.cache = os.path.abspath(cache or cache_dir_for(packs))
        self.json_root = os.path.join(self.cache, "json")
        self._json: dict[str, dict] = {}
        self._trees: dict[tuple, Tree] = {}
        self._surfaces: dict[tuple, Surface] = {}
        self._remaps: dict[str, Remap] = {}

    # -- raw JSON -----------------------------------------------------------
    def rel(self, path: str) -> str:
        """Cache-relative path of an asset. The JSON dumps reference files
        relative to the cache root; paths built here may be absolute."""
        if not os.path.isabs(path):
            return os.path.normpath(path)
        path = os.path.abspath(path)
        if not path.startswith(self.cache + os.sep):
            raise BuildError(f"asset outside cache: {path}")
        return path[len(self.cache) + 1:]

    def load(self, gbx_path: str) -> dict:
        rel = self.rel(gbx_path)
        if rel not in self._json:
            json_path = os.path.join(self.json_root, rel + ".json")
            if not os.path.exists(json_path):
                raise BuildError(f"missing asset dump {json_path}{self.extract_failure(rel)}")
            with open(json_path) as fh:
                self._json[rel] = json.load(fh)["node"]
        return self._json[rel]

    def extract_failure(self, rel: str) -> str:
        """Why GbxDump could not decode `rel` out of the paks, if that is the cause."""
        path = os.path.join(self.cache, "extract_failures.txt")
        if not os.path.exists(path):
            return ""
        for line in open(path):
            if line.startswith(rel + ": "):
                return " (extraction failed: " + line.rstrip()[len(rel) + 2:] + ")"
        return ""

    def find(self, suffix: str) -> str:
        """Locate a single GBX in the cache by path suffix."""
        matches = []
        for root, _, files in os.walk(self.json_root):
            for name in files:
                full = os.path.join(root, name)
                if full.endswith(suffix + ".json"):
                    matches.append(full)
        if len(matches) != 1:
            raise BuildError(f"{suffix}: {len(matches)} matches in {self.json_root}")
        rel = os.path.relpath(matches[0], self.json_root)[:-5]
        return os.path.join(self.cache, rel)

    # -- collection / zones / decoration ------------------------------------
    @cached_property
    def block_info_index(self) -> dict[str, str]:
        """Block info id -> GBX path, over every dumped CGameCtnBlockInfo*."""
        index: dict[str, str] = {}
        for root, _, files in os.walk(self.json_root):
            for name in files:
                if ".TMED" not in name or not name.endswith(".json"):
                    continue
                full = os.path.join(root, name)
                with open(full) as fh:
                    doc = json.load(fh)
                if not doc["class"].startswith("CGameCtnBlockInfo"):
                    continue
                rel = os.path.relpath(full, self.json_root)[:-5]
                gbx = os.path.join(self.cache, rel)
                self._json[rel] = doc["node"]
                ident = doc["node"]["id"]
                if ident in index and index[ident] != gbx:
                    raise BuildError(f"duplicate block info id {ident}")
                index[ident] = gbx
        return index

    def block_info(self, ident: str) -> dict:
        path = self.block_info_index.get(ident)
        if path is None:
            raise BuildError(f"unknown block {ident}")
        info = self.load(path)
        info["_path"] = self.rel(path)
        return info

    def block_info_at(self, path: str) -> dict:
        info = self.load(path)
        info["_path"] = self.rel(path)
        return info

    def collection(self, name: str) -> dict:
        return self.load(self.find(f"{name}.TMCollection.Gbx"))

    def zone(self, path: str) -> dict:
        z = self.load(path)
        z["_path"] = self.rel(path)
        return z

    def decoration(self, collection: str, ident: str) -> dict:
        return self.load(self.find(f"{collection}Base32x32{ident}.TMDecoration.Gbx"))

    def terrain_modifier(self, path: str) -> Remap:
        """CGameCtnDecorationTerrainModifier -> its material substitution."""
        rel = self.rel(path)
        if rel not in self._remaps:
            m = self.load(path)
            skin = m["remappingNode"]
            if skin is None:
                raise BuildError(f"{rel}: terrain modifier has no CGameSkin")
            for f in skin["fids"]:
                if f["classId"] != 0x09079000:
                    raise BuildError(f"{rel}: remaps non-material fid {f['name']} ({f['classId']:08X})")
            self._remaps[rel] = Remap(
                folder=m["remapFolder"].replace("\\", "/").strip("/"),
                names=frozenset(f["name"].lower() for f in skin["fids"]),
                label=m["idName"])
        return self._remaps[rel]

    # -- solids -------------------------------------------------------------
    def remapped_material(self, material: dict, remap: Remap | None) -> str | None:
        """Path of the material the fid remap substitutes for this slot, or
        None when the slot is untouched."""
        if remap is None or material["file"] is None:
            return None
        rel = self.rel(material["file"])
        name = os.path.basename(rel)
        if name.split(".")[0].lower() not in remap.names:
            return None
        # remapFolder is relative to the pak root, i.e. the first cache component.
        target = os.path.join(self.cache, rel.split("/")[0], remap.folder, name)
        if not os.path.exists(os.path.join(self.json_root, self.rel(target) + ".json")):
            # The skin lists the name but this modifier ships no such file
            # (Fabric has no StadiumFabricFloor); the original fid stands.
            return None
        return target

    def material_surface_id(self, material: dict, remap: Remap | None) -> int:
        """Physics id of one CPlugSurface material slot, after the fid remap
        that was active while the solid loaded."""
        target = self.remapped_material(material, remap)
        if target is None:
            return material["materialSurfaceId"]
        return self.load(target)["surfaceId"]

    def effective_remap(self, tree: dict, remap: Remap | None) -> Remap | None:
        """A parameterised load only yields a distinct node when some fid in
        the file actually resolves differently; otherwise the fid cache hands
        back the plain node (and its GmSurfMesh)."""
        if remap is None:
            return None
        stack = [tree]
        while stack:
            node = stack.pop()
            if node["surface"]:
                for m in node["surface"]["materials"]:
                    if self.remapped_material(m, remap) is not None:
                        return remap
            stack.extend(node["children"])
            stack.extend(level["tree"] for level in node["mipLevels"] or [])
        return None

    def surface(self, node: dict, remap: Remap | None) -> Surface:
        if not any(self.remapped_material(m, remap) is not None for m in node["materials"]):
            remap = None
        key = (id(node), remap)
        if key not in self._surfaces:
            mesh = node["mesh"]
            if mesh is None:
                raise BuildError(f"surface {node['geomName']!r} has no mesh ({node['surfType']})")
            materials = bytes(self.material_surface_id(m, remap) for m in node["materials"])
            self._surfaces[key] = Surface(
                box=box_from_bytes(base64.b64decode(node["box"])),
                materials=materials,
                vertices=base64.b64decode(mesh["vertices"]),
                faces=base64.b64decode(mesh["faces"]),
                nodes=base64.b64decode(mesh["nodes"]),
                surf_type=node["surfType"],
                mesh_key=id(node))
        return self._surfaces[key]

    def tree_from_json(self, node: dict, remap: Remap | None) -> Tree:
        loc = iso_from_bytes(base64.b64decode(node["location"])) if node["location"] else None
        surface = self.surface(node["surface"], remap) if node["surface"] else None
        children = [self.tree_from_json(c, remap) for c in node["children"]]
        # CPlugTreeVisualMip::GetChildCount/GetChild (0x0086AAC0/0x0086AEA0)
        # expose the mip level trees after the regular children, and every
        # recursive walk (HideInvalidTrees, AddStaticSurfacesFromTree) goes
        # through those virtuals.
        children += [self.tree_from_json(level["tree"], remap) for level in node["mipLevels"] or []]
        return Tree(node["name"], node["flags"], loc, surface, children, node["hasVisual"])

    def solid_tree(self, solid_node: dict, label: str, mobil_file: str | None,
                   remap: Remap | None) -> Tree:
        """Prototype tree of a CPlugSolid (inline or via its .Solid.Gbx). The
        game caches loaded files by fid, and a remap changes the material fids,
        so each (file, remap) pair is a distinct load."""
        if solid_node is None:
            raise BuildError(f"{label}: mobil has no solid")
        path = solid_node["treeFile"]
        if path is None:
            # Tree embedded in the mobil: the game loads a .Mobil.Gbx once and
            # shares it between every block info that references it, so key
            # by file; a mobil inlined in its block info is private to it.
            if solid_node["tree"] is None:
                raise BuildError(f"{label}: solid has neither tree nor tree file")
            remap = self.effective_remap(solid_node["tree"], remap)
            key = (f"mobil:{self.rel(mobil_file)}" if mobil_file else f"inline:{id(solid_node)}", remap)
            if key not in self._trees:
                self._trees[key] = self.tree_from_json(solid_node["tree"], remap)
            return self._trees[key]
        # The game loads each .Solid.Gbx once and shares its GmSurfMesh data
        # between every mobil that references it.
        rel = self.rel(path)
        solid = self.load(path)
        if solid["tree"] is None:
            raise BuildError(f"{label}: {rel} has no tree")
        remap = self.effective_remap(solid["tree"], remap)
        key = (rel, remap)
        if key not in self._trees:
            self._trees[key] = self.tree_from_json(solid["tree"], remap)
        return self._trees[key]

    def mobil_tree(self, mobil: dict, label: str, mobil_file: str | None = None,
                   remap: Remap | None = None) -> Tree:
        """A fresh clone of a CSceneMobil's tree (CSceneMobil vtbl+0xAC)."""
        return self.solid_tree(mobil["solidNode"], label, mobil_file, remap).clone()


def iso_field(node: dict, key: str):
    raw = node.get(key)
    return iso_from_bytes(base64.b64decode(raw)) if raw else None
