"""Reproduces CGameCtnChallenge::InitChallengeData and the scene population
that follows it, up to the ordered list of static corpora the collision
manager sees. Function addresses refer to TmForever.exe (TMNF 2.11.26).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

import fp
from assets import Assets, BLOCK_TYPES, BuildError, Remap, Tree

FLAT, FRONTIER, CLASSIC, ROAD, CLIP, PYLON = 0, 1, 2, 3, 4, 7

# CGameCtnFieldUnit terrain kinds (ETerrain).
T_TERRAIN, T_GROUND, T_AIR = 0, 1, 3


def neighbour(coord, direction):
    """CGameCtnUtils::GetNeighbourCoord (0x00658800)."""
    x, y, z = coord
    return ((x, y, z + 1), (x - 1, y, z), (x, y, z - 1), (x + 1, y, z))[direction]


def opposed(direction):
    return (direction + 2) & 3


@dataclass
class Mobil:
    root: Tree
    static: bool = True
    label: str = ""


# ESolidQuality, from CGameCtnApp::ChallengeCreateSceneGraph (0x005FE560):
# the warp decorator runs with High (2) when the display config's quality
# level (CSystemConfig+0x170) is >= 2 and Low (0) otherwise.
SOLID_QUALITY = {"low": 0, "high": 2}


def cond(bool_cond: int, quality: int) -> bool:
    """GetBoolFromCond (0x00898310): CPlugDecoratorTree::EBoolCond vs ESolidQuality."""
    return {0: False, 1: quality == 0, 2: quality <= 1, 3: quality == 1,
            4: quality >= 1, 5: quality == 2, 6: True}[bool_cond]


def find_tree(root: Tree, name: str, parent: Tree | None = None):
    """CPlugSolid::GetPlugFromId: first tree with that id, depth first."""
    if root.name == name:
        return root, parent
    for child in root.children:
        hit = find_tree(child, name, root)
        if hit is not None:
            return hit
    return None


def decorate_solid(root: Tree, decorators: list, quality: int) -> None:
    """CPlugDecoratorSolid::DecorateSolid (0x00898C00) on a mobil's tree. Only
    the flag effects matter here: exist removes the tree from its parent,
    visible drives bit 3, shadow caster bit 14 and collidable bit 7 (which is
    also propagated up to the root)."""
    for d in decorators:
        hit = find_tree(root, d["treeId"])
        if hit is None:
            continue
        tree, parent = hit
        if d["material"] or d["treeLight"]:
            raise BuildError(f"decorator {d['treeId']!r}: material/light overrides are not supported")
        if not cond(d["existCond"], quality):
            if parent is None:
                raise BuildError(f"decorator {d['treeId']!r} removes the root tree")
            parent.children.remove(tree)
            continue
        # The shadow condition is read from ShadowCasterCond only once the
        # global at 0x00d1696c is set (after the first scene graph of the
        # session); before that it re-uses VisibleCond. Refuse to guess.
        if d["shadowCasterCond"] != d["visibleCond"]:
            raise BuildError(f"decorator {d['treeId']!r}: shadow and visible conditions differ")
        visible = 8 if cond(d["visibleCond"], quality) else 0
        shadow = 0x4000 if cond(d["shadowCasterCond"], quality) else 0
        tree.flags = tree.flags & ~0x4008 | visible | shadow
        if d["visibleApplyOnChilds"]:
            for t in iter_tree(tree):
                t.flags = t.flags & ~8 | visible
        if d["shadowCasterApplyOnChilds"]:
            for t in iter_tree(tree):
                t.flags = t.flags & ~0x4000 | shadow
        if d["transformVisualToSurface"]:
            raise BuildError(f"decorator {d['treeId']!r}: TransformVisualToSurface is not supported")
        collidable = cond(d["collidableCond"], quality)
        tree.flags = tree.flags & ~0x80 | (0x80 if collidable else 0)
        if collidable:
            for ancestor in path_to(root, tree):
                ancestor.flags |= 0x80


def iter_tree(tree: Tree):
    yield tree
    for child in tree.children:
        yield from iter_tree(child)


def path_to(root: Tree, target: Tree) -> list:
    if root is target:
        return []
    for child in root.children:
        if child is target:
            return [root]
        below = path_to(child, target)
        if below is not None:
            return [root] + below
    return None


@dataclass
class Unit:
    block: "Block"
    info: dict
    coord: tuple
    clips: list  # 4 block info dicts or None, indexed by absolute direction


@dataclass
class FieldUnit:
    terrain: int
    unit: Unit | None = None


@dataclass
class Block:
    index: int
    name: str
    coord: tuple
    direction: int
    flags: int
    info: dict
    units: list = field(default_factory=list)
    mobil: Mobil | None = None
    extra_mobils: list = field(default_factory=list)
    terrain_modifier: str | None = None

    @property
    def ground(self) -> bool:
        return bool(self.flags & 0x1000)

    @property
    def variant(self) -> int:
        return self.flags & 0x3F

    @property
    def sub_variant(self) -> int:
        return (self.flags >> 6) & 0x3F

    @property
    def type(self) -> int:
        return BLOCK_TYPES[self.info["type"]]

    def __repr__(self):
        return f"{self.name}@{self.coord} dir={self.direction} flags={self.flags:#x}"


def unit_infos(info: dict, ground: bool) -> list:
    """CGameCtnBlockInfo::GetBlockUnitInfo family (+0xEC ground, +0xF4 air)."""
    return (info["groundUnits"] if ground else info["airUnits"]) or []


def info_size(info: dict, ground: bool) -> tuple:
    """CGameCtnBlockInfo::UpdateSize (0x00661630)."""
    sx = sy = sz = 1
    for u in unit_infos(info, ground):
        ox, oy, oz = u["relativeOffset"]
        sx, sy, sz = max(sx, ox + 1), max(sy, oy + 1), max(sz, oz + 1)
    return sx, sy, sz


def rotated_offset(info: dict, unit_index: int, direction: int, ground: bool) -> tuple:
    """CGameCtnBlockInfo::GetRotatedOffset (0x00661B30)."""
    u = unit_infos(info, ground)[unit_index]
    ox, oy, oz = u["relativeOffset"]
    if BLOCK_TYPES[info["type"]] in (FLAT, FRONTIER):
        oy = 0
    sx, _, sz = info_size(info, ground)
    if direction == 0:
        return ox, oy, oz
    if direction == 1:
        return sz - oz - 1, oy, ox
    if direction == 2:
        return sx - ox - 1, oy, sz - oz - 1
    return oz, oy, sx - ox - 1


def get_mobil(info: dict, ground: bool, variant: int, sub_variant: int) -> dict | None:
    """CGameCtnBlockInfo::GetMobil (0x00662190)."""
    rows = (info["groundMobils"] if ground else info["airMobils"]) or []
    if variant >= len(rows):
        return None
    row = rows[variant]
    if not row:
        return None
    if sub_variant >= len(row):
        sub_variant = 0
    entry = row[sub_variant]
    if entry["mobil"] is None:
        raise BuildError(f"{info['id']}: mobil {entry['file']} was not dumped")
    return entry


class Challenge:
    def __init__(self, assets: Assets, challenge: dict, quality: str = "low"):
        self.assets = assets
        self.doc = challenge
        self.quality = SOLID_QUALITY[quality]
        coll = assets.collection(challenge["collection"])
        self.collection = coll
        self.square = float(coll["squareSize"])
        self.height = float(coll["squareHeight"])
        self.default_zone = assets.zone(coll["defaultZone"])
        self.zones = {}
        for path in coll["zones"]:
            z = assets.zone(path)
            self.zones[z["common"]["zoneId"]] = z
        self.decoration = assets.decoration(challenge["collection"], challenge["decoration"]["id"])
        deco_size = self.decoration["decoSizeNode"]
        self.size = tuple(deco_size["size"])
        self.base_height = deco_size["baseHeightBase"]
        if deco_size["offsetBlockY"] or deco_size["baseHeightOffset"]:
            raise BuildError("decoration size with block Y offset is not supported")
        self.field: dict[tuple, list[FieldUnit]] = {}
        self.zone_id: dict[tuple, str] = {}
        self.zone_height: dict[tuple, int] = {}
        self.blocks: list[Block] = []
        self.add_list: list[Block] = []

    # -- play field ---------------------------------------------------------
    def editable(self, coord) -> bool:
        x, y, z = coord
        return 0 <= x < self.size[0] and 0 <= y < self.size[1] and 0 <= z < self.size[2]

    def field_unit(self, coord) -> FieldUnit | None:
        if not self.editable(coord):
            return None
        column = self.field.get((coord[0], coord[2]))
        if column is None or coord[1] >= len(column):
            return None
        return column[coord[1]]

    def create_field_unit(self, coord, terrain) -> FieldUnit:
        column = self.field.setdefault((coord[0], coord[2]), [])
        if len(column) != coord[1]:
            raise BuildError(f"field column {coord} is not contiguous")
        fu = FieldUnit(terrain)
        column.append(fu)
        return fu

    def terrain_at(self, coord) -> int:
        """CGameCtnChallenge::GetTerrainFromPlayField (0x005AB9F0)."""
        if not self.editable(coord):
            return 4
        fu = self.field_unit(coord)
        return T_AIR if fu is None else fu.terrain

    def block_at(self, coord) -> Block | None:
        fu = self.field_unit(coord)
        return fu.unit.block if fu is not None and fu.unit is not None else None

    def unit_at(self, coord) -> Unit | None:
        fu = self.field_unit(coord)
        return fu.unit if fu is not None else None

    def real_zone(self, coord) -> dict:
        """CGameCtnChallenge::GetRealZone (0x005AB850): zone of the terrain
        block at the bottom of the column."""
        b = self.block_at((coord[0], 0, coord[2]))
        if b is None:
            raise BuildError(f"no terrain block under {coord}")
        return self.zone_from_land_info(b.info)

    def zone_from_land_info(self, info: dict) -> dict:
        for z in self.zones.values():
            for key in ("blockInfoFlat", "blockInfoClip", "blockInfoRoad", "blockInfoPylon", "blockInfoFrontier"):
                if z.get(key) == info["_path"]:
                    return z
        raise BuildError(f"{info['id']} belongs to no zone")

    # -- ground test --------------------------------------------------------
    def is_block_on_ground(self, info: dict, coord, direction) -> bool:
        """CGameCtnChallenge::IsBlockOnGround (0x005AD0A0)."""
        units = unit_infos(info, True)
        if not units:
            return False
        min_y = 0
        if any(u["underground"] for u in units):
            ys = [u["relativeOffset"][1] for u in units if not u["underground"]]
            min_y = min(ys) if ys else 0
        for i, u in enumerate(units):
            if u["relativeOffset"][1] == min_y and not u["underground"]:
                off = rotated_offset(info, i, direction, True)
                c = (coord[0] + off[0], coord[1] + off[1], coord[2] + off[2])
                if self.terrain_at(c) != T_GROUND:
                    return False
        return True

    # -- InitChallengeData --------------------------------------------------
    def init(self) -> None:
        """CGameCtnChallenge::InitChallengeData (0x005AA2C0)."""
        for raw in self.doc["blocks"]:
            info = self.assets.block_info(raw["name"])
            self.blocks.append(Block(raw["index"], raw["name"], tuple(raw["coord"]),
                                     raw["direction"], raw["flags"], info))
        # Pass 1: terrain blocks fix their height and stamp the column zone.
        for b in self.blocks:
            if b.type in (FLAT, FRONTIER):
                self.check_terrain_block(b)
                zone = self.zone_from_land_info(b.info)
                if "blockInfoFrontier" in zone:
                    zone = self.zones[zone["childZoneId"]]
                x, y, z = b.coord
                self.zone_id[(x, z)] = zone["common"]["zoneId"]
                self.zone_height[(x, z)] = y
        # Pass 2: terrain blocks get field units and mobils.
        for b in self.blocks:
            if b.type in (FLAT, FRONTIER):
                self.update_field_units(b)
                self.create_mobil_for_block(b)
                self.add_list.append(b)
        # Pass 3: fill the ground level with the default zone's flat block.
        grass_info = self.assets.block_info_at(self.default_zone["blockInfoFlat"])
        grass_zone = self.default_zone["common"]["zoneId"]
        for x in range(self.size[0]):
            for z in range(self.size[2]):
                if self.block_at((x, 0, z)) is None:
                    self.create_block(grass_info, (x, self.base_height, z))
                    self.zone_id[(x, z)] = grass_zone
                    self.zone_height[(x, z)] = self.base_height
        # Pass 4: every other block in file order.
        i = 0
        while i < len(self.blocks):
            b = self.blocks[i]
            if b.type not in (FLAT, FRONTIER):
                if not self.update_field_units(b):
                    raise BuildError(f"block rejected by UpdateFieldUnits: {b}")
                self.create_mobil_for_block(b)
                self.add_list.append(b)
            i += 1
        # Pass 5: terrain modifiers over ground blocks.
        for b in list(self.blocks):
            modifier = None
            if b.ground or b.type in (FLAT, FRONTIER):
                for u in b.units:
                    m = self.column_modifier(u.coord)
                    if m is not None:
                        modifier = m
            if modifier is not None and not self.is_terrain_modifier(b.info, b.ground):
                b.terrain_modifier = modifier
                self.create_mobil_for_block(b)

    def check_terrain_block(self, b: Block) -> None:
        """CGameCtnChallenge::CheckTerrainBlock (0x005A2F80): frontier blocks
        placed at y=0 (and blocks of obsolete zones) are lifted to the zone
        height. The Stadium collection has no obsolete zones, so
        GetUpToDateZone is the identity."""
        zone = self.zone_from_land_info(b.info)
        if zone["common"]["oldZone"]:
            raise BuildError(f"{b}: obsolete zone {zone['common']['zoneId']} is not supported")
        if "blockInfoFrontier" in zone and b.coord[1] == 0:
            b.coord = (b.coord[0], zone["common"]["height"], b.coord[2])

    def create_block(self, info: dict, coord) -> Block:
        """CGameCtnChallenge::CreateBlock (0x005A9E50) for generated blocks."""
        if BLOCK_TYPES[info["type"]] != FLAT:
            raise BuildError(f"generated block of type {info['type']} is not supported")
        b = Block(-1, info["id"], coord, 0, 0, info)
        self.blocks.append(b)
        self.update_field_units(b)
        self.create_mobil_for_block(b)
        self.add_list.append(b)
        return b

    def update_field_units(self, b: Block) -> bool:
        """CGameCtnChallenge::UpdateFieldUnits (0x005A51E0)."""
        info = b.info
        ground = True
        if b.type not in (FLAT, FRONTIER):
            ground = self.is_block_on_ground(info, b.coord, b.direction)
        units = unit_infos(info, ground)
        if not units:
            ground = self.is_block_on_ground(info, b.coord, b.direction)
            units = unit_infos(info, ground)
        for i in range(len(units)):
            off = rotated_offset(info, i, b.direction, ground)
            if not self.editable((b.coord[0] + off[0], b.coord[1] + off[1], b.coord[2] + off[2])):
                return False
        if b.type in (FLAT, FRONTIER):
            x, y, z = b.coord
            for yy in range(y + 1):
                fu = self.field_unit((x, yy, z))
                if fu is None:
                    self.create_field_unit((x, yy, z), T_TERRAIN)
                else:
                    fu.terrain = T_TERRAIN
            fu = self.field_unit((x, y + 1, z))
            if fu is None:
                self.create_field_unit((x, y + 1, z), T_GROUND)
            else:
                fu.terrain = T_GROUND
            off = rotated_offset(info, 0, b.direction, ground)
            fu = self.field_unit((x + off[0], 0, z + off[2]))
            unit = self.make_unit(b, units[0], (x + off[0], y + off[1], z + off[2]))
            fu.unit = unit
            zone = self.zone_from_land_info(info)
            depth, height = zone["common"]["depth"], zone["common"]["height"]
            if depth:
                lo = max(0, y - depth - height)
                top = max(0, y - height)
                for yy in range(lo + 1, top + 1):
                    fu = self.field_unit((x, yy, z))
                    fu.unit = self.make_unit(b, units[0], (x, yy, z))
            return True
        if ground != b.ground:
            raise BuildError(f"{b}: file ground flag {b.ground} but IsBlockOnGround={ground}")
        fresh = not b.units
        for i, ui in enumerate(units):
            off = rotated_offset(info, i, b.direction, ground)
            c = (b.coord[0] + off[0], b.coord[1] + off[1], b.coord[2] + off[2])
            fu = self.field_unit(c)
            if fu is None:
                for yy in range(c[1] + 1):
                    if self.field_unit((c[0], yy, c[2])) is None:
                        self.create_field_unit((c[0], yy, c[2]), T_AIR)
                fu = self.field_unit(c)
            # Road blocks additionally derive a pylon-accept mask here; the
            # Stadium collection has no pylons so it is not modelled.
            unit = self.make_unit(b, ui, c) if fresh else b.units[i]
            fu.unit = unit
            if fu.terrain == T_GROUND:
                under = self.block_at((c[0], 0, c[2]))
                self.remove_block(under)
        return True

    def make_unit(self, b: Block, ui: dict, coord) -> Unit:
        """CGameCtnBlockUnit::CGameCtnBlockUnit (0x00700A90): unit clips are
        rotated into absolute directions."""
        clips = [None] * 4
        for i, path in enumerate(ui["clips"] or []):
            clips[(i + b.direction) & 3] = self.assets.block_info_at(path) if path else None
        unit = Unit(b, ui, coord, clips)
        b.units.append(unit)
        return unit

    def remove_block(self, b: Block) -> None:
        """CGameCtnChallenge::AddBlockToRemoveList (0x005AC870). The block's
        mobils are never in the scene at this point, so dropping it from the
        add list is the whole effect."""
        if b in self.add_list:
            self.add_list.remove(b)

    def column_modifier(self, coord) -> str | None:
        """CGameCtnChallenge::GetColumnModifierId (0x005AC250)."""
        x, _, z = coord
        y = self.zone_height[(x, z)] + 2
        while y < self.size[1]:
            unit = self.unit_at((x, y, z))
            if unit is not None and unit.info["terrainModifierId"]:
                return unit.info["terrainModifierId"]
            y += 1
        return None

    @staticmethod
    def is_terrain_modifier(info: dict, ground: bool) -> bool:
        return any(u["terrainModifierId"] for u in unit_infos(info, ground))

    # -- mobils -------------------------------------------------------------
    def block_remap(self, b: Block) -> Remap | None:
        """The CSystemFidParameters CGameCtnChallenge::CreateMobilForBlock
        (0x005A7B00) pushes before loading the block's solid: material fids
        are redirected by a CGameCtnDecorationTerrainModifier's skin.

        1. A ground block whose unit surface differs from the zone it sits on
           takes the collection's replacement modifier for that (surface,
           zone) pair, e.g. Grass units on a Dirt zone -> TerrainModifierDirt.
           A clip block borrows the surface of any neighbouring clip.
        2. Otherwise a block flagged by a terrain modifier column (Fabric
           units above it) takes the decoration's TerrainModifierCovered.
        3. Otherwise the decoration's TerrainModifierBase, when it has one.
        Only one of the three applies; TerrainModifierBase is unset in every
        TMNF decoration so the game's extra "base after replacement" push is
        a no-op here."""
        info = b.info
        if b.ground and b.type not in (FLAT, FRONTIER) and unit_infos(info, True) and b.units:
            real = self.real_zone(b.units[0].coord)["common"]["zoneId"]
            surface = unit_infos(info, True)[0]["surface"]
            if b.type == CLIP:
                surface = real
                for d in range(4):
                    unit = self.unit_at(neighbour(b.coord, d))
                    clip = unit.clips[opposed(d)] if unit is not None else None
                    if clip is not None:
                        s = unit_infos(clip, True)[0]["surface"]
                        if s != real:
                            surface = s
            if surface != real:
                for i, pair in enumerate(self.collection["zoneStrings"]):
                    if pair["baseId"] == surface and pair["replacement"] == real:
                        return self.assets.terrain_modifier(self.collection["replacementTerrainModifiers"][i])
        if b.terrain_modifier is not None and self.decoration["terrainModifierCovered"] is not None:
            return self.assets.terrain_modifier(self.decoration["terrainModifierCovered"])
        if self.decoration["terrainModifierBase"] is not None:
            return self.assets.terrain_modifier(self.decoration["terrainModifierBase"])
        return None

    def create_mobil_for_block(self, b: Block) -> None:
        """CGameCtnChallenge::CreateMobilForBlock (0x005A7B00)."""
        remap = self.block_remap(b)
        if b.type == CLIP:
            self.create_mobil_for_clip(b, remap)
        else:
            self.create_block_mobil(b, remap)

    def create_block_mobil(self, b: Block, remap: Remap | None) -> None:
        """CGameCtnBlock::CreateBlockMobil (0x0060C3B0)."""
        info = b.info
        if b.sub_variant == 0x3F:
            rows = (info["groundMobils"] if b.ground else info["airMobils"]) or []
            if b.variant < len(rows) and len(rows[b.variant]) > 1:
                raise BuildError(f"{b}: random sub variant over {len(rows[b.variant])} mobils")
        mobil = get_mobil(info, b.ground, b.variant, b.sub_variant)
        if mobil is None:
            if b.type == PYLON:
                raise BuildError(f"{b}: pylon blocks are not supported")
            mobil = get_mobil(info, not b.ground, b.variant, b.sub_variant)
        if mobil is None:
            b.mobil = None
            return
        root = self.assets.mobil_tree(mobil["mobil"], repr(b), mobil["file"], remap)
        root.flags |= 0x40  # CGameCtnBlockInfo::BuildBlockMobil (0x00662230)
        b.mobil = Mobil(root, True, repr(b))
        b.extra_mobils = []

    def create_mobil_for_clip(self, b: Block, remap: Remap | None) -> None:
        """CGameCtnChallenge::CreateMobilForClip (0x005A5960)."""
        coord = b.coord
        default_clip = None
        if self.terrain_at(coord) != T_TERRAIN:
            zone = self.real_zone(coord)
            if "blockInfoFrontier" in zone:
                if b.ground:
                    return  # GetZoneFlat rejects frontier zones: no mobil
                # CGameCtnCollection::GetBasicZone (0x0060D9E0): the child zone.
                zone = self.zones[zone["childZoneId"]]
            path = zone["blockInfoClip"] or self.default_zone["blockInfoClip"]
            default_clip = self.assets.block_info_at(path)
        else:
            default_clip = self.assets.block_info_at(self.default_zone["blockInfoClip"])
        unit = b.units[0]
        for d in range(4):
            nb = self.block_at(neighbour(coord, d))
            junction = None
            if nb is not None and nb.type not in (ROAD, CLIP):
                nunit = self.unit_at(neighbour(coord, d))
                if nunit is not None:
                    junction = nunit.clips[opposed(d)]
            unit.clips[d] = junction if junction is not None else default_clip
        # With a light-mapped mood every clip becomes its own mobil (and
        # corpus); otherwise the clip trees are merged under one empty root.
        separate = self.decoration["decoMoodLightMap"] is not None
        mobils: list[Tree] = []
        if not separate:
            root = Tree("", 0x1E80A, fp.IDENTITY, None, [], False)  # CPlugTree::CPlugTree
            root.flags = (root.flags & ~4) | 4 | 0x10000  # SetUseLocation(1)
            mobils.append(root)
        for d in range(4):
            clip_info = unit.clips[d]
            mobil = get_mobil(clip_info, b.ground, b.variant, 0)
            if mobil is None:
                continue
            # The clip's editor helper mobil goes to a separate CSceneMobil
            # whose CHmsItem keeps the constructor flags (0x0053C340:
            # IsCollisionStatic clear), so it never reaches the static tree.
            loc = fp.mobil_loc((0, 0, 0), d, (1, 1, 1), self.square, self.height)
            tree = self.assets.mobil_tree(mobil["mobil"], f"{b} clip {clip_info['id']} dir {d}",
                                          mobil["file"], remap)
            if separate:
                mobils.append(tree)
            else:
                mobils[0].children.append(tree)
            tree.flags = (tree.flags & ~4) | 4 | 0x10000
            tree.loc = fp.iso_set_mult(tree.loc if tree.loc is not None else fp.IDENTITY, loc)
        for root in mobils:
            root.hide_invalid()
            root.flags |= 0x40 | 0x80
            if not root.children:
                root.flags &= ~0x4000
        if not mobils:
            b.mobil = None
            b.extra_mobils = []
            return
        # CFastBuffer::ReplaceByLastAt(0, 1) pops the main mobil, so the
        # last clip mobil moves to the front of the extra buffer.
        b.mobil = Mobil(mobils[0], True, repr(b))
        extra = mobils[1:]
        if len(extra) > 1:
            extra = [extra[-1]] + extra[:-1]
        b.extra_mobils = [Mobil(t, True, f"{b} clip #{i}") for i, t in enumerate(extra, 1)]

    # -- scene population ---------------------------------------------------
    def block_mobil_loc(self, b: Block) -> tuple:
        """CGameCtnBlock::GetMobilLoc (0x0060B320) as used by
        CGameCtnApp::AddBlockMobilToScene (0x005EF190)."""
        size = info_size(b.info, b.ground)
        loc = fp.mobil_loc(b.coord, b.direction, size, self.square, self.height)
        if b.type == FRONTIER:
            # Frontier mobils are authored relative to the parent zone level.
            zone = self.zone_from_land_info(b.info)
            ty = fp.sub(loc[10], fp.mul(float(zone["common"]["height"]), self.height))
            loc = loc[:10] + (ty,) + loc[11:]
        return loc

    def corpora(self) -> list[tuple[Mobil, tuple]]:
        """Static corpora in collision-group order: the decoration scene, then
        CGameCtnApp::UpdateBlockMobils (0x005F77D0) over the add list."""
        out: list[tuple[Mobil, tuple]] = []
        scene = self.decoration["decoSizeNode"]["sceneNode"]
        for entry, loc in zip(scene["mobils"], scene["locations"]):
            mobil = entry["mobil"]
            if mobil is None:
                raise BuildError("decoration scene entry without mobil")
            root = self.assets.mobil_tree(mobil, f"decoration {mobil['name']!r}")
            if mobil["name"] == "Warp" and self.decoration["decoratorSolidWarp"] is not None:
                deco = self.assets.load(self.decoration["decoratorSolidWarp"])
                decorate_solid(root, deco["treeDecorators"], self.quality)
            # CHmsItem+0x18 bit 19 (IsCollisionStatic) comes straight from the
            # mobil file for scene mobils; blocks get it set explicitly.
            static = bool(mobil["itemFlags"] >> 19 & 1)
            out.append((Mobil(root, static, f"decoration {mobil['name']!r}"),
                        fp.iso_from_bytes(base64.b64decode(loc["location"]))))
        for b in self.add_list:
            if b.mobil is None:
                continue
            loc = self.block_mobil_loc(b)
            out.append((b.mobil, loc))
            for extra in b.extra_mobils:
                out.append((extra, loc))
        return out
