"""TMNFTRK1 v3 snapshot reader, writer, canonicalizer and differ.

Layout mirrors src/track.h. All *_rel fields on disk are byte offsets from the
file base; sections are laid out in id order, each 8-byte aligned, directly
after the 0x150-byte header. Sections 10 and 11 are the water owner
(CHmsZone +0x154 GmMap2 header, +0x178 level, +0x17c floor) and its cells.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

MAGIC = b"TMNFTRK1"
VERSION = 3
HEADER_SIZE = 0x150
SECTION_COUNT = 12
PAYLOAD_SHA256 = 0x60 + SECTION_COUNT * 16      # 0x120
EXE_SHA256 = bytes.fromhex(
    "3847cf9f20bfc63914450060ed528c12104f743d96ad23d6e76abd178de8c84f")
STRIDES = [0x60, 0x18, 0x38, 0x0C, 0x20, 0x20, 1, 8, 0x14, 0x30, 0x28, 1]
NAMES = ["entries", "surfaces", "meshes", "vertices", "faces", "nodes",
         "material_ids", "material_data", "collision_pairs", "corpus_isos",
         "water", "water_cells"]

ENTRY = struct.Struct("<I6f12fIQII")
SURFACE = struct.Struct("<QQII")
MESH = struct.Struct("<IHBBIIQIIQIIQ")
WATER = struct.Struct("<4fIIIffI")


def align8(value: int) -> int:
    return (value + 7) & ~7


@dataclass
class Entry:
    skip: int
    box: bytes          # 6 floats raw
    iso: bytes          # 12 floats raw
    flags: int
    surface: int        # index into surfaces
    tree: int
    corpus: int

    @property
    def active(self) -> bool:
        return self.flags != 0


@dataclass
class Surface:
    mesh: int           # index into meshes
    materials: bytes    # one physics id byte per material slot


@dataclass
class Mesh:
    vtable: int
    material_index: int
    type: int
    reserved: int
    vertices: bytes     # 12 bytes each
    faces: bytes        # 32 bytes each
    nodes: bytes        # 32 bytes each

    @property
    def key(self) -> bytes:
        return self.vertices + self.faces + self.nodes


@dataclass
class Water:
    cell_x: float
    cell_z: float
    origin_x: float
    origin_z: float
    width: int
    height: int
    default_cell: int
    level: float
    floor: float
    cells: bytes        # width * height, row-major by z

    def header(self) -> bytes:
        return WATER.pack(self.cell_x, self.cell_z, self.origin_x, self.origin_z,
                          self.width, self.height, self.default_cell,
                          self.level, self.floor, 0)


@dataclass
class Track:
    entries: list[Entry] = field(default_factory=list)
    surfaces: list[Surface] = field(default_factory=list)
    meshes: list[Mesh] = field(default_factory=list)
    material_data: bytes = b""
    collision_pairs: list[bytes] = field(default_factory=list)
    corpus_isos: list[bytes] = field(default_factory=list)   # index = corpus id - 1
    water: Water | None = None
    track_sha256: bytes = b"\0" * 32

    def payload(self) -> bytes:
        return build_image(self)[HEADER_SIZE:]


def load(path: str) -> Track:
    d = open(path, "rb").read()
    if d[:8] != MAGIC:
        raise ValueError(f"{path}: bad magic")
    version, = struct.unpack_from("<I", d, 8)
    if version != VERSION:
        raise ValueError(f"{path}: version {version}, expected {VERSION}")
    secs = [struct.unpack_from("<QII", d, 0x60 + i * 16) for i in range(SECTION_COUNT)]
    t = Track()
    t.track_sha256 = d[0x40:0x60]
    so, sc, ss = secs[1]
    mo, mc, ms = secs[2]
    eo, ec, es = secs[0]
    for i in range(ec):
        f = ENTRY.unpack_from(d, eo + i * es)
        raw = d[eo + i * es: eo + (i + 1) * es]
        t.entries.append(Entry(
            skip=f[0], box=raw[4:0x1C], iso=raw[0x1C:0x4C], flags=f[19],
            surface=(f[20] - so) // ss, tree=f[21], corpus=f[22]))
    mid_off = secs[6][0]
    for i in range(sc):
        mesh_rel, mat_rel, matc, _ = SURFACE.unpack_from(d, so + i * ss)
        t.surfaces.append(Surface(
            mesh=(mesh_rel - mo) // ms, materials=d[mat_rel:mat_rel + matc]))
    for i in range(mc):
        vt, mi, ty, rs, vc, _, vrel, fc, _, frel, nc, _, nrel = MESH.unpack_from(d, mo + i * ms)
        t.meshes.append(Mesh(
            vtable=vt, material_index=mi, type=ty, reserved=rs,
            vertices=d[vrel:vrel + vc * 12], faces=d[frel:frel + fc * 32],
            nodes=d[nrel:nrel + nc * 32]))
    mdo, mdc, mds = secs[7]
    t.material_data = d[mdo:mdo + mdc * mds]
    po, pc, ps = secs[8]
    t.collision_pairs = [d[po + i * ps: po + (i + 1) * ps] for i in range(pc)]
    co, cc, cs = secs[9]
    t.corpus_isos = [d[co + i * cs: co + (i + 1) * cs] for i in range(cc)]
    wo, wc, ws = secs[10]
    if (wc, ws) != (1, WATER.size):
        raise ValueError(f"{path}: water section {wc}x{ws}")
    fields = WATER.unpack_from(d, wo)
    ko, kc, ks = secs[11]
    if ks != 1 or kc != fields[4] * fields[5]:
        raise ValueError(f"{path}: water cells {kc}x{ks} for {fields[4]}x{fields[5]}")
    t.water = Water(*fields[:9], cells=d[ko:ko + kc])
    return t


def build_image(t: Track) -> bytearray:
    """Serialize with the tracer's layout: arrays deduplicated in first-use
    order, sections contiguous and 8-byte aligned."""
    vertices = bytearray()
    faces = bytearray()
    nodes = bytearray()
    mesh_rows = []
    for m in t.meshes:
        mesh_rows.append((m, len(vertices), len(faces), len(nodes)))
        vertices += m.vertices
        faces += m.faces
        nodes += m.nodes
    material_ids = bytearray()
    surface_rows = []
    for s in t.surfaces:
        surface_rows.append((s, len(material_ids)))
        material_ids += s.materials
    water = t.water.header()
    sizes = [
        len(t.entries) * 0x60, len(t.surfaces) * 0x18, len(t.meshes) * 0x38,
        len(vertices), len(faces), len(nodes), len(material_ids),
        len(t.material_data), len(t.collision_pairs) * 0x14,
        len(t.corpus_isos) * 0x30, len(water), len(t.water.cells),
    ]
    counts = [
        len(t.entries), len(t.surfaces), len(t.meshes), len(vertices) // 12,
        len(faces) // 32, len(nodes) // 32, len(material_ids),
        len(t.material_data) // 8, len(t.collision_pairs), len(t.corpus_isos),
        1, len(t.water.cells),
    ]
    offsets = []
    cursor = HEADER_SIZE
    for size in sizes:
        cursor = align8(cursor)
        offsets.append(cursor)
        cursor += size
    image = bytearray(cursor)
    o = offsets[0]
    for e in t.entries:
        struct.pack_into("<I", image, o, e.skip)
        image[o + 4:o + 0x1C] = e.box
        image[o + 0x1C:o + 0x4C] = e.iso
        struct.pack_into("<IQII", image, o + 0x4C, e.flags,
                         offsets[1] + e.surface * 0x18, e.tree, e.corpus)
        o += 0x60
    o = offsets[1]
    for s, mat_off in surface_rows:
        SURFACE.pack_into(image, o, offsets[2] + s.mesh * 0x38,
                          offsets[6] + mat_off, len(s.materials), 0)
        o += 0x18
    o = offsets[2]
    for m, vo, fo, no in mesh_rows:
        MESH.pack_into(image, o, m.vtable, m.material_index, m.type, m.reserved,
                       len(m.vertices) // 12, 0, offsets[3] + vo,
                       len(m.faces) // 32, 0, offsets[4] + fo,
                       len(m.nodes) // 32, 0, offsets[5] + no)
        o += 0x38
    image[offsets[3]:offsets[3] + len(vertices)] = vertices
    image[offsets[4]:offsets[4] + len(faces)] = faces
    image[offsets[5]:offsets[5] + len(nodes)] = nodes
    image[offsets[6]:offsets[6] + len(material_ids)] = material_ids
    image[offsets[7]:offsets[7] + len(t.material_data)] = t.material_data
    o = offsets[8]
    for p in t.collision_pairs:
        image[o:o + 0x14] = p
        o += 0x14
    o = offsets[9]
    for iso in t.corpus_isos:
        image[o:o + 0x30] = iso
        o += 0x30
    image[offsets[10]:offsets[10] + len(water)] = water
    image[offsets[11]:offsets[11] + len(t.water.cells)] = t.water.cells
    struct.pack_into("<8sIIIIQ", image, 0, MAGIC, VERSION, 0x12345678,
                     HEADER_SIZE, SECTION_COUNT, len(image))
    image[0x20:0x40] = EXE_SHA256
    image[0x40:0x60] = t.track_sha256
    for i, (offset, count, stride) in enumerate(zip(offsets, counts, STRIDES)):
        struct.pack_into("<QII", image, 0x60 + i * 16, offset, count, stride)
    image[PAYLOAD_SHA256:PAYLOAD_SHA256 + 32] = hashlib.sha256(image[HEADER_SIZE:]).digest()
    return image


def save(t: Track, path: str) -> bytes:
    image = build_image(t)
    open(path, "wb").write(image)
    return bytes(image[PAYLOAD_SHA256:PAYLOAD_SHA256 + 32])


def canonicalize(t: Track) -> Track:
    """Remove the parts of a snapshot that are heap noise in the game.

    Inner BVH nodes are allocated with CFastBuffer::AddNewElem and only their
    skip count, box and surface pointer (null) are ever written; iso, tree and
    corpus hold whatever the heap contained. The tracer turns those stale
    pointers into stable ids, which inflates the id space and the corpus iso
    table with zero rows. Canonical form: inactive entries carry zero iso, tree
    and corpus; ids are renumbered densely in first-seen order over active
    entries; the corpus iso table is compacted to match. GmSurf+7 is struct
    padding the game never writes, so mesh.reserved is zeroed too.

    CPlugTree flag 0x1000 is set by CGameAdvertising::SetSolidFlagsForFixedAds
    (0x006FAB20) on visible decoration trees whose shader carries an ad
    texture; it depends on the advertising configuration and is never read by
    the physics (only bit 0x80 of tree_flags is, see src/collision.c and
    src/race.c), so it is cleared.
    """
    meshes = [Mesh(m.vtable, m.material_index, m.type, 0, m.vertices, m.faces, m.nodes)
              for m in t.meshes]
    out = Track(surfaces=t.surfaces, meshes=meshes, material_data=t.material_data,
                collision_pairs=t.collision_pairs, water=t.water,
                track_sha256=t.track_sha256)
    tree_ids: dict[int, int] = {}
    corpus_ids: dict[int, int] = {}
    isos: list[bytes] = []
    for e in t.entries:
        if not e.active:
            out.entries.append(Entry(e.skip, e.box, b"\0" * 0x30, 0, e.surface, 0, 0))
            continue
        if e.tree not in tree_ids:
            tree_ids[e.tree] = len(tree_ids) + 1
        if e.corpus not in corpus_ids:
            corpus_ids[e.corpus] = len(corpus_ids) + 1
            isos.append(t.corpus_isos[e.corpus - 1])
        out.entries.append(Entry(e.skip, e.box, e.iso, e.flags & ~0x1000, e.surface,
                                 tree_ids[e.tree], corpus_ids[e.corpus]))
    out.corpus_isos = isos
    return out


def fmt_iso(raw: bytes) -> str:
    v = struct.unpack("<12f", raw)
    return "[" + " ".join(f"{x:g}" for x in v[:9]) + "] t=(" + ",".join(f"{x:g}" for x in v[9:]) + ")"


def fmt_box(raw: bytes) -> str:
    v = struct.unpack("<6f", raw)
    return "c=(" + ",".join(f"{x:g}" for x in v[:3]) + ") h=(" + ",".join(f"{x:g}" for x in v[3:]) + ")"


def compare(a: Track, b: Track, label_a: str = "built", label_b: str = "oracle",
            limit: int = 20) -> list[str]:
    """Section-by-section diff. Returns human readable lines; empty means the
    payloads are byte identical."""
    lines: list[str] = []
    ia, ib = build_image(a), build_image(b)
    if ia[HEADER_SIZE:] == ib[HEADER_SIZE:]:
        return lines

    def sec(name, xa, xb, fmt):
        if xa == xb:
            return
        lines.append(f"{name}: {label_a} {len(xa)} vs {label_b} {len(xb)}")
        shown = 0
        for i in range(max(len(xa), len(xb))):
            va = xa[i] if i < len(xa) else None
            vb = xb[i] if i < len(xb) else None
            if va != vb:
                if shown < limit:
                    lines.append(f"  [{i}] {label_a}: {fmt(va)}")
                    lines.append(f"  [{i}] {label_b}: {fmt(vb)}")
                shown += 1
        if shown > limit:
            lines.append(f"  ... {shown - limit} more differing rows")

    def fe(e):
        if e is None:
            return "<missing>"
        return (f"skip={e.skip} flags={e.flags:#x} surf={e.surface} tree={e.tree} "
                f"corpus={e.corpus} {fmt_box(e.box)} {fmt_iso(e.iso)}")

    def fs(s):
        if s is None:
            return "<missing>"
        return f"mesh={s.mesh} mats={s.materials.hex()}"

    def fm(m):
        if m is None:
            return "<missing>"
        return (f"vt={m.vtable:#x} mi={m.material_index} ty={m.type} "
                f"v={len(m.vertices) // 12} f={len(m.faces) // 32} n={len(m.nodes) // 32} "
                f"sha={hashlib.sha256(m.key).hexdigest()[:12]}")

    sec("entries", a.entries, b.entries, fe)
    sec("surfaces", a.surfaces, b.surfaces, fs)
    sec("meshes", a.meshes, b.meshes, fm)
    if a.material_data != b.material_data:
        lines.append("material_data differs")
    sec("collision_pairs", a.collision_pairs, b.collision_pairs,
        lambda p: "<missing>" if p is None else p.hex())
    sec("corpus_isos", a.corpus_isos, b.corpus_isos,
        lambda p: "<missing>" if p is None else fmt_iso(p))
    if a.water.header() != b.water.header():
        lines.append(f"water: {label_a} {a.water.header().hex()} vs {label_b} {b.water.header().hex()}")
    if a.water.cells != b.water.cells:
        wa, wb = a.water, b.water
        differing = [i for i in range(min(len(wa.cells), len(wb.cells))) if wa.cells[i] != wb.cells[i]]
        lines.append(f"water_cells: {len(differing)} cells differ, first "
                     + " ".join(f"({i % wa.width},{i // wa.width}) {label_a}={wa.cells[i]} {label_b}={wb.cells[i]}"
                                for i in differing[:limit]))
    if a.track_sha256 != b.track_sha256:
        lines.append(f"track_sha256: {a.track_sha256.hex()} vs {b.track_sha256.hex()}")
    return lines
