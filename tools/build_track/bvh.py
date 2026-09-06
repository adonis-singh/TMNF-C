"""Static collision tree: CHmsCollisionManager::SGroup::UpdateStaticCollisionTrees
(0x0053AE50) flattens every static corpus into SColOctreeCell leaves and
GmOctree<SColOctreeCell>::Build (0x0053AD70) turns them into the flat
skip-list BVH the snapshot stores.
"""
from __future__ import annotations

from dataclasses import dataclass

import fp
from assets import BuildError, Surface, Tree


@dataclass
class Cell:
    box: tuple
    iso: tuple
    surface: Surface
    tree: Tree
    corpus: int          # index into the corpus list (0-based)


@dataclass
class Node:
    skip: int
    box: tuple
    cell: Cell | None    # None for inner nodes


def add_static_surfaces(cells: list[Cell], corpus: int, tree: Tree, parent: tuple) -> None:
    """CHmsCollisionManager::SGroup::AddStaticSurfacesFromTree (0x005398E0)."""
    if not (tree.flags & 0x80):
        return
    if tree.flags & 4:
        iso = fp.iso_set_mult(tree.loc if tree.loc is not None else fp.IDENTITY, parent)
    else:
        iso = parent
    for child in tree.children:
        add_static_surfaces(cells, corpus, child, iso)
    surface = tree.surface
    if surface is not None:
        if surface.surf_type != "Mesh":
            raise BuildError(f"static surface {tree.name!r} is a {surface.surf_type}, not a mesh")
        if 0.0 <= surface.box[3]:
            cells.append(Cell(fp.box_set_mult(surface.box, iso), iso, surface, tree, corpus))


def flatten(corpora) -> list[Cell]:
    cells: list[Cell] = []
    for index, (mobil, loc) in enumerate(corpora):
        if not mobil.static:
            continue
        add_static_surfaces(cells, index, mobil.root, loc)
    return cells


def build(cells: list[Cell]) -> list[Node]:
    """GmOctree<SColOctreeCell>::Build with bintree=1, no depth/count/volume
    limits. Returns the cell array in memory order."""
    out: list[Node] = [Node(0, (0.0,) * 6, None)]
    _bintree(out, cells)
    out[0].skip = len(out)
    return out


def _bintree(out: list[Node], cells: list[Cell]) -> int:
    """GmOctree<SColOctreeCell>::BuildBintreeRecurse (0x005391E0). Returns the
    number of cells written including the caller's node."""
    n = len(cells)
    if n == 0:
        return 0
    box = cells[0].box
    for c in cells[1:]:
        box = fp.box_union(box, c.box)
    out[-1].box = box
    cx, cy, cz, hx, hy, hz = box
    result = 1
    if fp.add(fp.add(fp.mul(hz, hz), fp.mul(hy, hy)), fp.mul(hx, hx)) != 0.0:
        ex = fp.sub(fp.add(cx, hx), fp.sub(cx, hx))
        ey = fp.sub(fp.add(cy, hy), fp.sub(cy, hy))
        ez = fp.sub(fp.add(cz, hz), fp.sub(cz, hz))
        axis = 0
        cur = ex
        if ex < ey:
            cur = ey
            axis = 1
        if cur < ez:
            axis = 2
        lo = (fp.sub(cx, hx), fp.sub(cy, hy), fp.sub(cz, hz))[axis]
        hi = (fp.add(cx, hx), fp.add(cy, hy), fp.add(cz, hz))[axis]
        mid = fp.mul(fp.add(lo, hi), 0.5)
        # SEntInfo buffer: (side, index) pairs.
        info = []
        for i, c in enumerate(cells):
            cc = c.box[axis]
            ch = c.box[3 + axis]
            top = fp.add(ch, cc)
            side = 0
            if top < mid or top == mid:
                side = 1
            elif mid <= fp.sub(cc, ch):
                side = 2
            info.append([side, i])

        def take(side):
            picked = []
            i = 0
            while i < len(info):
                if info[i][0] == side:
                    picked.append(cells[info[i][1]])
                    info[i] = info[-1]      # ReplaceByLastAt
                    info.pop()
                else:
                    i += 1
            return picked

        for side in (1, 2):
            part = take(side)
            if len(part) == 1:
                out.append(Node(1, part[0].box, part[0]))
                result += 1
            elif len(part) > 1:
                node_index = len(out)
                out.append(Node(0, (0.0,) * 6, None))
                r = _bintree(out, part)
                out[node_index].skip = r
                result += r
        rest = [cells[i] for _, i in info]
        if len(rest) == n or len(rest) < 2:
            for c in rest:
                out.append(Node(1, c.box, c))
            result += len(rest)
        else:
            node_index = len(out)
            out.append(Node(0, (0.0,) * 6, None))
            r = _bintree(out, rest)
            out[node_index].skip = r
            result += r
        return result
    for c in cells:
        out.append(Node(1, c.box, c))
    return n + 1
