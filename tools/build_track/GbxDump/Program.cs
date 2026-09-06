// Dumps the TMNF GBX assets the offline track builder needs as JSON:
// challenges, collections, zones, decorations, block infos, and solids with
// their collision surfaces. Geometry is emitted as raw little-endian bytes
// (base64) so no float ever passes through text.
using System.Text.Json;
using System.Text.Json.Serialization;
using GBX.NET;
using GBX.NET.Components;
using GBX.NET.Engines.Game;
using GBX.NET.Engines.Hms;
using GBX.NET.Engines.Plug;
using GBX.NET.Engines.Scene;
using GBX.NET.Interfaces.Game;
using GBX.NET.LZO;
using GBX.NET.PAK;

Gbx.LZO = new MiniLZO();

if (args.Length == 0)
{
    Fail("expected command: dump, extract");
}

switch (args[0])
{
    case "dump":
        if (args.Length != 3)
        {
            Fail("usage: dump INPUT.Gbx OUTPUT.json");
        }
        Dump(args[1], args[2]);
        break;
    case "dump-tree":
        if (args.Length < 4)
        {
            Fail("usage: dump-tree INPUT_ROOT OUTPUT_ROOT SUFFIX...");
        }
        DumpTree(args[1], args[2], args[3..]);
        break;
    case "extract":
        if (args.Length < 4)
        {
            Fail("usage: extract PACKS_DIR OUTPUT_DIR CANDIDATES_FILE SUFFIX...");
        }
        await Extract(args[1], args[2], args[3], args[4..]);
        break;
    case "classes":
        foreach (var type in typeof(Gbx).Assembly.GetTypes()
            .Where(t => typeof(IClass).IsAssignableFrom(t) && t.IsClass))
        {
            var id = GBX.NET.Managers.ClassManager.GetId(type);
            if (id is null) continue;
            var parent = type.BaseType is null ? 0
                : GBX.NET.Managers.ClassManager.GetId(type.BaseType) ?? 0;
            Console.WriteLine($"{id:X8} {parent:X8} {type.Name}");
        }
        break;
    case "decompress":
        if (args.Length != 3)
        {
            Fail("usage: decompress INPUT.Gbx OUTPUT.Gbx");
        }
        Gbx.Decompress(args[1], args[2]);
        break;
    case "probe":
        if (args.Length != 4)
        {
            Fail("usage: probe PACKS_DIR PAK_ID FILE_SUFFIX");
        }
        await Probe(args[1], args[2], args[3]);
        break;
    default:
        Fail($"unknown command {args[0]}");
        break;
}

static GbxReadSettings Settings() => new() { SafeSkippableChunks = true };

static void DumpTree(string inputRoot, string outputRoot, string[] suffixes)
{
    RefRoots.Root = Path.GetFullPath(inputRoot);
    var files = Directory.EnumerateFiles(inputRoot, "*", SearchOption.AllDirectories)
        .Where(path => suffixes.Any(suffix =>
            path.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)))
        .Order()
        .ToArray();
    var failures = 0;
    foreach (var path in files)
    {
        var relative = Path.GetRelativePath(inputRoot, path);
        var output = Path.Combine(outputRoot, relative + ".json");
        try
        {
            Dump(path, output);
        }
        catch (Exception error)
        {
            failures++;
            Console.Error.WriteLine($"gbx-dump: {relative}: {error.GetType().Name}: {error.Message}");
        }
    }
    Console.WriteLine($"dumped {files.Length - failures}/{files.Length} files");
}

static void Dump(string input, string output)
{
    var node = Gbx.ParseNode(input, Settings())
        ?? throw new InvalidDataException($"{input}: no main node");
    object result = node switch
    {
        CGameCtnChallenge challenge => SerializeChallenge(challenge),
        CGameCtnCollection collection => SerializeCollection(collection),
        CGameCtnZoneFlat zone => SerializeZoneFlat(zone),
        CGameCtnZoneFrontier zone => SerializeZoneFrontier(zone),
        CGameCtnDecoration decoration => SerializeDecoration(decoration),
        CGameCtnDecorationSize size => SerializeDecorationSize(size),
        CGameCtnDecorationTerrainModifier modifier => SerializeTerrainModifier(modifier),
        CPlugGameSkin skin => SerializeGameSkin(skin),
        CPlugDecoratorSolid deco => SerializeDecoratorSolid(deco),
        CGameCtnBlockInfo info => SerializeBlockInfo(info),
        CPlugSolid solid => SerializeSolid(solid),
        CSceneLayout layout => SerializeLayout(layout),
        CSceneMobil mobil => SerializeMobil(mobil),
        CPlugMaterial material => new { surfaceId = (int)material.SurfaceId },
        _ => throw new InvalidDataException(
            $"{input}: unsupported class {node.GetType().Name}")
    };
    WriteJson(output, new
    {
        source = RefRoots.Root is null ? Path.GetFullPath(input)
            : Path.GetRelativePath(RefRoots.Root, Path.GetFullPath(input)),
        @class = node.GetType().Name,
        node = result
    });
}

static object SerializeChallenge(CGameCtnChallenge map)
{
    var tmf = (IGameCtnChallengeTMF)map;
    var blocks = tmf.GetBlocks().Select((block, index) => new
    {
        index,
        name = block.Name,
        coord = new[] { block.Coord.X, block.Coord.Y, block.Coord.Z },
        direction = (int)block.Direction,
        ground = block.IsGround,
        variant = block.Variant,
        subVariant = block.SubVariant,
        flags = block.Flags,
        isClip = block.IsClip,
    }).ToArray();
    return new
    {
        mapUid = map.MapUid,
        mapName = map.MapName,
        collection = map.Collection?.ToString(),
        decoration = map.Decoration is null ? null : new
        {
            id = map.Decoration.Id,
            collection = map.Decoration.Collection.ToString(),
            author = map.Decoration.Author
        },
        size = new[] { map.Size.X, map.Size.Y, map.Size.Z },
        needUnlock = map.NeedUnlock,
        blocks
    };
}

static string? RefPath(GbxRefTableFile? file)
{
    var full = file?.GetFullPath();
    if (full is null || RefRoots.Root is null) return full;
    var relative = Path.GetRelativePath(RefRoots.Root, full);
    if (relative.StartsWith("..")) throw new InvalidDataException($"{full}: reference outside {RefRoots.Root}");
    return relative;
}


static object SerializeCollection(CGameCtnCollection c)
{
    return new
    {
        collection = c.Collection.ToString(),
        squareSize = c.SquareSize,
        squareHeight = c.SquareHeight,
        defaultZone = RefPath(c.DefaultZoneFile),
        zones = c.CompleteListZoneList?.Select(z => RefPath(z.File)).ToArray(),
        defaultDecoration = RefPath(c.DefaultDecorationFile),
        defaultZoneId = c.DefaultZoneId,
        vehicle = c.Vehicle?.ToString(),
        // Chunk 0x0303301D: CGameCtnCollection+0x144 replacement modifiers and
        // the (base zone, replacement zone) string pairs that select them.
        zoneStrings = c.ZoneStrings?.Select(z => new { baseId = z.Base, replacement = z.Replacement }).ToArray(),
        replacementTerrainModifiers = c.ReplacementTerrainModifiers?.Select(m => RefPath(m.File)).ToArray()
    };
}

// CGameCtnDecorationTerrainModifier: CPlugGameSkin + folder. CGameSkin::
// AddRemappingParams (0x006FCAA0) turns the skin's customizable fids into
// file substitutions rooted at remapFolder.
static object SerializeTerrainModifier(CGameCtnDecorationTerrainModifier m)
{
    var skin = m.Remapping;
    return new
    {
        idName = m.IdName,
        remapFolder = m.RemapFolder,
        remapping = RefPath(m.RemappingFile),
        remappingNode = skin is null ? null : SerializeGameSkin(skin)
    };
}

static object SerializeGameSkin(CPlugGameSkin skin)
{
    return new
    {
        dirName = skin.DirName,
        fids = skin.CustomizableFids?.Select(f => new
        {
            classId = f.ClassId,
            name = f.Name,
            file = RefPath(f.NodeFile)
        }).ToArray()
    };
}

static object SerializeZoneCommon(CGameCtnZone z)
{
    return new
    {
        zoneId = z.ZoneId,
        surfaceId = z.SurfaceId,
        height = z.Height,
        depth = z.Depth,
        oldZone = z.OldZone,
        hasWater = z.HasWater,
        isLargeZone = z.IsLargeZone,
        waterId = z.WaterId
    };
}

static object SerializeZoneFlat(CGameCtnZoneFlat z)
{
    return new
    {
        common = SerializeZoneCommon(z),
        blockInfoFlat = RefPath(z.BlockInfoFlatFile),
        blockInfoClip = RefPath(z.BlockInfoClipFile),
        blockInfoRoad = RefPath(z.BlockInfoRoadFile),
        blockInfoPylon = RefPath(z.BlockInfoPylonFile),
        groundOnly = z.GroundOnly
    };
}

static object SerializeZoneFrontier(CGameCtnZoneFrontier z)
{
    return new
    {
        common = SerializeZoneCommon(z),
        blockInfoFrontier = RefPath(z.BlockInfoFrontierFile),
        parentZoneId = z.ParentZoneId,
        childZoneId = z.ChildZoneId,
        blockYOffsetFromParent = z.BlockYOffsetFromParent
    };
}

static object SerializeDecoration(CGameCtnDecoration d)
{
    return new
    {
        ident = d.Ident.ToString(),
        decoSize = RefPath(d.DecoSizeFile),
        decoSizeNode = d.DecoSize is null ? null : SerializeDecorationSize(d.DecoSize),
        terrainModifierBase = RefPath(d.TerrainModifierBaseFile),
        terrainModifierCovered = RefPath(d.TerrainModifierCoveredFile),
        decoratorSolidWarp = RefPath(d.DecoratorSolidWarpFile),
        decoMood = RefPath(d.DecoMoodFile),
        // CGameCtnChallenge::CreateMobilForClip (0x005A5960) assembles clip
        // mobils differently depending on whether the mood carries a light map.
        decoMoodLightMap = d.DecoMood is null
            ? throw new InvalidDataException($"{d.Ident}: DecoMood {d.DecoMoodFile?.GetFullPath()} not loaded")
            : RefPath(d.DecoMood.HmsLightMapFile),
        decorationZoneFrontierId = d.DecorationZoneFrontierId
    };
}

// CPlugDecoratorSolid::DecorateSolid (0x00898C00) applies these per tree id,
// selecting by the mobil quality (ESolidQuality 0 low / 1 medium / 2 high).
static object SerializeDecoratorSolid(CPlugDecoratorSolid d)
{
    return new
    {
        treeDecorators = d.TreeDecorators.Select(t => t is null ? null : new
        {
            treeId = t.TreeId,
            material = t.Material is not null,
            treeLight = t.TreeLight is not null,
            existCond = t.ExistCond,
            visibleCond = t.VisibleCond,
            visibleApplyOnChilds = t.VisibleApplyOnChilds,
            shadowCasterCond = t.ShadowCasterCond,
            shadowCasterApplyOnChilds = t.ShadowCasterApplyOnChilds,
            transformVisualToSurface = t.TransformVisualToSurface,
            noLocation = t.NoLocation,
            collidableCond = t.CollidableCond
        }).ToArray()
    };
}

static object SerializeDecorationSize(CGameCtnDecorationSize s)
{
    return new
    {
        baseHeightBase = s.BaseHeightBase,
        size = new[] { s.Size.X, s.Size.Y, s.Size.Z },
        offsetBlockY = s.OffsetBlockY,
        baseHeightOffset = s.BaseHeightOffset,
        scene = RefPath(s.SceneFile),
        sceneNode = s.Scene is null ? null : SerializeLayout(s.Scene)
    };
}

static object SerializeLayout(CSceneLayout layout)
{
    var mobils = layout.Scene?.Select((entry, index) => new
    {
        index,
        u01 = entry.U01,
        mobil = entry.Mobil is CSceneMobil mobil ? SerializeMobil(mobil) : null,
        @class = entry.Mobil?.GetType().Name
    }).ToArray();
    var locations = layout.SceneLocations?.Select(loc => new
    {
        @class = loc.U01?.GetType().Name,
        location = Iso(loc.U02)
    }).ToArray();
    return new { mobils, locations };
}

static object SerializeBlockInfo(CGameCtnBlockInfo info)
{
    return new
    {
        ident = info.Ident.ToString(),
        id = info.Ident.Id,
        type = info.GetType().Name,
        isPillar = info.IsPillar,
        noRespawn = info.NoRespawn,
        wayPointType = info.WayPointType.ToString(),
        isReplacement = info.IsReplacement,
        spawnLocGround = info.SpawnLocGround is null ? null : Iso(info.SpawnLocGround.Value),
        spawnLocAir = info.SpawnLocAir is null ? null : Iso(info.SpawnLocAir.Value),
        groundUnits = info.GroundBlockUnitInfos?.Select(SerializeUnit).ToArray(),
        airUnits = info.AirBlockUnitInfos?.Select(SerializeUnit).ToArray(),
        groundMobils = SerializeMobilRows(info.GroundMobils),
        airMobils = SerializeMobilRows(info.AirMobils),
        groundHelperMobil = info.GroundHelperMobil is null ? null : SerializeMobil(info.GroundHelperMobil),
        airHelperMobil = info.AirHelperMobil is null ? null : SerializeMobil(info.AirHelperMobil),
        pylon = info is CGameCtnBlockInfoPylon pylon ? new
        {
            pylonOffset = pylon.PylonOffset,
            pylonAmount = pylon.PylonAmount.ToString(),
            pylonPlacement = pylon.PylonPlacement.ToString(),
            blockHeightOffset = pylon.BlockHeightOffset
        } : null,
        clip = info is CGameCtnBlockInfoClip clip ? new
        {
            asymmetricalClipId = clip.ASymmetricalClipId,
            isFullFreeClip = clip.IsFullFreeClip,
            isExclusiveFreeClip = clip.IsExclusiveFreeClip,
            clipType = clip.ClipType.ToString()
        } : null
    };
}

static object SerializeUnit(CGameCtnBlockUnitInfo unit)
{
    return new
    {
        placePylons = unit.PlacePylons,
        acceptPylons = unit.AcceptPylons,
        relativeOffset = new[] { unit.RelativeOffset.X, unit.RelativeOffset.Y, unit.RelativeOffset.Z },
        clips = unit.Clips?.Select(clip => RefPath(clip.File)).ToArray(),
        surface = unit.Surface,
        frontier = unit.Frontier,
        dir = (int)unit.Dir,
        underground = unit.Underground,
        replacementBlockInfo = RefPath(unit.ReplacementBlockInfoFile),
        replacementId = unit.ReplacementId,
        multiDir = unit.MultiDir.ToString(),
        terrainModifierId = unit.TerrainModifierId
    };
}

static object[][]? SerializeMobilRows(External<CSceneMobil>[][]? rows)
{
    return rows?.Select(row => row.Select(mobil =>
    {
        var node = mobil.GetNode(Settings());
        return node is null ? new { file = RefPath(mobil.File), mobil = (object?)null }
            : new { file = RefPath(mobil.File), mobil = (object?)SerializeMobil(node) };
    }).Cast<object>().ToArray()).ToArray();
}

static object SerializeMobil(CSceneMobil mobil)
{
    var item = mobil.Item;
    // CHmsItem::Chunk 0x06003011 (0x0053E0C0) reads the 8 flag bytes at
    // CHmsItem+0x18 verbatim; bit 19 of the first dword is IsCollisionStatic.
    var flags = item?.Chunks.Get<CHmsItem.Chunk06003011>();
    return new
    {
        name = mobil.Name,
        itemFlags = flags?.U01,
        solidNode = item?.Solid is CPlugSolid solid ? SerializeSolid(solid) : null,
        objectLinks = mobil.ObjectLink?.Select(link => new
        {
            mobilFile = RefPath(link.MobilFile),
            mobil = link.Mobil is CSceneMobil linked ? SerializeMobil(linked) : null,
            relativeLocation = Iso(link.RelativeLocation),
            mobilTreeId = link.MobilTreeId,
            name = link.Name
        }).ToArray()
    };
}

static object SerializeSolid(CPlugSolid solid)
{
    var tree = solid.Tree as CPlugTree;
    return new
    {
        treeFile = RefPath(solid.TreeFile),
        tree = tree is null ? null : SerializeTree(tree)
    };
}

static object SerializeTree(CPlugTree tree)
{
    return new
    {
        @class = tree.GetType().Name,
        name = tree.Name,
        flags = tree.Flags,
        location = tree.Location is null ? null : Iso(tree.Location.Value),
        surface = tree.Surface is CPlugSurface surface ? SerializeSurface(surface) : null,
        surfaceClass = tree.Surface?.GetType().Name,
        hasVisual = tree.Visual is not null,
        children = tree.Children.Select(SerializeTree).ToArray(),
        mipLevels = tree is CPlugTreeVisualMip mip
            ? mip.Levels.Select(level => new { farZ = level.FarZ, tree = SerializeTree(level.Tree) }).ToArray()
            : null
    };
}

static object SerializeSurface(CPlugSurface surface)
{
    var geom = surface.Geom;
    var chunk = geom?.Chunks.OfType<CPlugSurfaceGeom.Chunk0900F004>().FirstOrDefault();
    var mesh = geom?.Surf as CPlugSurface.Mesh;
    return new
    {
        geomName = chunk?.U01,
        box = chunk is null ? null : Box(chunk.U02),
        surfType = geom?.Surf?.GetType().Name,
        geomSurfaceId = geom?.SurfaceId?.ToString(),
        geomSurfaceIdValue = geom?.SurfaceId is null ? (int?)null : (int)geom.SurfaceId,
        materials = surface.Materials.Select(material => new
        {
            file = RefPath(material.MaterialFile),
            surfaceId = material.SurfaceId?.ToString(),
            surfaceIdValue = material.SurfaceId is null ? (int?)null : (int)material.SurfaceId,
            materialSurfaceId = MaterialSurfaceId(material)
        }).ToArray(),
        mesh = mesh is null ? null : SerializeMesh(mesh)
    };
}

static int? MaterialSurfaceId(CPlugSurface.SurfMaterial material)
{
    var node = material.GetMaterial(Settings());
    return node is null ? null : (int)node.SurfaceId;
}

static object SerializeMesh(CPlugSurface.Mesh mesh)
{
    using var vertices = new MemoryStream();
    using var faces = new MemoryStream();
    using var nodes = new MemoryStream();
    using (var w = new BinaryWriter(vertices, System.Text.Encoding.UTF8, true))
    {
        foreach (var v in mesh.Vertices)
        {
            w.Write(v.X); w.Write(v.Y); w.Write(v.Z);
        }
    }
    using (var w = new BinaryWriter(faces, System.Text.Encoding.UTF8, true))
    {
        foreach (var t in mesh.CookedTriangles ?? [])
        {
            w.Write(t.U01.X); w.Write(t.U01.Y); w.Write(t.U01.Z); w.Write(t.U01.W);
            w.Write(t.Indices.X); w.Write(t.Indices.Y); w.Write(t.Indices.Z);
            w.Write(t.SurfaceIndex); w.Write(t.U04); w.Write(t.U05);
        }
    }
    using (var w = new BinaryWriter(nodes, System.Text.Encoding.UTF8, true))
    {
        foreach (var c in mesh.OctreeCells ?? [])
        {
            w.Write(c.U01);
            w.Write(c.U02.X); w.Write(c.U02.Y); w.Write(c.U02.Z);
            w.Write(c.U03.X); w.Write(c.U03.Y); w.Write(c.U03.Z);
            w.Write(c.U04);
        }
    }
    return new
    {
        version = mesh.Version,
        octreeVersion = mesh.OctreeVersion,
        vertexCount = mesh.Vertices.Length,
        faceCount = mesh.CookedTriangles?.Length ?? 0,
        nodeCount = mesh.OctreeCells?.Length ?? 0,
        vertices = Convert.ToBase64String(vertices.ToArray()),
        faces = Convert.ToBase64String(faces.ToArray()),
        nodes = Convert.ToBase64String(nodes.ToArray())
    };
}

static string Iso(Iso4 iso)
{
    using var stream = new MemoryStream();
    using (var w = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
    {
        w.Write(iso.XX); w.Write(iso.XY); w.Write(iso.XZ);
        w.Write(iso.YX); w.Write(iso.YY); w.Write(iso.YZ);
        w.Write(iso.ZX); w.Write(iso.ZY); w.Write(iso.ZZ);
        w.Write(iso.TX); w.Write(iso.TY); w.Write(iso.TZ);
    }
    return Convert.ToBase64String(stream.ToArray());
}

static string Box(BoxAligned box)
{
    using var stream = new MemoryStream();
    using (var w = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
    {
        w.Write(box.X); w.Write(box.Y); w.Write(box.Z);
        w.Write(box.X2); w.Write(box.Y2); w.Write(box.Z2);
    }
    return Convert.ToBase64String(stream.ToArray());
}

static async Task Extract(
    string packsDirectory, string outputDirectory, string candidatesFile, string[] suffixes)
{
    var packListPath = Path.Combine(packsDirectory, PakList.FileName);
    if (!File.Exists(packListPath))
    {
        Fail($"missing {packListPath}");
    }
    var keys = (await PakList.ParseAsync(packListPath)).ToKeyInfoDictionary();
    var candidates = File.ReadAllLines(candidatesFile).Where(line => line.Length > 0);
    var hashes = await Pak.BruteforceFileHashesAsync(
        packsDirectory, keys, progress: null, keepUnresolvedHashes: false,
        additionalFileHashes: candidates);
    var count = 0;
    var failures = new List<string>();
    var listing = new List<string>();
    foreach (var pakPath in Directory.GetFiles(packsDirectory, "*.pak"))
    {
        var pakId = Path.GetFileNameWithoutExtension(pakPath);
        await using var pak = keys.TryGetValue(pakId, out var key)
            ? await Pak.ParseAsync(pakPath, key)
            : await Pak.ParseAsync(pakPath);
        foreach (var file in pak.Files.Values)
        {
            var name = (hashes.GetValueOrDefault(file.Name) ?? file.Name)
                .Replace('\\', Path.DirectorySeparatorChar);
            var relative = Path.Combine(pakId, file.FolderPath, name);
            listing.Add(relative);
            if (!suffixes.Any(suffix =>
                name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)))
            {
                continue;
            }
            var destination = Path.Combine(outputDirectory, relative);
            Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
            await using var stream = File.Create(destination);
            try
            {
                var gbx = await OpenGbx(pak, file, null);
                if (gbx.Header is GbxHeaderUnknown)
                {
                    CopyFile(pak, file, stream);
                }
                else
                {
                    gbx.Save(stream);
                }
            }
            catch (Exception error)
            {
                // A raw copy would carry the mis-keyed 8-byte blocks (silent
                // single-bit damage in float data), so refuse the file instead.
                stream.SetLength(0);
                failures.Add($"{relative}: {error.Message.Split('\n')[0]}");
                Console.Error.WriteLine($"gbx-dump: cannot extract {relative}: {error.Message}");
                continue;
            }
            count++;
        }
    }
    Directory.CreateDirectory(outputDirectory);
    File.WriteAllLines(Path.Combine(outputDirectory, "listing.txt"), listing.Order());
    // Files that could not be decoded are recorded so that a build needing one
    // of them fails with the extraction error instead of "file not found".
    File.WriteAllLines(Path.Combine(outputDirectory, "extract_failures.txt"), failures.Order());
    Console.WriteLine($"extracted {count} matching files, {failures.Count} failed");
}

static async Task Probe(string packsDirectory, string candidatesFile, string suffix)
{
    var keys = (await PakList.ParseAsync(Path.Combine(packsDirectory, PakList.FileName)))
        .ToKeyInfoDictionary();
    var candidates = File.ReadAllLines(candidatesFile).Where(line => line.Length > 0);
    var hashes = await Pak.BruteforceFileHashesAsync(
        packsDirectory, keys, progress: null, keepUnresolvedHashes: false,
        additionalFileHashes: candidates);
    foreach (var pakPath in Directory.GetFiles(packsDirectory, "*.pak"))
    {
        var pakId = Path.GetFileNameWithoutExtension(pakPath);
        await using var pak = keys.TryGetValue(pakId, out var key)
            ? await Pak.ParseAsync(pakPath, key)
            : await Pak.ParseAsync(pakPath);
        foreach (var file in pak.Files.Values)
        {
            var name = hashes.GetValueOrDefault(file.Name) ?? file.Name;
            if (!name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            Console.WriteLine($"{pakId}/{file.FolderPath}{name}: class={file.ClassId:X8} " +
                $"flags={file.Flags:X} enc={file.IsEncrypted} comp={file.IsCompressed} " +
                $"dummy={!file.DontUseDummyWrite} size={file.UncompressedSize}/{file.CompressedSize}");
            try
            {
                var header = pak.OpenGbxFileHeader(file);
                Console.WriteLine($"  header: {header.Header.ClassId:X8} node={header.Node?.GetType().Name}");
            }
            catch (Exception error)
            {
                Console.WriteLine($"  header failed: {error}");
            }
            try
            {
                var gbx = await OpenGbx(pak, file, new ConsoleLogger());
                Console.WriteLine($"  parsed: {gbx.Node?.GetType().Name}");
            }
            catch (Exception error)
            {
                Console.WriteLine($"  parse failed: {error}");
            }
        }
    }
}


// Mirrors Pak.OpenGbxFileAsync but routes the cipher re-keying through GameRekey
// so the parent class ids fed to Blowfish are the game's, not GBX.NET's.
static async Task<Gbx> OpenGbx(Pak pak, PakFile file, Microsoft.Extensions.Logging.ILogger? logger)
{
    using var stream = pak.OpenFile(file, out var initializer);
    var settings = Settings() with { Logger = logger };
    if (initializer is not null && !file.DontUseDummyWrite)
    {
        var rekey = new GameRekey(initializer, logger);
        settings = settings with { EncryptionInitializer = rekey, Logger = rekey };
    }
    return await Gbx.ParseAsync(stream, settings);
}

static void CopyFile(Pak pak, PakFile file, Stream output)
{
    using var input = pak.OpenFile(file, out _);
    input.CopyTo(output);
}

static void WriteJson(string output, object value)
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
    var options = new JsonSerializerOptions
    {
        WriteIndented = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never
    };
    File.WriteAllText(output, JsonSerializer.Serialize(value, options) + "\n");
}

static void Fail(string message)
{
    Console.Error.WriteLine($"gbx-dump: {message}");
    Environment.Exit(2);
}

sealed class ConsoleLogger : Microsoft.Extensions.Logging.ILogger
{
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
    public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel logLevel) => true;
    public void Log<TState>(Microsoft.Extensions.Logging.LogLevel logLevel,
        Microsoft.Extensions.Logging.EventId eventId, TState state, Exception? exception,
        Func<TState, Exception?, string> formatter)
    {
        Console.WriteLine($"  [{logLevel}] {formatter(state, exception)}");
    }
}

// CMwNod::Archive (0x00924120) starts every node, referenced or direct, with a
// dummy write of CMwDeprecated::UnWrapClassId(parentClassInfo->classId) (with
// 0x07031000 mapped to 0x07001000); in a reading CClassicBufferCrypted that
// write only folds the bytes into the Blowfish CBC IV applied at the next
// 0x100-byte block (BlowfishCBC_Write, 0x00910CD0). GBX.NET feeds its own C#
// base class instead, which differs from the game's hierarchy for some classes
// (CHmsSoundSource, CMotionDayTime, ...) and only knows a few of the unwrapped
// ids; a wrong feed flips bits in exactly one 8-byte block, silently when it
// lands in float data. This wrapper resolves the class about to be read from
// GBX.NET's "NodeRef" log line (raw id) or the "{ClassName}" read scope
// (direct nodes), and feeds the value the game computes from the tables
// extracted out of TmForever.exe (tmnf_class_parents.txt, tmnf_unwrap_class_ids.txt).
sealed class GameRekey : IEncryptionInitializer, Microsoft.Extensions.Logging.ILogger
{
    static readonly Dictionary<uint, uint> Parents = LoadTable("TMNF_CLASS_PARENTS");
    static readonly Dictionary<uint, uint> Unwrap = LoadTable("TMNF_UNWRAP_CLASS_IDS");
    static readonly Dictionary<uint, uint> Wrap = Unwrap.ToDictionary(kv => kv.Value, kv => kv.Key);
    static readonly Dictionary<string, uint> GbxNetIds = typeof(Gbx).Assembly.GetTypes()
        .Where(t => typeof(IClass).IsAssignableFrom(t) && t.IsClass)
        .Select(t => (t.Name, Id: GBX.NET.Managers.ClassManager.GetId(t)))
        .Where(x => x.Id is not null)
        .GroupBy(x => x.Name)
        .ToDictionary(g => g.Key, g => g.First().Id!.Value);
    readonly EncryptionInitializer inner;
    readonly Microsoft.Extensions.Logging.ILogger? forward;
    readonly bool trace = Environment.GetEnvironmentVariable("GBXDUMP_TRACE_REKEY") is not null;
    uint? pending;

    public GameRekey(EncryptionInitializer inner, Microsoft.Extensions.Logging.ILogger? forward)
    {
        this.inner = inner;
        this.forward = forward;
    }

    static Dictionary<uint, uint> LoadTable(string variable)
    {
        var path = Environment.GetEnvironmentVariable(variable)
            ?? throw new InvalidOperationException($"{variable} is not set");
        var table = new Dictionary<uint, uint>();
        foreach (var line in File.ReadLines(path))
        {
            if (line.Length == 0 || line[0] == '#') continue;
            var parts = line.Split(' ');
            table[Convert.ToUInt32(parts[0], 16)] = Convert.ToUInt32(parts[1], 16);
        }
        return table;
    }

    // What the game's CMwNod::Archive writes for a node of class `raw` (as
    // stored in the file: either the runtime id or a deprecated one).
    // GBX.NET renames classes that later games moved between engines; the
    // TMNF file carries the old id (checked against CGameSkin::GetMwClassId).
    static readonly Dictionary<uint, uint> GbxNetToTmnf = new()
    {
        [0x090F4000] = 0x03031000, // CPlugGameSkin -> CGameSkin
    };

    static uint Feed(uint raw)
    {
        raw = GbxNetToTmnf.GetValueOrDefault(raw, raw);
        var id = Parents.ContainsKey(raw) ? raw
            : Wrap.TryGetValue(raw, out var wrapped) ? wrapped
            : throw new InvalidDataException($"class {raw:X8} is not in TMNF_CLASS_PARENTS");
        var parent = Parents[id];
        var fed = Unwrap.GetValueOrDefault(parent, parent);
        return fed == 0x07031000 ? 0x07001000 : fed;
    }

    public void Initialize(byte[] data, uint offset, uint count)
    {
        if (pending is not uint raw)
        {
            // Not a node start: a chunk-level dummy write the game also does
            // with plain data (CPlugSurfaceGeom 0x0900F004 feeds a float).
            inner.Initialize(data, offset, count);
            return;
        }
        pending = null;
        if (count != 4)
        {
            throw new InvalidDataException($"class {raw:X8}: expected 4-byte parent feed, got {count}");
        }
        var fed = Feed(raw);
        if (trace)
        {
            var theirs = BitConverter.ToUInt32(data, (int)offset);
            Console.Error.WriteLine($"rekey {raw:X8}: game {fed:X8}, GBX.NET {theirs:X8}{(fed != theirs ? " DIFF" : "")}");
        }
        inner.Initialize(BitConverter.GetBytes(fed), 0, 4);
    }

    // GbxReader.ReadNode opens a "{ClassName}" scope for every node it reads,
    // including direct nodes, which never get a "NodeRef" line.
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull
    {
        if (pending is null && state is IReadOnlyList<KeyValuePair<string, object?>> values)
        {
            foreach (var (key, value) in values)
            {
                if (key == "ClassName" && value is string name)
                {
                    pending = GbxNetIds.TryGetValue(name, out var id)
                        ? id
                        : throw new InvalidDataException($"GBX.NET class {name} has no class id");
                }
            }
        }
        return null;
    }

    public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel logLevel) => true;

    public void Log<TState>(Microsoft.Extensions.Logging.LogLevel logLevel,
        Microsoft.Extensions.Logging.EventId eventId, TState state, Exception? exception,
        Func<TState, Exception?, string> formatter)
    {
        forward?.Log(logLevel, eventId, state, exception, formatter);
        if (state is not IReadOnlyList<KeyValuePair<string, object?>> values) return;
        string? format = null;
        uint? classId = null;
        uint? rawClassId = null;
        foreach (var (key, value) in values)
        {
            switch (key)
            {
                case "{OriginalFormat}": format = value as string; break;
                case "ClassId": classId = Convert.ToUInt32(value); break;
                case "RawClassId": rawClassId = Convert.ToUInt32(value); break;
            }
        }
        if (format is null || !format.StartsWith("NodeRef #") || classId is null) return;
        pending = rawClassId ?? classId;
    }
}
// Root the reference paths in the JSON are relative to (the cache root for
// dump-tree); null keeps them absolute.
static class RefRoots
{
    public static string? Root;
}
