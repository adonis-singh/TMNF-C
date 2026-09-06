using GBX.NET;
using GBX.NET.Components;
using GBX.NET.Engines.Game;
using GBX.NET.Engines.Plug;
using GBX.NET.Engines.Scene;
using GBX.NET.Exceptions;
using GBX.NET.Interfaces.Game;
using GBX.NET.LZO;
using GBX.NET.PAK;
using System.Text.Json;

Gbx.LZO = new MiniLZO();

if (args.Length == 0)
{
    Fail("expected command: challenge, solid, block-info, extract, surface-materials, material-ids, or list");
}

switch (args[0])
{
    case "challenge":
        RequireArgs(args, 3);
        WriteChallenge(args[1], args[2]);
        break;
    case "solid":
        RequireArgs(args, 3);
        WriteSolid(args[1], args[2]);
        break;
    case "block-info":
        RequireArgs(args, 3);
        WriteBlockInfo(args[1], args[2]);
        break;
    case "extract":
        if (args.Length < 4)
        {
            Fail("usage: extract PACKS_DIR OUTPUT_DIR SUFFIX [SUFFIX...]");
        }
        // Optional "--candidates FILE" (one file name per line) supplies
        // names whose hashes GBX.NET's own scan does not resolve.
        {
            var rest = args[3..];
            string? candidates = null;
            var index = Array.IndexOf(rest, "--candidates");
            if (index >= 0)
            {
                if (index + 1 >= rest.Length)
                {
                    Fail("--candidates needs a file");
                }
                candidates = rest[index + 1];
                rest = rest.Where((_, i) => i != index && i != index + 1).ToArray();
            }
            await Extract(args[1], args[2], rest, candidates);
        }
        break;
    case "surface-materials":
        // Physical material ids of every *.Surface.Gbx under a directory,
        // one JSON object per file (analysis/materials.md).
        WriteSurfaceMaterials(args[1], args[2]);
        break;
    case "material-ids":
        // Physical surface id of every *.Material.Gbx under a directory.
        WriteMaterialIds(args[1], args[2]);
        break;
    case "list":
        RequireArgs(args, 3);
        await ListFiles(args[1], args[2]);
        break;
    default:
        Fail($"unknown command {args[0]}");
        break;
}

static void RequireArgs(string[] commandArgs, int count)
{
    if (commandArgs.Length != count)
    {
        Fail($"usage: {commandArgs[0]} INPUT OUTPUT");
    }
}

static void WriteChallenge(string input, string output)
{
    var map = Gbx.ParseNode<CGameCtnChallenge, IGameCtnChallengeTMF>(
        input, new GbxReadSettings { SafeSkippableChunks = true });
    var blocks = map.GetBlocks().Select((block, index) => new
    {
        index,
        model = block.Name,
        coord = new[] { block.Coord.X, block.Coord.Y, block.Coord.Z },
        direction = (int)block.Direction,
        directionName = block.Direction.ToString(),
        ground = block.IsGround,
        variant = block.Variant,
        subVariant = block.SubVariant,
        flags = block.Flags
    }).ToArray();
    WriteJson(output, new
    {
        source = Path.GetFullPath(input),
        blocks
    });
}

static void WriteSolid(string input, string output)
{
    var solid = Gbx.ParseNode<CPlugSolid>(
        input, new GbxReadSettings { SafeSkippableChunks = true });
    var tree = solid.Tree as CPlugTree
        ?? throw new InvalidDataException($"{input} has no CPlugTree");
    WriteJson(output, new
    {
        source = Path.GetFullPath(input),
        tree = SerializeTree(tree)
    });
}

static void WriteSurfaceMaterials(string directory, string output)
{
    var rows = new List<object>();
    foreach (var path in Directory.EnumerateFiles(
        directory, "*.Solid.Gbx", SearchOption.AllDirectories).Order())
    {
        var file = Path.GetRelativePath(directory, path);
        try
        {
            var solid = Gbx.ParseNode<CPlugSolid>(
                path, new GbxReadSettings { SafeSkippableChunks = true });
            if (solid.Tree is CPlugTree tree)
            {
                CollectSurfaces(tree, file, "", rows);
            }
        }
        catch (Exception error)
        {
            rows.Add(new
            {
                file,
                error = error.GetType().Name + ": " + error.Message
            });
        }
    }
    WriteJson(output, rows);
    Console.WriteLine($"{rows.Count} surfaces");
}

static void CollectSurfaces(
    CPlugTree tree, string file, string treePath, List<object> rows)
{
    var name = treePath + "/" + (tree.Name ?? "");
    if (tree.Surface is CPlugSurface surface)
    {
        var mesh = surface.Surf as CPlugSurface.Mesh;
        rows.Add(new
        {
            file,
            tree = name,
            surf = surface.Surf?.GetType().Name,
            materials = surface.Materials?.Select(material => new
            {
                surfaceId = material.SurfaceId?.ToString(),
                materialFile = material.MaterialFile?.GetFullPath(),
                materialSurfaceId = material.Material?.SurfaceId.ToString(),
            }).ToArray(),
            triangle = mesh is null || mesh.Triangles.Length == 0 ? null
                : JsonSerializer.Serialize(mesh.Triangles[0],
                    new JsonSerializerOptions { IncludeFields = true })
        });
    }
    foreach (var child in tree.Children)
    {
        CollectSurfaces(child, file, name, rows);
    }
    if (tree is CPlugTreeVisualMip mip)
    {
        foreach (var level in mip.Levels)
        {
            CollectSurfaces(level.Tree, file, name + "/mip", rows);
        }
    }
}

static void WriteMaterialIds(string directory, string output)
{
    var rows = new List<object>();
    foreach (var path in Directory.EnumerateFiles(
        directory, "*.Material.Gbx", SearchOption.AllDirectories).Order())
    {
        var file = Path.GetRelativePath(directory, path);
        try
        {
            var material = Gbx.ParseNode<CPlugMaterial>(
                path, new GbxReadSettings { SafeSkippableChunks = true });
            rows.Add(new { file, surfaceId = material.SurfaceId.ToString() });
        }
        catch (Exception error)
        {
            rows.Add(new { file, error = error.GetType().Name + ": " + error.Message });
        }
    }
    WriteJson(output, rows);
    Console.WriteLine($"{rows.Count} materials");
}

static void WriteBlockInfo(string input, string output)
{
    var info = Gbx.ParseNode(
        input, new GbxReadSettings { SafeSkippableChunks = true })
        as CGameCtnBlockInfo
        ?? throw new InvalidDataException($"{input} is not block info");
    WriteJson(output, new
    {
        source = Path.GetFullPath(input),
        model = info.Ident.Id,
        ground = SerializeMobils(info.GroundMobils),
        air = SerializeMobils(info.AirMobils)
    });
}

static object[][] SerializeMobils(External<CSceneMobil>[][]? rows)
{
    if (rows is null)
    {
        return [];
    }
    return rows.Select(row => row.Select(mobil =>
    {
        var node = mobil.GetNode(
            new GbxReadSettings { SafeSkippableChunks = true });
        var solidRef = node?.Item?.Solid;
        var solid = solidRef?.Tree as CPlugSolid;
        return (object)new
        {
            mobil = mobil.File?.GetFullPath(),
            solid = solidRef?.TreeFile?.GetFullPath(),
            tree = solid?.Tree is CPlugTree tree ? SerializeTree(tree) : null,
            objectLinks = node?.ObjectLink?.Select(link => new
            {
                solid = link.Mobil?.Item?.Solid?.TreeFile?.GetFullPath(),
                tree = link.Mobil?.Item?.Solid?.Tree is CPlugSolid linkSolid
                    && linkSolid.Tree is CPlugTree linkTree
                    ? SerializeTree(linkTree)
                    : null
            }).ToArray()
        };
    }).ToArray()).ToArray();
}

static object SerializeTree(CPlugTree tree)
{
    var visual = tree.Visual as CPlugVisualIndexedTriangles;
    var material = tree.Shader as CPlugMaterial;
    return new
    {
        name = tree.Name,
        material = GbxPath.GetFileNameWithoutExtension(
            tree.ShaderFile?.FilePath),
        materialTextures = material?.CustomMaterial?.Textures?
            .Select(texture => new
            {
                slot = texture.Name,
                texture = texture.TextureFile?.GetFullPath(),
                image = (texture.Texture as CPlugBitmap)?.ImageFile?.GetFullPath()
            })
            .ToArray(),
        visible = tree.IsVisible,
        transform = tree.Location.HasValue
            ? new[]
            {
                tree.Location.Value.XX, tree.Location.Value.XY,
                tree.Location.Value.XZ, tree.Location.Value.YX,
                tree.Location.Value.YY, tree.Location.Value.YZ,
                tree.Location.Value.ZX, tree.Location.Value.ZY,
                tree.Location.Value.ZZ, tree.Location.Value.TX,
                tree.Location.Value.TY, tree.Location.Value.TZ
            }
            : null,
        visual = visual is null ? null : SerializeVisual(visual),
        children = tree.Children.Select(SerializeTree).ToArray(),
        visualMip = tree is CPlugTreeVisualMip mip
            ? mip.Levels.Select(level => new
            {
                distance = level.FarZ,
                tree = SerializeTree(level.Tree)
            }).ToArray()
            : null
    };
}

static object SerializeVisual(CPlugVisualIndexedTriangles visual)
{
    Vec3[] positions;
    Vec3[] normals;
    Vec2[] texCoords;

    if (visual.VertexStreams.Count > 0)
    {
        var stream = visual.VertexStreams[0];
        positions = stream.Positions ?? [];
        normals = stream.Normals ?? [];
        texCoords = stream.UVs.Values.FirstOrDefault() ?? [];
    }
    else
    {
        positions = visual.Vertices.Select(vertex => vertex.Position).ToArray();
        normals = visual.Vertices
            .Select(vertex => vertex.Normal)
            .OfType<Vec3>()
            .ToArray();
        texCoords = visual.TexCoords.Length == 0
            ? []
            : visual.TexCoords[0].TexCoords
                .Select(coord => coord.UV)
                .ToArray();
    }

    return new
    {
        positions = positions.SelectMany(value =>
            new[] { value.X, value.Y, value.Z }).ToArray(),
        normals = normals.SelectMany(value =>
            new[] { value.X, value.Y, value.Z }).ToArray(),
        texCoords = texCoords.SelectMany(value =>
            new[] { value.X, value.Y }).ToArray(),
        indices = visual.IndexBuffer?.Indices ?? []
    };
}

static async Task Extract(
    string packsDirectory, string outputDirectory, string[] suffixes,
    string? candidatesFile)
{
    var packListPath = Path.Combine(packsDirectory, PakList.FileName);
    if (!File.Exists(packListPath))
    {
        Fail($"missing {packListPath}");
    }
    var keys = (await PakList.ParseAsync(packListPath)).ToKeyInfoDictionary();
    // GBX.NET resolves pak file names by hashing names it finds while
    // scanning; block infos nothing references (StadiumFabricPillarCornerOut)
    // stay unresolved unless their names are supplied here.
    var candidates = candidatesFile is null
        ? Enumerable.Empty<string>()
        : File.ReadAllLines(candidatesFile).Where(line => line.Length > 0);
    var hashes = await Pak.BruteforceFileHashesAsync(
        packsDirectory, keys, progress: null, keepUnresolvedHashes: false,
        additionalFileHashes: candidates);
    var selected = new HashSet<string>(
        suffixes, StringComparer.OrdinalIgnoreCase);
    var count = 0;

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
            if (!selected.Any(suffix =>
                name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)))
            {
                continue;
            }
            var destination = Path.Combine(
                outputDirectory, pakId, file.FolderPath, name);
            Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
            await using var stream = File.Create(destination);
            try
            {
                var gbx = await pak.OpenGbxFileAsync(file);
                if (gbx.Header is GbxHeaderUnknown)
                {
                    CopyFile(pak, file, stream);
                }
                else
                {
                    gbx.Save(stream);
                }
            }
            catch (Exception)
            {
                stream.SetLength(0);
                stream.Position = 0;
                try
                {
                    CopyFile(pak, file, stream);
                }
                catch (Exception error)
                {
                    // Every TMUF environment pak holds one entry (the
                    // TurboRoulette shader) whose zlib stream fails its
                    // checksum; nothing the viewer renders comes from it.
                    Console.WriteLine(
                        $"skipped {pakId}: {file.FolderPath}{name}: {error.Message}");
                    stream.SetLength(0);
                    continue;
                }
            }
            count++;
        }
    }
    Console.WriteLine($"extracted {count} matching files");
}

static async Task ListFiles(string packsDirectory, string output)
{
    var packListPath = Path.Combine(packsDirectory, PakList.FileName);
    if (!File.Exists(packListPath))
    {
        Fail($"missing {packListPath}");
    }
    var keys = (await PakList.ParseAsync(packListPath)).ToKeyInfoDictionary();
    var hashes = await Pak.BruteforceFileHashesAsync(
        packsDirectory, keys, keepUnresolvedHashes: false);
    var files = new List<string>();
    foreach (var pakPath in Directory.GetFiles(packsDirectory, "*.pak"))
    {
        var pakId = Path.GetFileNameWithoutExtension(pakPath);
        await using var pak = keys.TryGetValue(pakId, out var key)
            ? await Pak.ParseAsync(pakPath, key)
            : await Pak.ParseAsync(pakPath);
        files.AddRange(pak.Files.Values.Select(file =>
            Path.Combine(
                pakId,
                file.FolderPath,
                (hashes.GetValueOrDefault(file.Name) ?? file.Name)
                    .Replace('\\', Path.DirectorySeparatorChar))));
    }
    Directory.CreateDirectory(
        Path.GetDirectoryName(Path.GetFullPath(output))!);
    File.WriteAllLines(output, files.Order());
    Console.WriteLine($"listed {files.Count} files");
}

static void CopyFile(Pak pak, PakFile file, Stream output)
{
    using var input = pak.OpenFile(file, out _);
    input.CopyTo(output);
}

static void WriteJson(string output, object value)
{
    Directory.CreateDirectory(
        Path.GetDirectoryName(Path.GetFullPath(output))!);
    var options = new JsonSerializerOptions
    {
        WriteIndented = false
    };
    File.WriteAllText(output, JsonSerializer.Serialize(value, options) + "\n");
}

static void Fail(string message)
{
    Console.Error.WriteLine($"gbx-tool: {message}");
    Environment.Exit(2);
}
