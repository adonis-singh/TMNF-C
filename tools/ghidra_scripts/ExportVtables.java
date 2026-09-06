// Dumps MSVC vftables (map symbols containing "6B@") as ordered lists of the
// function pointers they hold, and lists every indirect call site inside the
// physics tree with local decompiler context to aid manual resolution.
// @category TMNF

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.mem.MemoryAccessException;

public class ExportVtables extends GhidraScript {
	private final TreeMap<Long, String> codeSymbols = new TreeMap<>();

	@Override
	protected void run() throws Exception {
		String[] args = getScriptArgs();
		if (args.length != 2) {
			throw new IllegalArgumentException(
				"usage: ExportVtables.java RAW_MAP OUTPUT_DIRECTORY");
		}
		File rawMap = new File(args[0]);
		File outDir = new File(args[1]);
		outDir.mkdirs();

		FunctionManager fm = currentProgram.getFunctionManager();

		// Pattern: " 0001:000xxxxx  ?name  0xVA f  lib:obj"  (code symbols)
		//          " 0002:...       ??_7Foo@@6B@  0xVA      lib:obj" (vftable data)
		Pattern rowPattern = Pattern.compile(
			"^\\s+\\d{4}:[0-9a-fA-F]+\\s+(\\S+)\\s+([0-9a-fA-F]{6,8})\\s+(f\\s+)?");

		List<long[]> vftables = new ArrayList<>(); // {va}
		List<String> vftableNames = new ArrayList<>();

		try (BufferedReader r = new BufferedReader(new FileReader(rawMap))) {
			String line;
			while ((line = r.readLine()) != null) {
				Matcher m = rowPattern.matcher(line);
				if (!m.find()) {
					continue;
				}
				String name = m.group(1);
				long va = Long.parseUnsignedLong(m.group(2), 16);
				boolean isFunc = m.group(3) != null;
				if (isFunc) {
					codeSymbols.put(va, name);
				}
				if (name.contains("6B@")) {
					vftables.add(new long[] { va });
					vftableNames.add(name);
				}
			}
		}

		File vtDir = new File(outDir, "vtables");
		vtDir.mkdirs();
		int slots = 0;
		try (PrintWriter idx = new PrintWriter(new FileWriter(new File(vtDir, "_index.csv")))) {
			idx.println("vftable_va,vftable_name,slot,entry_va,entry_name");
			for (int i = 0; i < vftables.size(); i++) {
				monitor.checkCancelled();
				long base = vftables.get(i)[0];
				String vfName = vftableNames.get(i);
				Address a = toAddr(base);
				int slot = 0;
				while (true) {
					if (slot > 0 && codeSymbols.containsKey(a.getUnsignedOffset())) {
						break; // ran into the next symbol
					}
					long ptr;
					try {
						ptr = Integer.toUnsignedLong(getInt(a));
					}
					catch (MemoryAccessException e) {
						break;
					}
					Function tgt = fm.getFunctionAt(toAddr(ptr));
					String tgtName = tgt != null ? tgt.getName(true)
						: codeSymbols.getOrDefault(ptr, null);
					if (tgtName == null) {
						break; // not a function pointer -> end of vftable
					}
					idx.println(String.join(",",
						hex(base), csv(vfName), Integer.toString(slot),
						hex(ptr), csv(tgtName)));
					slot++;
					slots++;
					a = a.add(4);
				}
			}
		}
		println("Dumped " + vftables.size() + " vftables, " + slots + " slots to " + vtDir);
	}

	private static String hex(long value) {
		return String.format("0x%08X", value);
	}

	private static String csv(String value) {
		return "\"" + value.replace("\"", "\"\"") + "\"";
	}
}
