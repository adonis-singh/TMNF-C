// Exports decompiled C and disassembly for every function listed in a CSV
// (first column = VA like 0x00549C90). One .c and one .asm file per function.
// @category TMNF

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.listing.Listing;

public class ExportDecompilation extends GhidraScript {
	@Override
	protected void run() throws Exception {
		String[] args = getScriptArgs();
		if (args.length != 2) {
			throw new IllegalArgumentException(
				"usage: ExportDecompilation.java FUNCTION_CSV OUTPUT_DIRECTORY");
		}

		List<Long> vas = readVas(args[0]);
		File outDir = new File(args[1]);
		File cDir = new File(outDir, "c");
		File asmDir = new File(outDir, "asm");
		cDir.mkdirs();
		asmDir.mkdirs();

		FunctionManager fm = currentProgram.getFunctionManager();
		Listing listing = currentProgram.getListing();

		DecompInterface decomp = new DecompInterface();
		DecompileOptions options = new DecompileOptions();
		decomp.setOptions(options);
		decomp.toggleCCode(true);
		decomp.toggleSyntaxTree(true);
		decomp.setSimplificationStyle("decompile");
		if (!decomp.openProgram(currentProgram)) {
			throw new IllegalStateException("decompiler failed to open program");
		}

		int done = 0;
		for (long va : vas) {
			monitor.checkCancelled();
			Function fn = fm.getFunctionContaining(toAddr(va));
			if (fn == null) {
				throw new IllegalStateException("no function at " + hex(va));
			}
			String base = fileBase(va, fn);

			DecompileResults res = decomp.decompileFunction(fn, 120, monitor);
			String cText = res.decompileCompleted()
				? res.getDecompiledFunction().getC()
				: "// DECOMPILE FAILED: " + res.getErrorMessage() + "\n";
			try (PrintWriter w = new PrintWriter(new FileWriter(new File(cDir, base + ".c")))) {
				w.println("// " + hex(va) + "  " + fn.getName(true));
				w.println("// prototype: " + fn.getPrototypeString(true, true));
				w.print(cText);
			}

			try (PrintWriter w = new PrintWriter(new FileWriter(new File(asmDir, base + ".asm")))) {
				w.println("; " + hex(va) + "  " + fn.getName(true));
				InstructionIterator it = listing.getInstructions(fn.getBody(), true);
				while (it.hasNext()) {
					Instruction ins = it.next();
					Address a = ins.getAddress();
					StringBuilder sb = new StringBuilder();
					sb.append(hex(a.getUnsignedOffset())).append("  ");
					sb.append(ins.toString());
					Address[] flows = ins.getFlows();
					if (ins.getFlowType().isCall() && flows.length > 0) {
						Function tgt = fm.getFunctionAt(flows[0]);
						if (tgt != null) {
							sb.append("   ; -> ").append(tgt.getName(true));
						}
					}
					w.println(sb.toString());
				}
			}
			done++;
		}
		decomp.dispose();
		println("Decompiled " + done + " functions to " + outDir);
	}

	private List<Long> readVas(String path) throws Exception {
		List<Long> vas = new ArrayList<>();
		try (BufferedReader r = new BufferedReader(new FileReader(path))) {
			String line = r.readLine();
			while ((line = r.readLine()) != null) {
				if (line.isBlank()) {
					continue;
				}
				String first = line.split(",", 2)[0].replace("\"", "").trim();
				vas.add(Long.parseUnsignedLong(first.substring(2), 16));
			}
		}
		return vas;
	}

	private String fileBase(long va, Function fn) {
		String name = fn.getName(true)
			.replaceAll("[^A-Za-z0-9_]", "_");
		if (name.length() > 100) {
			name = name.substring(0, 100);
		}
		return String.format("%08X_%s", va, name);
	}

	private static String hex(long value) {
		return String.format("0x%08X", value);
	}
}
