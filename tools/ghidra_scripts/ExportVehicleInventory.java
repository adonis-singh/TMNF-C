// Exports the static call tree rooted at the resolved vehicle-force callback.
// @category TMNF

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.TreeMap;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.lang.OperandType;
import ghidra.program.model.lang.Register;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.listing.Listing;

public class ExportVehicleInventory extends GhidraScript {
	private static final long ROOT_VA = 0x007C7D40L;

	private record MapSymbol(String mangledName, String module, boolean isFunction) {}
	private record QueueItem(Function function, int depth) {}
	private record Counts(
		long instructionCount,
		long indirectCalls,
		long x87,
		long sseScalar,
		long ssePacked) {}

	private final Map<Long, MapSymbol> mapSymbols = new TreeMap<>();
	private final Map<Long, Function> reached = new TreeMap<>();
	private final Map<Long, Integer> minimumDepth = new HashMap<>();
	private final Map<Long, List<Function>> directCallees = new HashMap<>();
	private FunctionManager functionManager;
	private Listing listing;

	@Override
	protected void run() throws Exception {
		String[] args = getScriptArgs();
		if (args.length != 2) {
			throw new IllegalArgumentException(
				"usage: ExportVehicleInventory.java SYMBOLS_TSV OUTPUT_DIRECTORY");
		}

		readMapSymbols(args[0]);
		File outputDirectory = new File(args[1]);
		if (!outputDirectory.isDirectory()) {
			throw new IllegalArgumentException(
				"output directory does not exist: " + outputDirectory);
		}

		functionManager = currentProgram.getFunctionManager();
		listing = currentProgram.getListing();
		walkRoot();
		writeInventory(new File(outputDirectory, "vehicle_inventory.csv"));
		writeEdges(new File(outputDirectory, "vehicle_call_edges.csv"));
		println("Exported " + reached.size() + " vehicle call-tree functions");
	}

	private void readMapSymbols(String path) throws Exception {
		try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
			String header = reader.readLine();
			if (!"va\tmangled_name\tis_function\tmodule".equals(header)) {
				throw new IllegalArgumentException("unexpected TSV header: " + header);
			}

			String line;
			while ((line = reader.readLine()) != null) {
				String[] fields = line.split("\t", -1);
				if (fields.length != 4) {
					throw new IllegalArgumentException("invalid TSV row: " + line);
				}
				long va = Long.parseUnsignedLong(fields[0].substring(2), 16);
				mapSymbols.put(
					va, new MapSymbol(fields[1], fields[3], "1".equals(fields[2])));
			}
		}
	}

	private void walkRoot() throws Exception {
		Function root = functionManager.getFunctionAt(toAddr(ROOT_VA));
		if (root == null) {
			throw new IllegalStateException("root function missing at " + hex(ROOT_VA));
		}

		Queue<QueueItem> queue = new ArrayDeque<>();
		Set<Long> visited = new HashSet<>();
		queue.add(new QueueItem(root, 0));
		while (!queue.isEmpty()) {
			monitor.checkCancelled();
			QueueItem item = queue.remove();
			long va = item.function().getEntryPoint().getUnsignedOffset();
			if (!visited.add(va)) {
				continue;
			}

			MapSymbol mapSymbol = mapSymbols.get(va);
			if (mapSymbol == null || !mapSymbol.isFunction()) {
				throw new IllegalStateException("reached unmapped function at " + hex(va));
			}
			reached.put(va, item.function());
			minimumDepth.merge(va, item.depth(), Math::min);
			for (Function callee : getDirectCallees(item.function())) {
				queue.add(new QueueItem(callee, item.depth() + 1));
			}
		}
	}

	private List<Function> getDirectCallees(Function function) {
		long va = function.getEntryPoint().getUnsignedOffset();
		return directCallees.computeIfAbsent(va, ignored -> {
			Set<Long> functionModelCallees = new HashSet<>();
			for (Function callee : function.getCalledFunctions(monitor)) {
				functionModelCallees.add(callee.getEntryPoint().getUnsignedOffset());
			}

			Map<Long, Function> mapped = new TreeMap<>();
			InstructionIterator instructions =
				listing.getInstructions(function.getBody(), true);
			while (instructions.hasNext()) {
				Instruction instruction = instructions.next();
				if (!instruction.getFlowType().isCall() || isIndirectCall(instruction)) {
					continue;
				}
				for (Address target : instruction.getFlows()) {
					Function callee = functionManager.getFunctionAt(target);
					if (callee == null) {
						continue;
					}
					long calleeVa = callee.getEntryPoint().getUnsignedOffset();
					MapSymbol symbol = mapSymbols.get(calleeVa);
					if (functionModelCallees.contains(calleeVa)
						&& !callee.isExternal()
						&& symbol != null
						&& symbol.isFunction()) {
						mapped.put(calleeVa, callee);
					}
				}
			}
			return new ArrayList<>(mapped.values());
		});
	}

	private void writeInventory(File output) throws Exception {
		try (PrintWriter writer = new PrintWriter(new FileWriter(output))) {
			writer.println(
				"va,demangled_name,prototype,mangled_name,size_bytes,module,depth,"
				+ "direct_callees,indirect_call_sites,instruction_count,"
				+ "x87_count,sse_scalar_count,sse_packed_count");
			for (Map.Entry<Long, Function> entry : reached.entrySet()) {
				long va = entry.getKey();
				Function function = entry.getValue();
				MapSymbol symbol = mapSymbols.get(va);
				Counts counts = countInstructions(function);
				writer.println(String.join(",",
					csv(hex(va)),
					csv(function.getName(true)),
					csv(function.getPrototypeString(true, true)),
					csv(symbol.mangledName()),
					Long.toString(function.getBody().getNumAddresses()),
					csv(symbol.module()),
					Integer.toString(minimumDepth.get(va)),
					Integer.toString(getDirectCallees(function).size()),
					Long.toString(counts.indirectCalls()),
					Long.toString(counts.instructionCount()),
					Long.toString(counts.x87()),
					Long.toString(counts.sseScalar()),
					Long.toString(counts.ssePacked())));
			}
		}
	}

	private void writeEdges(File output) throws Exception {
		try (PrintWriter writer = new PrintWriter(new FileWriter(output))) {
			writer.println("caller_va,caller_name,callee_va,callee_name");
			for (Map.Entry<Long, Function> entry : reached.entrySet()) {
				for (Function callee : getDirectCallees(entry.getValue())) {
					writer.println(String.join(",",
						csv(hex(entry.getKey())),
						csv(entry.getValue().getName(true)),
						csv(hex(callee.getEntryPoint().getUnsignedOffset())),
						csv(callee.getName(true))));
				}
			}
		}
	}

	private Counts countInstructions(Function function) {
		long instructionCount = 0;
		long indirectCalls = 0;
		long x87 = 0;
		long sseScalar = 0;
		long ssePacked = 0;
		InstructionIterator instructions =
			listing.getInstructions(function.getBody(), true);
		while (instructions.hasNext()) {
			Instruction instruction = instructions.next();
			instructionCount++;
			String mnemonic = instruction.getMnemonicString().toUpperCase();
			if (instruction.getFlowType().isCall() && isIndirectCall(instruction)) {
				indirectCalls++;
			}
			if (mnemonic.startsWith("F")) {
				x87++;
			}
			else if (usesXmmRegister(instruction)) {
				if (mnemonic.contains("SS") || mnemonic.contains("SD")) {
					sseScalar++;
				}
				else if (mnemonic.contains("PS") || mnemonic.contains("PD")) {
					ssePacked++;
				}
			}
		}
		return new Counts(instructionCount, indirectCalls, x87, sseScalar, ssePacked);
	}

	private boolean isIndirectCall(Instruction instruction) {
		int operandType = instruction.getOperandType(0);
		return OperandType.isIndirect(operandType)
			|| OperandType.isRegister(operandType)
			|| OperandType.isDynamic(operandType)
			|| instruction.getFlows().length == 0;
	}

	private boolean usesXmmRegister(Instruction instruction) {
		for (Object object : instruction.getInputObjects()) {
			if (object instanceof Register register
				&& register.getName().toUpperCase().startsWith("XMM")) {
				return true;
			}
		}
		for (Object object : instruction.getResultObjects()) {
			if (object instanceof Register register
				&& register.getName().toUpperCase().startsWith("XMM")) {
				return true;
			}
		}
		return false;
	}

	private static String hex(long value) {
		return String.format("0x%08X", value);
	}

	private static String csv(String value) {
		return "\"" + value.replace("\"", "\"\"") + "\"";
	}
}
