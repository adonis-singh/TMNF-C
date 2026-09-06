// Applies linker map symbols before auto-analysis.
// @category TMNF

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import ghidra.app.cmd.disassemble.DisassembleCommand;
import ghidra.app.cmd.function.CreateFunctionCmd;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressSet;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.symbol.SourceType;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolTable;

public class ApplyMapSymbols extends GhidraScript {
	private record MapSymbol(Address address, String name, boolean isFunction) {}

	@Override
	protected void run() throws Exception {
		String[] args = getScriptArgs();
		if (args.length != 1) {
			throw new IllegalArgumentException("usage: ApplyMapSymbols.java SYMBOLS_TSV");
		}

		List<MapSymbol> symbols = readSymbols(args[0]);
		SymbolTable symbolTable = currentProgram.getSymbolTable();
		Set<Address> functionEntries = new LinkedHashSet<>();

		for (MapSymbol mapSymbol : symbols) {
			if (!currentProgram.getMemory().contains(mapSymbol.address())) {
				throw new IllegalStateException("symbol outside memory: " + mapSymbol);
			}

			Symbol symbol = symbolTable.createLabel(
				mapSymbol.address(), mapSymbol.name(), SourceType.IMPORTED);
			symbol.setPrimary();
			if (mapSymbol.isFunction()) {
				functionEntries.add(mapSymbol.address());
			}
		}

		AddressSet seeds = new AddressSet();
		for (Address entry : functionEntries) {
			seeds.add(entry);
		}
		DisassembleCommand disassemble = new DisassembleCommand(seeds, null, true);
		if (!disassemble.applyTo(currentProgram, monitor)) {
			throw new IllegalStateException("disassembly failed: " + disassemble.getStatusMsg());
		}

		FunctionManager functionManager = currentProgram.getFunctionManager();
		List<Address> descendingEntries = new ArrayList<>(functionEntries);
		descendingEntries.sort(Collections.reverseOrder());
		int created = 0;
		for (Address entry : descendingEntries) {
			monitor.checkCancelled();
			if (functionManager.getFunctionAt(entry) != null) {
				continue;
			}
			CreateFunctionCmd createFunction = new CreateFunctionCmd(entry);
			if (!createFunction.applyTo(currentProgram, monitor)) {
				throw new IllegalStateException(
					"function creation failed at " + entry + ": " +
					createFunction.getStatusMsg());
			}
			created++;
		}

		println("Applied " + symbols.size() + " symbols and created " +
			created + " functions");
	}

	private List<MapSymbol> readSymbols(String path) throws Exception {
		List<MapSymbol> symbols = new ArrayList<>();
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
				Address address = toAddr(Long.parseUnsignedLong(fields[0].substring(2), 16));
				symbols.add(new MapSymbol(address, fields[1], "1".equals(fields[2])));
			}
		}
		return symbols;
	}
}
