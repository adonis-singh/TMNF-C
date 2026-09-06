// Disables analyzers that do not affect disassembly or call references.
// @category TMNF

import java.util.Map;

import ghidra.app.script.GhidraScript;

public class ConfigureAnalysis extends GhidraScript {
	@Override
	protected void run() throws Exception {
		Map<String, String> options = getCurrentAnalysisOptionsAndValues(currentProgram);
		disable(options, "Decompiler Parameter ID");
		disable(options, "Function ID");

		if (!options.containsKey("Demangler Microsoft")) {
			throw new IllegalStateException("Demangler Microsoft analyzer is unavailable");
		}
		setAnalysisOption(currentProgram, "Demangler Microsoft", "true");
	}

	private void disable(Map<String, String> options, String analyzer) {
		if (!options.containsKey(analyzer)) {
			throw new IllegalStateException("analyzer is unavailable: " + analyzer);
		}
		setAnalysisOption(currentProgram, analyzer, "false");
	}
}
