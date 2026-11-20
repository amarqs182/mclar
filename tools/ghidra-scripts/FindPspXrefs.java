// FindPspXrefs.java — locate references to PSP data addresses and decompile
// their containing functions in the imported MC3 module.
//
// @category MidnightClub

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.symbol.Reference;
import ghidra.program.model.symbol.ReferenceIterator;

import java.util.LinkedHashSet;
import java.util.Set;

public class FindPspXrefs extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length == 0) {
            println("usage: FindPspXrefs <hex-address> [hex-address ...]");
            return;
        }

        DecompInterface decompiler = new DecompInterface();
        decompiler.openProgram(currentProgram);
        Memory memory = currentProgram.getMemory();

        for (String arg : args) {
            String normalized = arg.replace("_", "");
            if (normalized.startsWith("0x") || normalized.startsWith("0X"))
                normalized = normalized.substring(2);
            long value = Long.parseLong(normalized, 16);
            Address address = toAddr(value);
            println("=== references to " + address + " ===");

            try {
                byte[] bytes = new byte[16];
                int count = memory.getBytes(address, bytes);
                StringBuilder hex = new StringBuilder();
                for (int i = 0; i < count; i++)
                    hex.append(String.format("%02x", bytes[i] & 0xff));
                println("bytes (" + count + "): " + hex);
            }
            catch (Exception e) {
                println("bytes unavailable: " + e.getMessage());
            }

            Set<Function> functions = new LinkedHashSet<>();
            ReferenceIterator refs = currentProgram.getReferenceManager().getReferencesTo(address);
            while (refs.hasNext()) {
                Reference ref = refs.next();
                Function function = currentProgram.getFunctionManager()
                    .getFunctionContaining(ref.getFromAddress());
                println("ref: " + ref.getFromAddress() + " -> " + ref.getToAddress()
                    + (function == null ? " (no containing function)" :
                       " in " + function.getName() + "@" + function.getEntryPoint()));
                if (function != null) functions.add(function);
            }

            for (Function function : functions) {
                println("--- decompile " + function.getName() + "@" + function.getEntryPoint() + " ---");
                DecompileResults result = decompiler.decompileFunction(function, 60, monitor);
                if (result != null && result.decompileCompleted() &&
                    result.getDecompiledFunction() != null) {
                    println(result.getDecompiledFunction().getC());
                }
                else {
                    println("decompile failed: " +
                        (result == null ? "no result" : result.getErrorMessage()));
                }
            }
        }

        decompiler.dispose();
    }
}
