import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Pilot {
    private static void printHelp() {
        System.out.println("MLPilot Java CLI");
        System.out.println();
        System.out.println("Usage:");
        System.out.println("  java -cp java_cli Pilot <command> [options]");
        System.out.println();
        System.out.println("Commands:");
        System.out.println("  train      Train models and save the best one");
        System.out.println("  predict    Run inference using a saved model");
        System.out.println("  evaluate   Evaluate a saved model");
        System.out.println("  clean      Remove generated files");
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            printHelp();
            System.exit(1);
        }

        String command = args[0];

        if (command.equals("--help") || command.equals("-h")) {
            printHelp();
            return;
        }

        List<String> cmd = new ArrayList<>();
        cmd.add("python3");
        cmd.add("-m");
        cmd.add("python_cli.pilot");
        cmd.add(command);

        if (command.equals("train")) {
            cmd.addAll(Arrays.asList(Arrays.copyOfRange(args, 1, args.length)));
            cmd.add("--interface");
            cmd.add("java");
        } else if (
            command.equals("predict") ||
            command.equals("evaluate") ||
            command.equals("clean")
        ) {
            cmd.addAll(Arrays.asList(Arrays.copyOfRange(args, 1, args.length)));
        } else {
            System.err.println("Unsupported command: " + command);
            printHelp();
            System.exit(1);
        }

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.inheritIO();

        try {
            Process process = pb.start();
            int exitCode = process.waitFor();
            System.exit(exitCode);
        } catch (IOException | InterruptedException e) {
            System.err.println("Java wrapper error: " + e.getMessage());
            System.exit(1);
        }
    }
}