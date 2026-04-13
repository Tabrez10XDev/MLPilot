package main

import (
	"fmt"
	"os"
	"os/exec"
)

func printHelp() {
	fmt.Println("MLPilot Go CLI")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  go run go_cli/pilot.go <command> [options]")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  train      Train models and save the best one")
	fmt.Println("  predict    Run inference using a saved model")
	fmt.Println("  evaluate   Evaluate a saved model")
	fmt.Println("  clean      Remove generated files")
	fmt.Println()
	fmt.Println("Train options (forwarded to Python backend):")
	fmt.Println("  --max-models <N>         Limit number of candidate models")
	fmt.Println("  --max-train-seconds <S>  Timeout for full model search (default: 180)")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  go run go_cli/pilot.go train --data data/titanic.csv --target Survived")
	fmt.Println("  go run go_cli/pilot.go train --data data/big.csv --target Survived --max-models 3")
	fmt.Println("  go run go_cli/pilot.go predict --model models/model.pkl --input data/titanic_inference.csv")
	fmt.Println("  go run go_cli/pilot.go evaluate --model models/model.pkl --data data/titanic.csv --target Survived")
	fmt.Println("  go run go_cli/pilot.go clean")
}

func main() {
	if len(os.Args) < 2 {
		printHelp()
		os.Exit(1)
	}

	command := os.Args[1]

	if command == "--help" || command == "-h" {
		printHelp()
		return
	}

	valid := map[string]bool{
		"train":    true,
		"predict":  true,
		"evaluate": true,
		"clean":    true,
	}

	if !valid[command] {
		fmt.Printf("Unsupported command: %s\n\n", command)
		printHelp()
		os.Exit(1)
	}

	args := []string{"-m", "python_cli.pilot", command}
	args = append(args, os.Args[2:]...)

	if command == "train" {
		args = append(args, "--interface", "go")
	}

	cmd := exec.Command("python3", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin

	err := cmd.Run()
	if err != nil {
		fmt.Printf("Go wrapper error: %v\n", err)
		os.Exit(1)
	}
}
