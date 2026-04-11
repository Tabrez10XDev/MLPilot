#!/usr/bin/env bash

set -e

# -------- HELP --------
print_help() {
  echo "MLPilot Bash CLI"
  echo ""
  echo "Usage:"
  echo "  ./bash_cli/pilot.sh <command> [options]"
  echo ""
  echo "Commands:"
  echo "  train      Train models and save the best one"
  echo "  predict    Run inference using a saved model"
  echo "  evaluate   Evaluate a saved model"
  echo "  clean      Remove models and outputs"
  echo ""
  echo "Examples:"
  echo "  ./bash_cli/pilot.sh train --data data/titanic.csv --target Survived"
  echo "  ./bash_cli/pilot.sh predict --model models/model.pkl --input data/input.csv"
}

# -------- MAIN --------
if [ $# -lt 1 ]; then
  print_help
  exit 1
fi

COMMAND="$1"
shift

case "$COMMAND" in
  train)
    python3 -m python_cli.pilot train "$@" --interface bash
    ;;
  predict)
    python3 -m python_cli.pilot predict "$@"
    ;;
  evaluate)
    python3 -m python_cli.pilot evaluate "$@"
    ;;
  clean)
    python3 -m python_cli.pilot clean
    ;;
  --help|-h)
    print_help
    ;;
  *)
    echo "❌ Unsupported command: $COMMAND"
    print_help
    exit 1
    ;;
esac