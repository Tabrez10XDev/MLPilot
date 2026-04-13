#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct CommandLineOptions {
    std::string command;
    std::string data_path;
    std::string target_column;
    std::string model_path;
    std::string input_path;
    std::string output_path;
    std::string save_path;
    std::string metric;
    double test_size = 0.2;
    int seed = 42;
};

void print_main_help() {
    std::cout
        << "MLPilot C++ CLI Wrapper\n"
        << "Usage:\n"
        << "  ./mlpilot <command> [options]\n\n"
        << "Commands:\n"
        << "  train      Train models and save the best one\n"
        << "  predict    Load a saved model and run inference\n"
        << "  evaluate   Load a saved model and evaluate on labeled data\n\n"
        << "Examples:\n"
        << "  ./mlpilot train --data data/titanic.csv --target Survived\n"
        << "  ./mlpilot predict --model models/titanic_model_xxxxxxxx.pkl --input data/titanic_inference.csv\n"
        << "  ./mlpilot evaluate --model models/titanic_model_xxxxxxxx.pkl --data data/titanic.csv --target Survived\n";
}

void print_train_help() {
    std::cout
        << "Usage: ./mlpilot train --data <file> --target <column> [options]\n\n"
        << "Options:\n"
        << "  --data <file>         Path to training CSV file\n"
        << "  --target <column>     Target column name\n"
        << "  --test-size <value>   Test split size (default: 0.2)\n"
        << "  --seed <int>          Random seed (default: 42)\n"
        << "  --metric <name>       Primary metric for model selection\n"
        << "  --save <file>         Optional path to save trained model\n"
        << "  --output <file>       Output JSON report path\n";
}

void print_predict_help() {
    std::cout
        << "Usage: ./mlpilot predict --model <file> --input <file> [options]\n\n"
        << "Options:\n"
        << "  --model <file>        Path to saved model bundle\n"
        << "  --input <file>        Path to input CSV file for inference\n"
        << "  --output <file>       Output predictions CSV path\n";
}

void print_evaluate_help() {
    std::cout
        << "Usage: ./mlpilot evaluate --model <file> --data <file> --target <column>\n\n"
        << "Options:\n"
        << "  --model <file>        Path to saved model bundle\n"
        << "  --data <file>         Path to evaluation CSV file\n"
        << "  --target <column>     Target column name\n";
}

bool is_flag(const std::string& value) {
    return value.rfind("--", 0) == 0;
}

std::string require_value(const std::vector<std::string>& args, std::size_t& index, const std::string& flag) {
    if (index + 1 >= args.size() || is_flag(args[index + 1])) {
        throw std::runtime_error("Missing value for flag: " + flag);
    }
    ++index;
    return args[index];
}

CommandLineOptions parse_arguments(int argc, char* argv[]) {
    if (argc < 2) {
        print_main_help();
        throw std::runtime_error("No command provided.");
    }

    std::vector<std::string> args(argv + 1, argv + argc);
    CommandLineOptions options;
    options.command = args[0];

    if (options.command == "--help" || options.command == "-h") {
        print_main_help();
        std::exit(0);
    }

    if (options.command != "train" && options.command != "predict" && 
    options.command != "evaluate" && options.command != "clean") {
        throw std::runtime_error("Unsupported command: " + options.command);
    }

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& arg = args[i];

        if (arg == "--help" || arg == "-h") {
            if (options.command == "clean") {
            }
            else if (options.command == "train") {
                print_train_help();
            } else if (options.command == "predict") {
                print_predict_help();
            } else {
                print_evaluate_help();
            }
            std::exit(0);
        } else if (arg == "--data") {
            options.data_path = require_value(args, i, arg);
        } else if (arg == "--target") {
            options.target_column = require_value(args, i, arg);
        } else if (arg == "--model") {
            options.model_path = require_value(args, i, arg);
        } else if (arg == "--input") {
            options.input_path = require_value(args, i, arg);
        } else if (arg == "--output") {
            options.output_path = require_value(args, i, arg);
        } else if (arg == "--save") {
            options.save_path = require_value(args, i, arg);
        } else if (arg == "--metric") {
            options.metric = require_value(args, i, arg);
        } else if (arg == "--test-size") {
            options.test_size = std::stod(require_value(args, i, arg));
        } else if (arg == "--seed") {
            options.seed = std::stoi(require_value(args, i, arg));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.command == "train") {
        if (options.data_path.empty()) {
            throw std::runtime_error("train requires --data");
        }
        if (options.target_column.empty()) {
            throw std::runtime_error("train requires --target");
        }
    } else if (options.command == "predict") {
        if (options.model_path.empty()) {
            throw std::runtime_error("predict requires --model");
        }
        if (options.input_path.empty()) {
            throw std::runtime_error("predict requires --input");
        }
    } else if (options.command == "evaluate") {
        if (options.model_path.empty()) {
            throw std::runtime_error("evaluate requires --model");
        }
        if (options.data_path.empty()) {
            throw std::runtime_error("evaluate requires --data");
        }
        if (options.target_column.empty()) {
            throw std::runtime_error("evaluate requires --target");
        }
    }

    return options;
}

std::string quote_argument(const std::string& value) {
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"' || ch == '\\') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}


std::string build_python_command(const CommandLineOptions& options) {
    std::ostringstream command;
    command << "python3 -m python_cli.pilot " << options.command;

    if (options.command == "train") {
        command << " --data " << quote_argument(options.data_path)
                << " --target " << quote_argument(options.target_column)
                << " --test-size " << options.test_size
                << " --seed " << options.seed;

        if (!options.metric.empty()) {
            command << " --metric " << quote_argument(options.metric);
        }
        if (!options.save_path.empty()) {
            command << " --save " << quote_argument(options.save_path);
        }
        if (!options.output_path.empty()) {
            command << " --output " << quote_argument(options.output_path);
        }
        
        command << " --interface cpp";

    } else if (options.command == "predict") {
        command << " --model " << quote_argument(options.model_path)
                << " --input " << quote_argument(options.input_path);

        if (!options.output_path.empty()) {
            command << " --output " << quote_argument(options.output_path);
        }
    } else if (options.command == "evaluate") {
        command << " --model " << quote_argument(options.model_path)
                << " --data " << quote_argument(options.data_path)
                << " --target " << quote_argument(options.target_column);
    }

    return command.str();
}

int run_python_backend(const std::string& command) {
    std::cout << "[MLPilot C++ Wrapper] Running: " << command << "\n\n";
    return std::system(command.c_str());
}

int main(int argc, char* argv[]) {
    try {
        CommandLineOptions options = parse_arguments(argc, argv);
        const std::string command = build_python_command(options);
        const int exit_code = run_python_backend(command);

        if (exit_code != 0) {
            std::cerr << "C++ wrapper error: Python backend exited with code "
                      << exit_code << "\n";
            return exit_code;
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "C++ wrapper error: " << ex.what() << "\n\n";
        print_main_help();
        return 1;
    }
}
