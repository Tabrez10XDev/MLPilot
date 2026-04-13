# 🚀 MLPilot  
### Automated Model Selection & Evaluation Tool
🔗 GitHub Repository: https://github.com/Tabrez10XDev/MLPilot

MLPilot is a modular machine learning pipeline that automatically:
- selects the best model
- evaluates performance
- generates visual insights
- supports multiple CLI interfaces across different programming languages

---

## 🌐 Multi-Language CLI Support

MLPilot runs from **5 languages** — all routing to a single, unified Python ML backend. Write your CLI in whatever language fits your stack; the ML logic stays consistent.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](python_cli/)
[![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](cpp_cli/)
[![Bash](https://img.shields.io/badge/Bash-4EAA25?style=for-the-badge&logo=gnubash&logoColor=white)](bash_cli/)
[![Go](https://img.shields.io/badge/Go-00ADD8?style=for-the-badge&logo=go&logoColor=white)](go_cli/)
[![Java](https://img.shields.io/badge/Java-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white)](java_cli/)

> **Python is the base.** All other language wrappers (C++, Bash, Go, Java) delegate to the Python ML engine via subprocess calls, ensuring a single source of truth for model training, evaluation, and inference.

---

## ✨ Features
- 🤖 Automatic model selection (Logistic Regression, SVM, KNN, Random Forest, etc.)
- 📊 Performance evaluation (Accuracy, F1, RMSE, etc.)
- 📈 Visualization (Confusion Matrix, ROC Curve, Feature Importance)
- 🧠 Smart preprocessing (numeric + categorical handling)
- 💾 Model saving & loading
- 📜 Run history logging
- ⏱️ Performance & timing tracking
- 🌐 Multi-language CLI support: Python · C++ · Bash · Go · Java

---

## 📁 Project Structure

```
MLPilot/
│── python_cli/
│── cpp_cli/
│── bash_cli/
│── go_cli/
│── java_cli/
│── data/
│── models/
│── outputs/
│── requirements.txt
│── CMakeLists.txt
│── README.md
│── LICENSE
```

---

## ⚙️ Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Mac (for XGBoost):

```bash
brew install libomp
```

---

## 🚀 Usage

### 🐍 Python CLI

```bash
python3 -m python_cli.pilot train --data data/titanic.csv --target Survived
python3 -m python_cli.pilot predict --model models/<model>.pkl --input data/titanic_inference.csv
python3 -m python_cli.pilot evaluate --model models/<model>.pkl --data data/titanic.csv --target Survived
python3 -m python_cli.pilot clean
```

---

### ⚙️ C++ CLI

```bash
mkdir build && cd build
cmake ..
make
cd ..
./build/mlpilot train --data data/titanic.csv --target Survived
./build/mlpilot predict --model models/<model>.pkl --input data/titanic_inference.csv
./build/mlpilot evaluate --model models/<model>.pkl --data data/titanic.csv --target Survived
./build/mlpilot clean
```

---

### 🐚 Bash CLI

```bash
chmod +x bash_cli/pilot.sh
./bash_cli/pilot.sh train --data data/titanic.csv --target Survived
./bash_cli/pilot.sh predict --model models/<model>.pkl --input data/titanic_inference.csv
./bash_cli/pilot.sh evaluate --model models/<model>.pkl --data data/titanic.csv --target Survived
./bash_cli/pilot.sh clean
```

---

### 🟢 Go CLI

```bash
# Run directly
go run go_cli/pilot.go train --data data/titanic.csv --target Survived
go run go_cli/pilot.go predict --model models/<model>.pkl --input data/titanic_inference.csv
go run go_cli/pilot.go evaluate --model models/<model>.pkl --data data/titanic.csv --target Survived
go run go_cli/pilot.go clean

# Or build first
go build -o gopilot go_cli/pilot.go
./gopilot train --data data/titanic.csv --target Survived
```

---

### ☕ Java CLI

```bash
javac java_cli/Pilot.java
java -cp java_cli Pilot train --data data/titanic.csv --target Survived
java -cp java_cli Pilot predict --model models/<model>.pkl --input data/titanic_inference.csv
java -cp java_cli Pilot evaluate --model models/<model>.pkl --data data/titanic.csv --target Survived
java -cp java_cli Pilot clean
```

---

## 📊 Output

- `outputs/output.json`
- `outputs/history_log.json`
- `outputs/plots/`

---

## 📈 Visualizations

Automatically generated for every training run:

- **Model comparison chart** — compare all candidate models side by side
- **Confusion matrix** — evaluate classification performance
- **ROC curve** — visualize true/false positive trade-offs
- **Feature importance** — understand which features drive predictions

---

## 🚀 Advanced Features

- ⏱️ **Per-model training time tracking**  
  Compare performance vs. computational cost across all candidate models.

- 📊 **Model comparison visualization**  
  Automatically generates charts comparing all trained models.

- 🧠 **Best model explanation**  
  Provides reasoning for model selection based on metrics and performance.

- 🧾 **Run history tracking**  
  Maintains a persistent log of all training runs with metadata.

- 🔍 **Dataset-aware preprocessing**  
  Handles numerical and categorical data, with automatic feature encoding.

---

## 🧩 Architecture

```
Python · C++ · Bash · Go · Java  (CLI wrappers)
                  ↓
        python_cli (unified ML backend)
                  ↓
    scikit-learn Pipeline + ColumnTransformer
                  ↓
      Multi-model training & evaluation
                  ↓
     Best model selection (configurable metric)
                  ↓
      Outputs: model, plots, history log
```

### Design Decisions

- **Single ML backend, multiple interfaces** — All CLIs route to a unified Python engine for consistency and maintainability.
- **Pipeline-based architecture** — scikit-learn Pipelines and ColumnTransformers handle preprocessing and modeling in a single flow.
- **Automatic model selection** — Multiple models are trained and evaluated, with the best selected based on a configurable metric.
- **Graceful degradation of dependencies** — Optional dependencies like XGBoost are used only if available, preventing crashes.
- **Modular code structure** — Concerns are separated into commands, models, preprocessing, visualization, and utilities.
- **Reproducibility-first design** — Random seeds, dataset paths, and configurations are logged for every run.
- **Cross-platform compatibility** — Designed to run on macOS, Linux, and Windows with minimal setup.

---

## 🛡️ Robustness & Error Handling

MLPilot is designed to handle real-world imperfect data:

- Detects missing or invalid target columns
- Handles empty or corrupted datasets gracefully
- Prevents crashes from missing optional dependencies
- Provides clear, user-friendly error messages
- Validates input schema before training or inference

---

## 🧩 Extensibility

MLPilot is designed to be easily extendable:

- New models can be added with minimal changes
- Additional CLI languages can be integrated easily
- Visualization modules are plug-and-play
- Designed to support future integration of:
  - Hyperparameter tuning
  - REST APIs
  - Web dashboards

---

## 👥 Authors

- Tabrez Mohammed
- Ashish Ubale

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⭐ Highlights

- Multi-language system design (Python · C++ · Bash · Go · Java)
- Modular, extensible architecture
- Real-world ML workflow with robust error handling
- Clean CLI experience with consistent behavior across all languages
- Reproducible, logged training runs
