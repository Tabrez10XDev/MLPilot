# 🚀 MLPilot  
### Automated Model Selection & Evaluation Tool
🔗 GitHub Repository: https://github.com/Tabrez10XDev/MLPilot

MLPilot is a modular machine learning pipeline that automatically:
- selects the best model
- evaluates performance
- generates visual insights
- supports multiple CLI interfaces across different programming languages

---

## ✨ Features
- 🤖 Automatic model selection (Logistic Regression, SVM, KNN, Random Forest, etc.)
- 📊 Performance evaluation (Accuracy, F1, RMSE, etc.)
- 📈 Visualization (Confusion Matrix, ROC Curve, Feature Importance)
- 🧠 Smart preprocessing (numeric + categorical handling)
- 💾 Model saving & loading
- 📜 Run history logging
- ⏱️ Performance & timing tracking
- 🌐 Multi-language CLI support:
  - Python
  - C++
  - Bash
  - Go

---

## 📁 Project Structure

```
MLPilot/
│── python_cli/
│── cpp_cli/
│── bash_cli/
│── go_cli/
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

**Train:**
```bash
python3 -m python_cli.pilot train --data data/titanic.csv --target Survived
```

**Predict:**
```bash
python3 -m python_cli.pilot predict --model models/model.pkl --input data/test.csv
```

**Evaluate:**
```bash
python3 -m python_cli.pilot evaluate --model models/model.pkl --data data/titanic.csv --target Survived
```

**Clean:**
```bash
python3 -m python_cli.pilot clean
```

---

## 🌐 Multi-Language CLI Support

All CLIs (Python, C++, Bash, Go) route to a unified Python ML engine, ensuring consistency and maintainability across interfaces.

**Python:**
```bash
python3 -m python_cli.pilot train ...
```

**C++:**
```bash
mkdir build && cd build
cmake ..
make
./mlpilot train --data data/titanic.csv --target Survived
```

**Bash:**
```bash
./bash_cli/pilot.sh train --data data/titanic.csv --target Survived
```

**Go:**
```bash
go run go_cli/pilot.go train --data data/titanic.csv --target Survived
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
C++ / Bash / Go / Python CLI
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

- Multi-language system design
- Modular, extensible architecture
- Real-world ML workflow with robust error handling
- Clean CLI experience across Python, C++, Bash, and Go
- Reproducible, logged training runs
