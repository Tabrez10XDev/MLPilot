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

- Model comparison chart
- Confusion matrix
- ROC curve
- Feature importance

---

## 🧩 Architecture

```
C++ / Bash / Go / Python CLI
           ↓
     python_cli
           ↓
   ML pipeline
```

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
- Modular architecture
- Real-world ML workflow
- Clean CLI experience
