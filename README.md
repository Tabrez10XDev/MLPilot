# 🚀 MLPilot  
### Automated Model Selection & Evaluation Tool

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

---

## ⚙️ Setup
Python is mandatory, for additional CLIs install respective languages

python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt
brew install libomp (mac)
---

## 🚀 Usage (Python CLI)

Train:
python3 -m python_cli.pilot train --data data/titanic.csv --target Survived

Predict:
python3 -m python_cli.pilot predict --model models/model.pkl --input data/test.csv

Evaluate:
python3 -m python_cli.pilot evaluate --model models/model.pkl --data data/titanic.csv --target Survived

Clean:
python3 -m python_cli.pilot clean

---

## 🌐 Multi-Language CLI Support

Python:
python3 -m python_cli.pilot train ...

C++:
mkdir build && cd build  
cmake ..  
make  
./mlpilot train --data data/titanic.csv --target Survived  

Bash:
./bash_cli/pilot.sh train --data data/titanic.csv --target Survived  

Go:
go run go_cli/pilot.go train --data data/titanic.csv --target Survived  

Or:
go build -o gopilot go_cli/pilot.go  
./gopilot train --data data/titanic.csv --target Survived  

---

## 📊 Output

- outputs/output.json → latest run summary  
- outputs/history_log.json → all runs  
- outputs/plots/ → generated graphs  

---

## 📈 Visualizations

- Model comparison chart  
- Confusion matrix  
- ROC curve  
- Feature importance  

---

## 🛠️ Dependency Check

MLPilot automatically checks required packages.

If missing:
pip install -r requirements.txt

---

## 🧩 Architecture

C++ / Bash / Go / Python CLI  
           ↓  
     python_cli (core engine)  
           ↓  
   ML pipeline + evaluation  

---

## 🎯 Key Idea

One backend, multiple interfaces.

---

## ⭐ Highlights

- Multi-language system design  
- Modular architecture  
- Real-world ML workflow  
- Clean CLI experience  
- Performance + visualization insights  
