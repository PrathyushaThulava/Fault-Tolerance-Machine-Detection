# Fault-Tolerance-Machine-Detection

This project focuses on **fault tolerance in machine operations** using data-driven approaches and deep learning models. It monitors IoT equipment data, detects potential failures, and predicts Remaining Useful Life (RUL) to prevent unexpected breakdowns.

## 🚀 Features
- Real-time IoT equipment monitoring
- Fault detection using LSTM-based deep learning model
- Remaining Useful Life (RUL) prediction
- Dataset preprocessing and feature engineering
- Supports model persistence with `.pkl` and `.keras` files

## 📂 Project Structure
```

.
├── app.py                        # Main application entry
├── features.pkl                  # Preprocessed feature file
├── features_rul.pkl              # RUL-specific features
├── iot_equipment_monitoring_dataset.csv  # Dataset
├── lstm_failure_model.keras      # Trained LSTM model
├── README.md                     # Project documentation
└── .vscode/                      # Editor settings

````

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/PrathyushaThulava/Fault-Tolerance-Machine-Detection.git
cd Fault-Tolerance-Machine-Detection

# Install dependencies
pip install -r requirements.txt
````

## ▶️ Usage

```bash
python app.py
```

The app will start processing IoT data and predict machine health.

## 📊 Dataset

The dataset used is `iot_equipment_monitoring_dataset.csv`, which contains IoT sensor data for machine operations.

## 🤖 Model

* LSTM-based deep learning model trained to detect failures
* Serialized with `.keras` format for easy loading

## ✨ Future Improvements

* API integration for real-time IoT streaming
* Visualization dashboard
* Extended support for multiple machine types

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).


