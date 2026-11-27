# Botnet Attack Detection and Mitigation in SDN

## Quick start (Windows)

1. Install Python 3.10 or 3.11 and VS Code.
2. Extract this ZIP file.
3. Open the extracted `Botnet_SDN_Project` folder in VS Code.
4. Open Terminal in VS Code and run:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

5. A browser tab opens. Go to **Dataset**.
6. Since the original dataset is not available, click **Use built-in demo dataset** to test the application.
7. Go to **Train Algorithms** and click **Train all four algorithms**.
8. Go to **Results** to view measured metrics and comparison charts.
9. Go to **Prediction & Mitigation** to view flagged records and download the CSV.

## Actual dataset

The reference document uses `UNSW_2018_IoT_Botnet_Full5pc_4.csv` and a target column named `attack`.
When you obtain the dataset, upload it from the Dataset page and select its label column.

Dataset source to investigate:
https://research.unsw.edu.au/projects/bot-iot-dataset

## Models

- Default Decision Tree
- Bayesian-optimized Decision Tree (uses scikit-optimize when installed; otherwise a randomized-search fallback is used)
- SVM
- 1D CNN (TensorFlow). If TensorFlow cannot be imported, the app uses an explicitly labelled MLP fallback; an MLP is not a CNN.

## Important limitations

- The built-in dataset is synthetic and only verifies the software workflow. Do not report its scores as results on the UNSW dataset.
- The mitigation module labels records and produces a CSV for review. It does not connect to an SDN controller, install OpenFlow rules, or block live traffic.
- The original reference document's reported accuracies are not guaranteed; actual values depend on dataset, preprocessing, split, and model settings.
- For a fair evaluation, fit preprocessing only on training data and avoid using identifiers or target-derived columns as features.
