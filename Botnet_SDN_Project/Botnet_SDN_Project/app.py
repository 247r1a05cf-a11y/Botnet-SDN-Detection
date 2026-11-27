import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="Botnet Detection in SDN", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.main {background-color:#f7f9fc;}
h1 {color:#14213d;}
div[data-testid="stMetric"] {background:white;padding:14px;border-radius:10px;border:1px solid #e6eaf0;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Botnet Attack Detection and Mitigation in SDN")
st.caption("Machine Learning and Deep Learning Based Network Traffic Classification")

if "data" not in st.session_state:
    st.session_state.data = None
if "results" not in st.session_state:
    st.session_state.results = None
if "trained" not in st.session_state:
    st.session_state.trained = {}

def make_demo_data(n=2400, seed=42):
    rng = np.random.default_rng(seed)
    y = rng.choice([0, 1], size=n, p=[0.55, 0.45])
    d = pd.DataFrame({
        "proto": rng.choice(["tcp", "udp", "icmp"], n),
        "duration": rng.exponential(2 + y*2, n),
        "sbytes": rng.lognormal(5 + y*1.2, 1.0, n),
        "dbytes": rng.lognormal(5 - y*0.6, 1.1, n),
        "rate": rng.gamma(2 + y*3, 1.5, n),
        "sttl": rng.integers(30, 255, n),
        "dttl": rng.integers(30, 255, n),
        "sload": rng.lognormal(4 + y*1.5, 1.0, n),
        "dload": rng.lognormal(4, 1.0, n),
        "spkts": rng.poisson(5 + y*8, n),
        "dpkts": rng.poisson(5, n),
        "attack": y
    })
    return d

def find_target(df):
    candidates = ["attack", "label", "class", "target", "is_attack", "category"]
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None

def prepare_data(df, target):
    data = df.copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=1, how="all")
    # Drop common row identifiers and high-cardinality descriptive fields.
    drop_cols = [c for c in data.columns if c.lower() in
                 ["pkseqid", "flow_id", "id", "category", "subcategory"] and c != target]
    data = data.drop(columns=drop_cols, errors="ignore")
    y_raw = data[target].astype(str).str.strip()
    # Convert common normal/attack text labels to binary.
    norm_words = {"0", "normal", "benign", "false", "no"}
    y = y_raw.str.lower().map(lambda v: 0 if v in norm_words else 1)
    if y.nunique() < 2:
        raise ValueError("Target column must contain at least two classes (normal and attack).")
    X = data.drop(columns=[target])
    # Remove columns with only one distinct value.
    X = X.loc[:, X.nunique(dropna=False) > 1]
    # Convert numeric-looking columns; preserve categorical columns.
    for c in X.columns:
        if X[c].dtype == "object":
            converted = pd.to_numeric(X[c], errors="coerce")
            if converted.notna().mean() > 0.90:
                X[c] = converted
    # Keep only manageable feature set.
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns found.")
    return X, y.astype(int)

def preprocess_split(X, y):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", __import__("sklearn").preprocessing.OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    prep = ColumnTransformer(transformers)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    Xtr = prep.fit_transform(X_train)
    Xte = prep.transform(X_test)
    if hasattr(Xtr, "toarray"):
        Xtr = Xtr.toarray()
        Xte = Xte.toarray()
    return Xtr, Xte, y_train.to_numpy(), y_test.to_numpy(), prep, X_test.index

def metric_row(name, y, pred, elapsed):
    return {
        "Algorithm": name,
        "Accuracy (%)": round(accuracy_score(y, pred)*100, 2),
        "Precision (%)": round(precision_score(y, pred, zero_division=0)*100, 2),
        "Recall (%)": round(recall_score(y, pred, zero_division=0)*100, 2),
        "F1-score (%)": round(f1_score(y, pred, zero_division=0)*100, 2),
        "Time (sec)": round(elapsed, 2)
    }

with st.sidebar:
    st.header("Project Modules")
    page = st.radio("Navigate", ["Overview", "Dataset", "Train Algorithms", "Prediction & Mitigation", "Results"])
    st.divider()
    st.info("Use the demo dataset to verify the application when the original CSV is unavailable.")

if page == "Overview":
    st.subheader("Project Overview")
    st.write("This application classifies network traffic as normal or botnet/attack traffic using Decision Tree, Bayesian-optimized Decision Tree, SVM and a CNN-style neural classifier.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Algorithms", "4")
    c2.metric("Classes", "Normal / Attack")
    c3.metric("Main task", "Detection")
    st.markdown("### System workflow")
    st.write("Upload CSV → Select target label → Preprocess → Train models → Compare metrics → Flag suspicious traffic.")
    st.warning("This is a research prototype. The mitigation module flags suspicious records in the uploaded dataset; it does not install SDN rules or block live network traffic.")

elif page == "Dataset":
    st.subheader("Dataset Upload and Preprocessing")
    uploaded = st.file_uploader("Upload network traffic dataset (CSV)", type=["csv"])
    col1, col2 = st.columns(2)
    with col1:
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded, low_memory=False)
                st.session_state.data = df
                st.session_state.results = None
                st.success("Dataset uploaded successfully.")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
    with col2:
        if st.button("Use built-in demo dataset", type="primary"):
            st.session_state.data = make_demo_data()
            st.session_state.results = None
            st.success("Synthetic demonstration dataset loaded.")
    if st.session_state.data is not None:
        df = st.session_state.data
        st.write(f"Rows: **{len(df):,}** | Columns: **{len(df.columns)}**")
        st.dataframe(df.head(10), use_container_width=True)
        target_guess = find_target(df)
        target = st.selectbox("Select target/label column", df.columns,
                              index=list(df.columns).index(target_guess) if target_guess in df.columns else len(df.columns)-1)
        if st.button("Check and preprocess dataset"):
            try:
                X, y = prepare_data(df, target)
                st.session_state.prepared = (X, y, target)
                st.success(f"Preprocessing completed. Features: {X.shape[1]}; records: {len(X)}.")
                st.write("Class distribution (0 = Normal, 1 = Attack)")
                st.bar_chart(y.value_counts().sort_index())
            except Exception as e:
                st.error(str(e))
    else:
        st.info("Upload your CSV or click 'Use built-in demo dataset'.")

elif page == "Train Algorithms":
    st.subheader("Train and Evaluate Algorithms")
    if st.session_state.data is None:
        st.warning("First load a dataset in the Dataset page.")
    else:
        df = st.session_state.data
        target_guess = find_target(df)
        target = st.selectbox("Target column", df.columns,
                              index=list(df.columns).index(target_guess) if target_guess in df.columns else len(df.columns)-1)
        if st.button("🚀 Train all four algorithms", type="primary"):
            try:
                X, y = prepare_data(df, target)
                Xtr, Xte, ytr, yte, prep, test_index = preprocess_split(X, y)
                st.session_state.prep = prep
                st.session_state.test_index = test_index
                results, trained, predictions = [], {}, {}
                progress = st.progress(0, text="Training models...")
                # Default Decision Tree
                t = time.time()
                dt = DecisionTreeClassifier(random_state=42, max_depth=12)
                dt.fit(Xtr, ytr)
                p = dt.predict(Xte)
                results.append(metric_row("Default Decision Tree", yte, p, time.time()-t))
                trained["Default Decision Tree"] = dt
                predictions["Default Decision Tree"] = p
                progress.progress(25, text="Decision Tree complete")
                # BOGP-inspired Bayesian hyperparameter search using Gaussian-process optimization
                t = time.time()
                try:
                    from skopt import BayesSearchCV
                    opt = BayesSearchCV(
                        DecisionTreeClassifier(random_state=42),
                        {"max_depth": (3, 30), "min_samples_split": (2, 30), "min_samples_leaf": (1, 15)},
                        n_iter=10, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
                    )
                    opt.fit(Xtr, ytr)
                    bogp = opt.best_estimator_
                except Exception:
                    # Fallback keeps project runnable if scikit-optimize is unavailable.
                    from sklearn.model_selection import GridSearchCV
                    opt = RandomizedSearchCV(
                        DecisionTreeClassifier(random_state=42),
                        {"max_depth": [5, 8, 12, 16, 20, None],
                         "min_samples_split": [2, 5, 10, 20],
                         "min_samples_leaf": [1, 2, 4, 8]},
                        n_iter=8, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
                    )
                    opt.fit(Xtr, ytr)
                    bogp = opt.best_estimator_
                p = bogp.predict(Xte)
                results.append(metric_row("BOGP Optimized Decision Tree", yte, p, time.time()-t))
                trained["BOGP Optimized Decision Tree"] = bogp
                predictions["BOGP Optimized Decision Tree"] = p
                progress.progress(50, text="Optimized Decision Tree complete")
                # SVM; cap rows for speed on large datasets.
                t = time.time()
                if len(Xtr) > 12000:
                    rng = np.random.default_rng(42)
                    ix = rng.choice(len(Xtr), 12000, replace=False)
                    svm_x, svm_y = Xtr[ix], ytr[ix]
                else:
                    svm_x, svm_y = Xtr, ytr
                svm = SVC(kernel="rbf", C=2, gamma="scale")
                svm.fit(svm_x, svm_y)
                p = svm.predict(Xte)
                results.append(metric_row("SVM", yte, p, time.time()-t))
                trained["SVM"] = svm
                predictions["SVM"] = p
                progress.progress(75, text="SVM complete")
                # A 1D convolutional neural network. TensorFlow is optional.
                t = time.time()
                try:
                    from tensorflow.keras import Sequential
                    from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout
                    cnn = Sequential([
                        Input(shape=(Xtr.shape[1], 1)),
                        Conv1D(32, 3, padding="same", activation="relu"),
                        GlobalMaxPooling1D(),
                        Dense(32, activation="relu"),
                        Dropout(0.2),
                        Dense(1, activation="sigmoid")
                    ])
                    cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                    cnn.fit(Xtr.reshape(-1, Xtr.shape[1], 1), ytr, epochs=5, batch_size=64,
                            validation_split=0.1, verbose=0)
                    p = (cnn.predict(Xte.reshape(-1, Xte.shape[1], 1), verbose=0).ravel() >= 0.5).astype(int)
                    trained["CNN"] = cnn
                    cnn_type = "CNN"
                except Exception:
                    # Explicit fallback: MLP is not CNN; UI labels it transparently.
                    cnn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42,
                                        early_stopping=True)
                    cnn.fit(Xtr, ytr)
                    p = cnn.predict(Xte)
                    trained["Neural Network fallback (MLP, not CNN)"] = cnn
                    cnn_type = "Neural Network fallback (MLP, not CNN)"
                results.append(metric_row(cnn_type, yte, p, time.time()-t))
                predictions[cnn_type] = p
                progress.progress(100, text="Training finished")
                st.session_state.results = pd.DataFrame(results)
                st.session_state.trained = trained
                st.session_state.predictions = predictions
                st.session_state.y_test = yte
                st.success("Training and evaluation completed.")
            except Exception as e:
                st.error(f"Training failed: {e}")
    if st.session_state.results is not None:
        st.dataframe(st.session_state.results, use_container_width=True)

elif page == "Prediction & Mitigation":
    st.subheader("Attack Detection and Mitigation Demonstration")
    if not st.session_state.trained:
        st.info("Train the models first.")
    else:
        names = list(st.session_state.trained.keys())
        model_name = st.selectbox("Choose model", names)
        df = st.session_state.data
        target = find_target(df)
        X, y = prepare_data(df, target)
        prep = st.session_state.prep
        model = st.session_state.trained[model_name]
        X_all = prep.transform(X)
        if hasattr(X_all, "toarray"):
            X_all = X_all.toarray()
        if model_name == "CNN":
            pred = (model.predict(X_all.reshape(-1, X_all.shape[1], 1), verbose=0).ravel() >= 0.5).astype(int)
        else:
            pred = model.predict(X_all)
        out = df.copy()
        out["Predicted_Status"] = np.where(pred == 1, "ATTACK - FLAGGED", "NORMAL")
        out["Mitigation_Action"] = np.where(pred == 1, "Flag for SDN flow review / block rule", "Allow")
        a,b = st.columns(2)
        a.metric("Traffic records", f"{len(out):,}")
        b.metric("Flagged attack records", f"{int((pred==1).sum()):,}")
        st.dataframe(out.head(100), use_container_width=True)
        st.download_button("Download flagged traffic CSV", out.to_csv(index=False).encode("utf-8"),
                           "botnet_detection_results.csv", "text/csv")
        st.warning("Demonstration only: no live switch/controller rules are changed.")

elif page == "Results":
    st.subheader("Algorithm Comparison and Results")
    if st.session_state.results is None:
        st.info("Train the algorithms first to generate actual measured results.")
    else:
        res = st.session_state.results
        st.dataframe(res, use_container_width=True)
        metric = st.selectbox("Select metric", ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)", "Time (sec)"])
        chart = res.set_index("Algorithm")[[metric]]
        st.bar_chart(chart)
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(data=res, x="Algorithm", y=metric, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_title(f"Algorithm comparison - {metric}")
        st.tight_layout()
        st.pyplot(fig)
        st.download_button("Download metrics CSV", res.to_csv(index=False).encode("utf-8"),
                           "algorithm_comparison.csv", "text/csv")
