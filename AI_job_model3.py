import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1) CONFIG — SET YOUR DATA FILE HERE
# ============================================================
DATA_PATH = r"C:\Users\adity\OneDrive\Documents\Python_Projects\AI_models\ai_job_prediction_model\job_applicants_300.csv"  # <-- change if needed
SHEET_NAME = None                 # If Excel, put sheet name or leave None
SEED = 42
EPOCHS = 20
LR = 0.03
L2 = 0.01                         # L2 weight decay
THRESHOLD = 0.5                   # decision threshold for classification
MAX_ROWS_TO_PRINT = 10            # Adjusted to 10 for your random sample size
SAVE_PREDICTIONS = False          # set True to save per-epoch tables

# Initial weights: [bias, w_cgpa, w_exp]
w_init = np.array([-8.0, 2.0, 0.6], dtype=float)

pd.options.display.float_format = "{:.3f}".format


# ============================================================
# 2) DATA LOADING
# ============================================================
def load_dataset(path, sheet_name=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        engine = "openpyxl" if ext == ".xlsx" else "xlrd"
        df = pd.read_excel(path, sheet_name=sheet_name, engine=engine)
    else:
        raise ValueError("Unsupported file type. Use .csv, .xlsx, or .xls")

    # Normalize column names and ensure required columns are present
    cols = {c.lower().strip(): c for c in df.columns}
    required = ["cgpa", "experience", "label"]
    for r in required:
        if r not in cols:
            raise ValueError(f"Missing column '{r}' in the dataset. "
                             f"Found columns: {list(df.columns)}")

    # Standardize column names
    df = df.rename(columns={
        cols["cgpa"]: "cgpa",
        cols["experience"]: "experience",
        cols["label"]: "label"
    })

    # Clean types
    df["cgpa"] = pd.to_numeric(df["cgpa"], errors="coerce")
    df["experience"] = pd.to_numeric(df["experience"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")

    # Drop NaNs and enforce binary labels
    df = df.dropna()
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]

    return df[["cgpa", "experience", "label"]]


# ============================================================
# 3) MODEL FUNCTIONS (Logistic Regression)
# ============================================================
def sigmoid(z):
    # Clamp to avoid overflow in exp for large magnitude z
    z = np.clip(z, -709, 709)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X, w):
    return sigmoid(X @ w)


def bce_loss(y, p, eps=1e-12):
    """
    Binary cross entropy for each sample (no mean).
    Correct form: -[y*log(p) + (1-y)*log(1-p)]
    """
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def mean_bce(X, y, w):
    p = predict_proba(X, w)
    return bce_loss(y, p).mean()


def grad_bce(X, y, w, l2=L2):
    p = predict_proba(X, w)
    grad = (X.T @ (p - y)) / len(y)

    # L2 regularization (do not regularize bias term)
    reg_mask = np.array([0, 1, 1], dtype=float)
    grad += l2 * (w * reg_mask)
    return grad


def accuracy(X, y, w, threshold=THRESHOLD):
    p = predict_proba(X, w)
    return ((p >= threshold).astype(int) == y).mean()


# ============================================================
# 4) SPLIT + DESIGN MATRIX
# ============================================================
def train_test_split(df, ratio=0.8, seed=SEED):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train = int(ratio * len(df))
    return df[:n_train], df[n_train:]


def build_design(df):
    X = np.column_stack([np.ones(len(df)), df["cgpa"].values, df["experience"].values])
    y = df["label"].values.astype(int)
    return X, y


# ============================================================
# 5) PREDICTION TABLE HELPERS
# ============================================================
def predictions_dataframe(df, w, name, threshold=THRESHOLD):
    X, y = build_design(df)
    z = X @ w
    p = sigmoid(z)
    pred = (p >= threshold).astype(int)
    correct = (pred == y).astype(int)
    return pd.DataFrame({
        "set": name,
        "cgpa": df["cgpa"].values,
        "experience": df["experience"].values,
        "z": z,
        "prob": p,
        "pred": pred,
        "label": y,
        "correct": correct
    })


def print_table(df, title):
    print(f"\n{title}")
    # Print the provided rows directly (handled via sample in the training loop)
    print(df.to_string(index=False))


# ============================================================
# 6) TRAINING LOOP
# ============================================================
def train_model(df, epochs, lr, w0, l2=L2, threshold=THRESHOLD):
    df_train, df_test = train_test_split(df)
    X_train, y_train = build_design(df_train)
    X_test, y_test = build_design(df_test)

    w = w0.copy()
    history = {"train_loss": [], "test_loss": []}

    for ep in range(1, epochs + 1):
        grad = grad_bce(X_train, y_train, w, l2=l2)
        w -= lr * grad

        tr_loss = mean_bce(X_train, y_train, w)
        te_loss = mean_bce(X_test, y_test, w)
        tr_acc = accuracy(X_train, y_train, w, threshold=threshold)
        te_acc = accuracy(X_test, y_test, w, threshold=threshold)

        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)

        print(
            f"\n======================================================\n"
            f"Epoch {ep}: "
            f"Train Loss {tr_loss:.4f} | Acc {tr_acc:.3f} | "
            f"Test Loss {te_loss:.4f} | Acc {te_acc:.3f} | "
            f"w={w}"
        )

        # Generate predictions for all rows
        df_tr_pred = predictions_dataframe(df_train, w, "TRAIN", threshold=threshold)
        df_te_pred = predictions_dataframe(df_test, w, "TEST", threshold=threshold)
        
        # Take a random 10-row sample to display on the terminal
        df_tr_sample = df_tr_pred.sample(n=min(MAX_ROWS_TO_PRINT, len(df_tr_pred)))
        df_te_sample = df_te_pred.sample(n=min(MAX_ROWS_TO_PRINT, len(df_te_pred)))
        
        print_table(df_tr_sample, title=f"Train Predictions (10 Random Samples) — Epoch {ep}")
        print_table(df_te_sample, title=f"Test Predictions (10 Random Samples) — Epoch {ep}")

        if SAVE_PREDICTIONS:
            os.makedirs("epoch_preds", exist_ok=True)
            df_tr_pred.to_csv(f"epoch_preds/train_epoch_{ep}.csv", index=False)
            df_te_pred.to_csv(f"epoch_preds/test_epoch_{ep}.csv", index=False)

    return w, history, df_train, df_test


# ============================================================
# 7) PLOT LOSS
# ============================================================
def plot_loss(history):
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], "o-", label="Train Loss")
    plt.plot(history["test_loss"], "s-", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy")
    plt.title("Loss vs Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 8) SINGLE PREDICTION (CGPA, EXPERIENCE)
# ============================================================
def predict_single(cgpa, experience, w, threshold=THRESHOLD):
    """
    Returns (probability_of_label_1, predicted_label)
    """
    x = np.array([1.0, float(cgpa), float(experience)], dtype=float)
    prob = sigmoid(x @ w)
    label = int(prob >= threshold)
    return float(prob), label


def _read_float(prompt, allow_quit=True):
    """
    Repeatedly prompt user for a float. If allow_quit=True, 'q' quits.
    """
    while True:
        raw = input(prompt).strip()
        if allow_quit and raw.lower() in {"q", "quit", "exit"}:
            return None
        try:
            val = float(raw)
            return val
        except ValueError:
            print("Please enter a valid number (or 'q' to quit).")


def interactive_prediction_loop(w, threshold=THRESHOLD):
    print("\n=== Interactive Predictions ===")
    print("Enter CGPA and Experience to get a prediction. Type 'q' at any prompt to quit.\n")

    while True:
        cgpa = _read_float("CGPA (e.g., 7.5 on 0–10 scale): ")
        if cgpa is None:
            break

        experience = _read_float("Work Experience in years (e.g., 2): ")
        if experience is None:
            break

        # Soft validation hints (not hard constraints)
        if not (0 <= cgpa <= 10):
            print("Note: CGPA seems outside a 0–10 scale; continuing anyway.")
        if experience < 0:
            print("Note: Negative experience doesn't make sense; treating as entered.")

        prob, label = predict_single(cgpa, experience, w, threshold=threshold)
        print(f"Predicted probability of outcome=1: {prob:.4f}")
        print(f"Predicted label (threshold={threshold:.2f}): {label}\n")


# ============================================================
# 9) MAIN
# ============================================================
if __name__ == "__main__":
    try:
        df = load_dataset(DATA_PATH, SHEET_NAME)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    w_final, history, df_train, df_test = train_model(df, EPOCHS, LR, w_init, l2=L2, threshold=THRESHOLD)
    
    # Cleaned up summary formatting for final outputs
    print("\n======================================================")
    print("✨ TRAINING COMPLETE ✨")
    print("======================================================")
    print(f"Final Weights Matrix: {w_final}")
    print(f"  -> Final Bias (w0)        : {w_final[0]:.4f}")
    print(f"  -> Final CGPA Weight (w1) : {w_final[1]:.4f}")
    print(f"  -> Final Exp Weight (w2)  : {w_final[2]:.4f}")
    print("======================================================\n")

    # Show loss curves
    plot_loss(history)

    # Interactive predictions
    interactive_prediction_loop(w_final, threshold=THRESHOLD)

    print("Done.")