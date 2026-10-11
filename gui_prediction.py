import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib
import numpy as np

# ============================================
# Load Model and Preprocessor
# ============================================
try:
    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("best_model.pkl")
except:
    messagebox.showerror("Error", "Error loading model or preprocessor files!")
    raise SystemExit


# Prediction function
def predict_price(df):
    transformed = preprocessor.transform(df)
    preds = model.predict(transformed)
    return preds


# ============================================
# GUI Functions
# ============================================
def select_file():
    path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    if path:
        file_path_var.set(path)


def run_prediction():
    path = file_path_var.get()

    if path.strip() == "":
        messagebox.showwarning("Warning", "Please select a CSV file first!")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        messagebox.showerror("Error", f"Unable to read file:\n{e}")
        return

    # Remove price column if exists
    if "price" in df.columns:
        df = df.drop(columns=["price"])

    try:
        preds = predict_price(df)
    except Exception as e:
        messagebox.showerror("Error", f"Prediction error:\n{e}")
        return

    df_result = df.copy()
    df_result["predicted_price"] = preds
    df_result["predicted_price_formatted"] = df_result["predicted_price"].apply(lambda x: f"{x:,.2f} USD")

    output_path = "predictions_output.csv"
    df_result.to_csv(output_path, index=False)

    # Show first 10 results in UI
    preview_box.delete(*preview_box.get_children())
    for i in range(min(10, len(df_result))):
        row = df_result.iloc[i]
        preview_box.insert("", "end", values=(
            row["predicted_price_formatted"],
        ))

    messagebox.showinfo("Success", f"Predictions saved!\nFile: {output_path}")


# ============================================
# GUI Layout
# ============================================
app = tk.Tk()
app.title("House Price Prediction")
app.geometry("500x500")
app.resizable(False, False)

title_label = tk.Label(app, text="House Price Prediction App", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

file_frame = tk.Frame(app)
file_frame.pack(pady=10)

file_path_var = tk.StringVar()

path_entry = tk.Entry(file_frame, width=40, textvariable=file_path_var, font=("Arial", 10))
path_entry.pack(side=tk.LEFT, padx=5)

browse_button = tk.Button(file_frame, text="Browse", command=select_file)
browse_button.pack(side=tk.LEFT)

predict_button = tk.Button(app, text="Run Prediction", command=run_prediction, width=20, bg="#4CAF50", fg="white")
predict_button.pack(pady=10)

result_label = tk.Label(app, text="Preview (First 10 Predictions):", font=("Arial", 12))
result_label.pack(pady=5)

columns = ("price",)
preview_box = ttk.Treeview(app, columns=columns, show="headings", height=10)
preview_box.heading("price", text="Predicted Price")
preview_box.pack()

close_button = tk.Button(app, text="Close", command=app.destroy, width=15, bg="#d9534f", fg="white")
close_button.pack(pady=15)

app.mainloop()
