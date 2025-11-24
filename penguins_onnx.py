import os

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

os.makedirs("model", exist_ok=True)

# Load penguins and build a binary target (Adelie vs Gentoo)
penguins = sns.load_dataset("penguins")
penguins = penguins[penguins["species"].isin(["Adelie", "Gentoo"])].dropna()
penguins["target"] = penguins["species"].map({"Adelie": 0, "Gentoo": 1})

# Single model: bill dimensions + scaler
X = penguins[["bill_length_mm", "bill_depth_mm"]].values
y = penguins["target"].values

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200, random_state=42))
])

pipeline.fit(X, y)

initial_type = [('input', FloatTensorType([None, 2]))]

onx = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    target_opset=15
)

with open("model/penguin.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("Saved model/penguin.onnx")
