import os

import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

os.makedirs("model", exist_ok=True)

# Load penguins and build a binary target (Adelie vs Gentoo)
penguins = sns.load_dataset("penguins")
penguins = penguins[penguins["species"].isin(["Adelie", "Gentoo"])].dropna()

# Encode target as 0/1 for sklearn; keep labels for clarity
penguins["target"] = penguins["species"].map({"Adelie": 0, "Gentoo": 1})
X = penguins[
    ["bill_length_mm", "bill_depth_mm"]
]
y = penguins["target"]


def build_pmml(features: list[str], output_path: str) -> None:
    """Train a scaler + logistic regression on selected features and export to PMML."""
    preprocessor = ColumnTransformer(
        [("scale", StandardScaler(), features)],
        remainder="drop",
    )

    pipeline = PMMLPipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=200, random_state=42)),
    ])

    pipeline.fit(X[features], y)
    sklearn2pmml(pipeline, output_path, with_repr=True)


# Single model: bill dimensions
build_pmml(["bill_length_mm", "bill_depth_mm"], "model/penguin.pmml")

print("Saved model/penguin.pmml")
