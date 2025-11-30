import os

import seaborn as sns  # seabornからペンギンデータを取得するために使用
from sklearn.compose import ColumnTransformer  # 数値/カテゴリの前処理を分岐
from sklearn.impute import SimpleImputer  # 欠損値補完
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰モデル
from sklearn.pipeline import Pipeline  # 前処理とモデルをまとめるパイプライン
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 標準化とワンホット
from skl2onnx import convert_sklearn  # sklearnモデルをONNX形式に変換
from skl2onnx.common.data_types import FloatTensorType, StringTensorType  # ONNX入力型の指定に利用

os.makedirs("model", exist_ok=True)  # モデル保存用ディレクトリを作成

# ペンギンデータを読み込み、AdelieとGentooの2値分類データに絞る
penguins = sns.load_dataset("penguins")
penguins = penguins[penguins["species"].isin(["Adelie", "Gentoo"])]
penguins["island"] = penguins["island"].fillna("")  # カテゴリ欠損は空文字に
penguins["target"] = penguins["species"].map({"Adelie": 0, "Gentoo": 1})  # ラベルを0/1に変換

# 数値1特徴（bill_length_mm）とカテゴリ1特徴（island）のみを使用
X = penguins[["bill_length_mm", "island"]]
y = penguins["target"].values

# 数値は標準化、カテゴリはワンホット化する前処理
preprocess = ColumnTransformer(
    [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),  # 欠損値を中央値で補完
            ("scaler", StandardScaler()),                  # 標準化
        ]), ["bill_length_mm"]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", missing_values="")),  # 欠損値を最頻値で補完（空文字を欠損扱い）
            ("onehot", OneHotEncoder(handle_unknown="ignore")),    # ワンホット
        ]), ["island"]),
    ]
)

# 前処理 + ロジスティック回帰のパイプラインを構築
pipeline = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=200, random_state=42))
])

pipeline.fit(X, y)  # モデルを学習

initial_type = [
    ("bill_length_mm", FloatTensorType([None, 1])),  # 数値特徴
    ("island", StringTensorType([None, 1])),         # カテゴリ特徴
]  # ONNX変換用の入力型を定義

# sklearnモデルをONNX形式に変換
onx = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    target_opset=15
)

# 変換結果をファイルに書き出し
with open("model/penguin.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("Saved model/penguin.onnx")  # 保存完了メッセージ
