import os

import seaborn as sns  # seabornからペンギンデータセットを取得
from sklearn.compose import ColumnTransformer  # 列ごとの前処理指定に使用
from sklearn.impute import SimpleImputer  # 欠損値補完
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰モデル
from sklearn.pipeline import Pipeline  # 前処理をまとめるパイプライン
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 標準化とワンホット
from sklearn2pmml import sklearn2pmml  # sklearnモデルをPMML形式に変換
from sklearn2pmml.pipeline import PMMLPipeline  # PMML対応のパイプライン

os.makedirs("model", exist_ok=True)  # モデル保存ディレクトリを作成

# ペンギンデータを読み込み、AdelieとGentooに絞る
penguins = sns.load_dataset("penguins")
penguins = penguins[penguins["species"].isin(["Adelie", "Gentoo"])]
penguins["island"] = penguins["island"].fillna("")  # カテゴリ欠損を空文字に

# 種類を0/1ラベルにエンコードし、特徴量と目的変数を用意
penguins["target"] = penguins["species"].map({"Adelie": 0, "Gentoo": 1})
X = penguins[
    ["bill_length_mm", "island"]
]
y = penguins["target"]

# 単一モデルで使用する特徴（数値1+カテゴリ1）
num_features = ["bill_length_mm"]  # 数値
cat_features = ["island"]          # カテゴリ

# 数値は標準化、カテゴリはワンホット化する前処理
preprocessor = ColumnTransformer(
    [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),  # 欠損値を中央値で補完
            ("scaler", StandardScaler()),                  # 標準化
        ]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", missing_values="")),  # 欠損値を最頻値で補完（空文字扱い）
            ("onehot", OneHotEncoder(handle_unknown="ignore")),    # ワンホット
        ]), cat_features),
    ],
    remainder="drop",
)

# 前処理＋ロジスティック回帰をPMML対応パイプラインとして構築
pipeline = PMMLPipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(max_iter=200, random_state=42)),
])

pipeline.fit(X[num_features + cat_features], y)  # モデル学習
sklearn2pmml(pipeline, "model/penguin.pmml", with_repr=True)  # PMMLとして保存

print("Saved model/penguin.pmml")  # 保存完了メッセージ
