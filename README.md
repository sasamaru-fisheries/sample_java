## Python モデル生成
- `uv sync`
- `python penguins_onnx.py` で `model/penguin.onnx` を生成
- `python penguins_pmml.py` で `model/penguin.pmml` を生成

## Java 推論 (ONNX / PMML)
- `cd java_test`
- 初回はクラスをビルドするため `mvn compile` を実行
- ONNX 推論: `mvn -Dexec.mainClass=com.example.PenguinOnnx exec:java`
- PMML 推論: `mvn -Dexec.mainClass=com.example.PenguinPmml exec:java`

### 推論サンプルの入力（欠損を含んでも動作）
- ONNX: `bill_length_mm=NaN`, `island=""`（インプタで補完されて推論）
- PMML: `bill_length_mm=null`, `island=""`（インプタで補完されて推論）

## ONNX と PMML の違いメモ (Javaでの読み込み/推論/型指定)
- **ロード方法**
  - ONNX: `OrtEnvironment`/`OrtSession` を使って `createSession` にモデルパスを渡すだけ。
  - PMML: `PMMLUtil.unmarshal(InputStream)` でPMMLを読み込み、`ModelEvaluatorBuilder` で `Evaluator` を生成。
- **入力の作り方**
  - ONNX: 入力名ごとに `OnnxTensor` を作って `Map<String, OnnxTensor>` に詰める。数値は `float[][]`、カテゴリは `String[][]` を `createTensor` に渡す。
  - PMML: 生の `Map<String, Object>` を用意し、`InputField.prepare(...)` で型や前処理を適用したものを `Evaluator.evaluate(...)` に渡す。
- **出力の扱い**
  - ONNX: 出力名（例: `label`, `probabilities`）を決め打ちまたはフォールバックで取得し、値を配列にキャストして使う。
  - PMML: `evaluate` の結果から `ProbabilityDistribution` を取り出し、`getResult()` でラベル、`getProbability(<class>)` でクラスごとの確率を得る。
- **型指定のポイント**
  - ONNX: 事前にPython側で `initial_types` を指定（数値=FloatTensorType、カテゴリ=StringTensorType）。Java側ではテンソル生成時に `float`/`String` の2次元配列を渡す。
  - PMML: Python側でOneHotなどの前処理をパイプラインに含めておけば、Java側は生の値（数値/文字列）を渡し、`prepare` が適切に処理。
