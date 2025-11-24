## Python モデル生成
- `uv sync`
- `python penguins_onnx.py` で `model/penguin.onnx` を生成
- `python penguins_pmml.py` で `model/penguin.pmml` を生成

## Java 推論 (ONNX / PMML)
- `cd java_test`
- 初回はクラスをビルドするため `mvn compile` を実行
- ONNX 推論: `mvn -Dexec.mainClass=com.example.PenguinOnnx exec:java`
- PMML 推論: `mvn -Dexec.mainClass=com.example.PenguinPmml exec:java`
