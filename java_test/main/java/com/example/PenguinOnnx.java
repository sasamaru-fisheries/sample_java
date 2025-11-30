package com.example; // パッケージ宣言

import ai.onnxruntime.OnnxTensor; // ORTのテンソル表現クラス
import ai.onnxruntime.OrtEnvironment; // ORT環境
import ai.onnxruntime.OrtSession; // ORTセッション
import java.nio.file.Paths; // パス生成ユーティリティ
import java.util.Arrays; // 配列表示
import java.util.Map; // マップ

public class PenguinOnnx { // ONNX推論の最小サンプル

    public static void main(String[] args) throws Exception { // 実行エントリポイント
        String modelPath = Paths.get("..", "model", "penguin.onnx").toString(); // モデルパス（固定）

        // ORT環境とセッションを作成し、ONNXモデルを読み込む
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession session = env.createSession(modelPath, new OrtSession.SessionOptions())) {

            // 入力データ（数値1特徴＋カテゴリ1特徴）をテンソルに変換
            Map<String, OnnxTensor> inputs = Map.of(
                    "bill_length_mm", OnnxTensor.createTensor(env, new float[][]{{Float.NaN}}), // 数値は欠損ならNaN
                    "island", OnnxTensor.createTensor(env, new String[][]{{""}}) // カテゴリ欠損は空文字
            );

            // 推論を実行し、決め打ちの出力名で結果を取得
            try (OrtSession.Result r = session.run(inputs)) {
                // 出力名が異なる場合に備えて最初の出力をフォールバックで使う
                String labelName = r.get("label").isPresent()
                        ? "label"
                        : session.getOutputNames().iterator().next();
                String probName = r.get("probabilities").isPresent()
                        ? "probabilities"
                        : null;

                long label = ((long[]) r.get(labelName).get().getValue())[0]; // ラベル（long配列）
                float[] prob = probName == null
                        ? new float[] {}
                        : ((float[][]) r.get(probName).get().getValue())[0]; // 確率（float配列）

                // 結果を表示
                System.out.println("Model : " + modelPath);
                System.out.println("Input : bill_length_mm=40.3, island=Torgersen");
                System.out.println("Label : " + label);
                System.out.println("Prob  : " + Arrays.toString(prob)); // [Adelie, Gentoo]
            }
        }
    }
}
