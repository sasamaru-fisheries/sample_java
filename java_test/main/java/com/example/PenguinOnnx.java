package com.example;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class PenguinOnnx {

    public static void main(String[] args) throws Exception {
        Path modelA = Paths.get("..", "model", "modelA.onnx").toAbsolutePath(); // 推論用ONNXモデルAのパス
        Path modelB = Paths.get("..", "model", "modelB.onnx").toAbsolutePath(); // 推論用ONNXモデルBのパス

        System.out.println("=== Penguin ONNX inference (Adelie=0, Gentoo=1) ==="); // 見出し
        runModel(modelA, new float[] { 40.3f, 18.0f }, "ModelA bill_length_mm/bill_depth_mm"); // モデルA推論
        runModel(modelB, new float[] { 210f, 4500f }, "ModelB flipper_length_mm/body_mass_g"); // モデルB推論
    }

    private static void runModel(Path modelPath, float[] features, String label) throws Exception {
        try (OrtEnvironment env = OrtEnvironment.getEnvironment(); // ORT環境
             OrtSession.SessionOptions opts = new OrtSession.SessionOptions(); // セッション設定
             OrtSession session = env.createSession(modelPath.toString(), opts)) { // モデル読み込み

            // skl2onnx exported input name is "input" with shape [None, 2]
            float[][] input = new float[][] { features }; // 入力配列(1x2)
            Map<String, OnnxTensor> inputs = new HashMap<>(); // 入力マップ
            inputs.put("input", OnnxTensor.createTensor(env, input)); // テンソル化

            try (OrtSession.Result result = session.run(inputs)) { // 推論実行
                String labelOutput = session.getOutputNames().stream() // ラベル出力名を推定
                        .filter(n -> n.toLowerCase().contains("label"))
                        .findFirst()
                        .orElse(session.getOutputNames().iterator().next());

                String probOutput = session.getOutputNames().stream() // 確率出力名を推定
                        .filter(n -> n.toLowerCase().contains("prob"))
                        .findFirst()
                        .orElse(null);

                long[] labelArray = (long[]) result.get(labelOutput).get().getValue(); // ラベル取得
                float[] probArray = probOutput == null // 確率取得（無ければ空）
                        ? new float[] {}
                        : ((float[][]) result.get(probOutput).get().getValue())[0];

                System.out.println("[" + label + "]"); // 実行ラベル
                System.out.println("  Model: " + modelPath); // モデルパス表示
                System.out.println("  Input: " + Arrays.toString(features)); // 入力表示
                System.out.println("  Predicted label: " + labelArray[0]); // 予測ラベル表示
                if (probOutput != null) {
                    System.out.println("  Probabilities [Adelie, Gentoo]: " + Arrays.toString(probArray)); // 確率表示
                }
            }
        }
    }
}
