package com.example;

import ai.onnxruntime.OnnxMap;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PenguinOnnx {

    public static void main(String[] args) throws Exception {
        Path modelPath = Paths.get("..", "model", "penguin.onnx").toAbsolutePath(); // 推論用ONNXモデルのパス

        System.out.println("=== Penguin ONNX inference (Adelie=0, Gentoo=1) ==="); // 見出し
        runModel(modelPath, new float[] { 40.3f, 18.0f }, "bill_length_mm/bill_depth_mm"); // モデル推論
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

                Object rawLabel = result.get(labelOutput).get().getValue(); // ラベル取得(型はモデル依存)
                long predicted = extractLabel(rawLabel); // 予測ラベル

                float[] probArray = probOutput == null // 確率取得（無ければ空）
                        ? new float[] {}
                        : extractProbabilities(result.get(probOutput).get().getValue());

                System.out.println("[" + label + "]"); // 実行ラベル
                System.out.println("  Model: " + modelPath); // モデルパス表示
                System.out.println("  Input: " + Arrays.toString(features)); // 入力表示
                System.out.println("  Predicted label: " + predicted); // 予測ラベル表示
                if (probOutput != null) {
                    System.out.println("  Probabilities [Adelie, Gentoo]: " + Arrays.toString(probArray)); // 確率表示
                }
            }
        }
    }

    private static long extractLabel(Object raw) {
        if (raw instanceof long[] arr && arr.length > 0) return arr[0];
        if (raw instanceof int[] arr && arr.length > 0) return arr[0];
        if (raw instanceof List<?> list && !list.isEmpty() && list.get(0) instanceof Number num) {
            return num.longValue();
        }
        throw new IllegalArgumentException("Unsupported label type: " + raw);
    }

    private static float[] extractProbabilities(Object raw) throws ai.onnxruntime.OrtException {
        if (raw instanceof OnnxMap onnxMap) { // ORT OnnxMap wrapper
            Object val = onnxMap.getValue(); // may throw OrtException
            return extractProbabilities(val);
        }
        if (raw instanceof float[][] arr && arr.length > 0) return arr[0];
        if (raw instanceof double[][] arr && arr.length > 0) {
            double[] src = arr[0];
            float[] out = new float[src.length];
            for (int i = 0; i < src.length; i++) out[i] = (float) src[i];
            return out;
        }
        if (raw instanceof Map<?, ?> map) { // ONNXMap from skl2onnx (class -> prob)
            return mapToProbs(map);
        }
        if (raw instanceof List<?> list && !list.isEmpty()) {
            Object first = list.get(0);
            if (first instanceof OnnxMap onnxMap) { // ListにOnnxMapが入っている場合
                try {
                    return extractProbabilities(onnxMap.getValue());
                } catch (ai.onnxruntime.OrtException e) {
                    throw new IllegalArgumentException("Failed to read OnnxMap probabilities", e);
                }
            }
            if (first instanceof Map<?, ?> map) { // map keyed by class id as string/int
                return mapToProbs(map);
            }
            if (first instanceof float[] fa) return fa;
            if (first instanceof double[] da) {
                float[] out = new float[da.length];
                for (int i = 0; i < da.length; i++) out[i] = (float) da[i];
                return out;
            }
        }
        throw new IllegalArgumentException("Unsupported probability type: " + raw);
    }

    private static float[] mapToProbs(Map<?, ?> map) {
            float[] out = new float[2];
            out[0] = getProb(map, 0);
            out[1] = getProb(map, 1);
            return out;
    }

    private static float getProb(Map<?, ?> map, Object key) {
        Object v = map.get(key) != null ? map.get(key) : map.get(String.valueOf(key));
        if (v instanceof Number num) {
            return num.floatValue();
        }
        return Float.NaN;
    }
}
