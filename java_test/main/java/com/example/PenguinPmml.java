package com.example;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;

import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.ModelEvaluatorBuilder;
import org.jpmml.evaluator.ProbabilityDistribution;
import org.jpmml.evaluator.TargetField;
import org.jpmml.model.PMMLUtil;

public class PenguinPmml {

    public static void main(String[] args) throws Exception {
        Path modelPath = Paths.get("..", "model", "penguin.pmml").toAbsolutePath(); // PMMLモデルのパス

        System.out.println("=== Penguin PMML inference (Adelie=0, Gentoo=1) ==="); // 見出し
        runModel(modelPath, Map.of(
                "bill_length_mm", 40.3, // 嘴長
                "bill_depth_mm", 18.0   // 嘴深
        ), "bill_length_mm/bill_depth_mm"); // モデル推論
    }

    private static void runModel(Path modelPath, Map<String, Double> rawFeatures, String label) throws Exception {
        try (var is = Files.newInputStream(modelPath)) { // PMMLを読み込み
            PMML pmml = PMMLUtil.unmarshal(is); // PMMLパース
            Evaluator evaluator = new ModelEvaluatorBuilder(pmml).build(); // Evaluator生成
            evaluator.verify(); // モデル検証

            Map<String, ?> results = evaluator.evaluate(prepareArguments(evaluator, rawFeatures)); // 推論実行

            TargetField target = evaluator.getTargetFields().get(0); // 予測ターゲット
            Object targetVal = results.get(target.getName()); // ターゲット値

            ProbabilityDistribution<?> dist = null; // 確率分布
            Object predicted;

            if (targetVal instanceof ProbabilityDistribution<?> pd) { // ターゲットが分布の場合
                dist = pd;
                predicted = pd.getResult();
            } else {
                predicted = targetVal; // ラベル値がそのまま返ってくる場合
            }

            if (dist == null) { // 分布が別キーにある場合を探索
                for (Object val : results.values()) {
                    if (val instanceof ProbabilityDistribution<?> pd) {
                        dist = pd;
                        break;
                    }
                }
            }

            double probAdelie = (dist != null && dist.getProbability(0) != null)
                    ? dist.getProbability(0)
                    : Double.NaN; // Adelie確率
            double probGentoo = (dist != null && dist.getProbability(1) != null)
                    ? dist.getProbability(1)
                    : Double.NaN; // Gentoo確率

            System.out.println("[" + label + "]"); // 実行ラベル
            System.out.println("  Model: " + modelPath); // モデルパス表示
            System.out.println("  Input: " + rawFeatures); // 入力表示
            System.out.println("  Predicted label: " + predicted); // 予測ラベル表示
            System.out.println("  Probabilities {Adelie=0, Gentoo=1}: ["
                    + probAdelie + ", " + probGentoo + "]"); // 確率表示
        }
    }

    private static Map<String, Object> prepareArguments(Evaluator evaluator, Map<String, Double> raw) {
        Map<String, Object> arguments = new LinkedHashMap<>(); // 入力マップ
        for (InputField inputField : evaluator.getInputFields()) { // 入力フィールドを走査
            String name = inputField.getName(); // フィールド名
            Object rawValue = raw.get(name); // 生データ取得
            arguments.put(name, inputField.prepare(rawValue)); // 型・前処理適用
        }
        return arguments; // 準備済み入力を返す
    }
}
