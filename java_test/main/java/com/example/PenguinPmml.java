package com.example; // パッケージ宣言

import java.nio.file.Files; // ファイル読み込み
import java.nio.file.Path; // パス表現
import java.nio.file.Paths; // パス生成ユーティリティ
import java.util.LinkedHashMap; // nullを許容するマップ
import java.util.Map; // 入力マップ

import org.dmg.pmml.PMML; // PMMLモデル表現
import org.jpmml.evaluator.Evaluator; // 評価器インターフェース
import org.jpmml.evaluator.InputField; // 入力フィールド情報
import org.jpmml.evaluator.ModelEvaluatorBuilder; // 評価器ビルダー
import org.jpmml.evaluator.ProbabilityDistribution; // 確率分布
import org.jpmml.evaluator.TargetField; // 予測ターゲット情報
import org.jpmml.model.PMMLUtil; // PMMLユーティリティ

public class PenguinPmml { // PMML推論の最小サンプル

    public static void main(String[] args) throws Exception { // 実行エントリポイント
        Path modelPath = Paths.get("..", "model", "penguin.pmml"); // モデルパス（固定）

        // PMMLモデルを読み込み、評価器を作成
        PMML pmml;
        try (var is = Files.newInputStream(modelPath)) {
            pmml = PMMLUtil.unmarshal(is); // ストリームからPMMLを読み込み
        }
        Evaluator evaluator = new ModelEvaluatorBuilder(pmml).build();
        evaluator.verify(); // モデル検証

        // 入力データをそのままマップで用意
        Map<String, Object> rawFeatures = new LinkedHashMap<>(); // null許容の入力データ
        rawFeatures.put("bill_length_mm", null); // 数値欠損はnullを渡し、インプタで補完
        rawFeatures.put("island", "");          // カテゴリ欠損は空文字

        // 評価器が期待する形式に変換（型・前処理を適用）
        Map<String, Object> arguments = new java.util.LinkedHashMap<>();
        for (InputField inputField : evaluator.getInputFields()) {
            String name = inputField.getName();
            arguments.put(name, inputField.prepare(rawFeatures.get(name)));
        }

        // 推論を実行
        Map<String, ?> results = evaluator.evaluate(arguments);
        TargetField target = evaluator.getTargetFields().get(0);
        ProbabilityDistribution<?> dist = (ProbabilityDistribution<?>) results.get(target.getName());

        // ラベルと確率を取得して表示
        Object predicted = dist.getResult();
        double probAdelie = dist.getProbability(0);
        double probGentoo = dist.getProbability(1);

        System.out.println("Model : " + modelPath.toAbsolutePath());
        System.out.println("Input : " + rawFeatures);
        System.out.println("Label : " + predicted);
        System.out.println("Prob  : [Adelie=" + probAdelie + ", Gentoo=" + probGentoo + "]");
    }
}
