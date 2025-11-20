package com.example;

import ai.onnxruntime.*;
import java.nio.file.Paths;
import java.util.*;

public class OnnxTest {

    public static void main(String[] args) throws Exception {

        // Load model from src/main/resources
        var url = OnnxTest.class.getClassLoader().getResource("titanic_random_forest.onnx");
        if (url == null) {
            throw new RuntimeException("Model file not found in resources");
        }

        String modelPath = Paths.get(url.toURI()).toString();

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
             OrtSession session = env.createSession(modelPath, opts)) {

            System.out.println("Model loaded from: " + modelPath);

            // ============================
            // Input tensors (shape [1,1])
            // ============================

            // Float inputs
            float[][] passengerId = new float[][] { { 892f } };
            float[][] pclass      = new float[][] { { 3f } };
            float[][] age         = new float[][] { { 34.5f } };
            float[][] sibsp       = new float[][] { { 0f } };
            float[][] parch       = new float[][] { { 0f } };
            float[][] fare        = new float[][] { { 7.8292f } };

            // String inputs
            String[][] name      = new String[][] { { "Kelly, Mr. James" } };
            String[][] sex       = new String[][] { { "male" } };
            String[][] ticket    = new String[][] { { "330911" } };
            String[][] cabin     = new String[][] { { "" } };
            String[][] embarked  = new String[][] { { "Q" } };

            // ============================
            // Build input map
            // ============================
            Map<String, OnnxTensor> inputs = new HashMap<>();

            inputs.put("PassengerId", OnnxTensor.createTensor(env, passengerId));
            inputs.put("Pclass",      OnnxTensor.createTensor(env, pclass));
            inputs.put("Age",         OnnxTensor.createTensor(env, age));
            inputs.put("SibSp",       OnnxTensor.createTensor(env, sibsp));
            inputs.put("Parch",       OnnxTensor.createTensor(env, parch));
            inputs.put("Fare",        OnnxTensor.createTensor(env, fare));

            inputs.put("Name",     OnnxTensor.createTensor(env, name));
            inputs.put("Sex",      OnnxTensor.createTensor(env, sex));
            inputs.put("Ticket",   OnnxTensor.createTensor(env, ticket));
            inputs.put("Cabin",    OnnxTensor.createTensor(env, cabin));
            inputs.put("Embarked", OnnxTensor.createTensor(env, embarked));

            // ============================
            // Run inference
            // ============================

            try (OrtSession.Result result = session.run(inputs)) {

                // Extract outputs
                long[] label = (long[]) result.get("label").get().getValue();
                float[][] probabilities = (float[][]) result.get("probabilities").get().getValue();

                // Print nicely
                System.out.println("===== MODEL OUTPUT =====");
                System.out.println("Predicted label: " + label[0]);
                System.out.println("Probabilities: " + Arrays.toString(probabilities[0]));
            }
        }
    }
}
