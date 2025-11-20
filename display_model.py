import onnx

model = onnx.load("src/main/resources/titanic_random_forest.onnx")
print(model.graph.input)
