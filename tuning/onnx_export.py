from optimum.exporters.onnx import main_export

main_export("./output", output="output_onnx/", task="default", monolith=True)
