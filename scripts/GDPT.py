from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("PathTracer")
    PathTracer = createPass("MinimalPathTracer", {'maxBounces': 5})
    g.addPass(PathTracer, "PathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.markOutput("AccumulatePass.output")
    g.markOutput("PathTracer.color")
    # g.markOutput("PathTracer.gradientX")
    # g.markOutput("PathTracer.gradientY")

    # ErrorMeasurePass = createPass("ErrorMeasurePass", {'ReferenceImagePath': 'E:\\GDPT\\minimal_result\\reference-staircase.exr', 'UseLoadedReference': True, 'SelectedOutputId': 'Source'})
    # g.addPass(ErrorMeasurePass, "ErrorMeasurePass")
    # g.addEdge("AccumulatePass.output", "ErrorMeasurePass.Source")
    # g.markOutput("ErrorMeasurePass.Output")

    # SimpleGradient = createPass("SimpleGradient", {})
    # g.addPass(SimpleGradient, "SimpleGradient")
    # g.addEdge("AccumulatePass.output", "SimpleGradient.base")
    # g.addEdge("AccumulatePass.output", "SimpleGradient.input1")
    # g.addEdge("AccumulatePass.output", "SimpleGradient.input2")
    # g.addEdge("AccumulatePass.output", "SimpleGradient.input3")
    # g.addEdge("AccumulatePass.output", "SimpleGradient.input4")
    # g.markOutput("SimpleGradient.outputX")
    # g.markOutput("SimpleGradient.outputY")

    AccumulatePassX = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePassX, "AccumulatePassX")
    AccumulatePassY = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePassY, "AccumulatePassY")
    g.addEdge("PathTracer.gradientX", "AccumulatePassX.input")
    g.addEdge("PathTracer.gradientY", "AccumulatePassY.input")
    # g.markOutput("AccumulatePassX.output")
    # g.markOutput("AccumulatePassY.output")

    ErrorMeasureXPass = createPass("ErrorMeasurePass", {'ReferenceImagePath': 'E:\\GDPT\\result\\staircase\\reference-gradientX.exr', 'UseLoadedReference': True, 'SelectedOutputId': 'Source'})
    g.addPass(ErrorMeasureXPass, "ErrorMeasureXPass")
    g.addEdge("AccumulatePassX.output", "ErrorMeasureXPass.Source")
    g.markOutput("ErrorMeasureXPass.Output")

    ErrorMeasureYPass = createPass("ErrorMeasurePass", {'ReferenceImagePath': 'E:\\GDPT\\result\\staircase\\reference-gradientY.exr', 'UseLoadedReference': True, 'SelectedOutputId': 'Source'})
    g.addPass(ErrorMeasureYPass, "ErrorMeasureYPass")
    g.addEdge("AccumulatePassY.output", "ErrorMeasureYPass.Source")
    g.markOutput("ErrorMeasureYPass.Output")

    PostProcess = createPass("PostProcess", {'sigma': 3.0, 'kernelWidth': 11})
    g.addPass(PostProcess, "PostProcess")
    g.addEdge("AccumulatePass.variance", "PostProcess.Input")
    g.markOutput("PostProcess.Output")

    PostProcessX = createPass("PostProcess", {'sigma': 3.0, 'kernelWidth': 11})
    g.addPass(PostProcessX, "PostProcessX")
    g.addEdge("AccumulatePassX.variance", "PostProcessX.Input")
    g.markOutput("PostProcessX.Output")

    PostProcessY = createPass("PostProcess", {'sigma': 3.0, 'kernelWidth': 11})
    g.addPass(PostProcessY, "PostProcessY")
    g.addEdge("AccumulatePassY.variance", "PostProcessY.Input")
    g.markOutput("PostProcessY.Output")

    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None

m.clock.exitFrame = 1100
m.frameCapture.outputDir = "../../../../output"
m.frameCapture.addFrames(m.activeGraph, [4, 16, 32, 64, 128, 1024])
