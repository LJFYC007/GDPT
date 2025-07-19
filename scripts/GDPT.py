from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("PathTracer")
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1})
    g.addPass(PathTracer, "PathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.markOutput("AccumulatePass.output")

    PathTracer1 = createPass("PathTracer", {'samplesPerPixel': 1, 'xDiff': 1, 'yDiff': 0})
    g.addPass(PathTracer1, "PathTracer1")
    g.addEdge("VBufferRT.vbuffer", "PathTracer1.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer1.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer1.mvec")

    PathTracer2 = createPass("PathTracer", {'samplesPerPixel': 1, 'xDiff': 0, 'yDiff': 1})
    g.addPass(PathTracer2, "PathTracer2")
    g.addEdge("VBufferRT.vbuffer", "PathTracer2.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer2.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer2.mvec")

    PathTracer3 = createPass("PathTracer", {'samplesPerPixel': 1, 'xDiff': -1, 'yDiff': 0})
    g.addPass(PathTracer3, "PathTracer3")
    g.addEdge("VBufferRT.vbuffer", "PathTracer3.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer3.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer3.mvec")

    PathTracer4 = createPass("PathTracer", {'samplesPerPixel': 1, 'xDiff': 0, 'yDiff': -1})
    g.addPass(PathTracer4, "PathTracer4")
    g.addEdge("VBufferRT.vbuffer", "PathTracer4.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer4.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer4.mvec")

    SimpleGradient = createPass("SimpleGradient", {})
    g.addPass(SimpleGradient, "SimpleGradient")
    g.addEdge("PathTracer.color", "SimpleGradient.base")
    g.addEdge("PathTracer1.color", "SimpleGradient.input1")
    g.addEdge("PathTracer2.color", "SimpleGradient.input2")
    g.addEdge("PathTracer3.color", "SimpleGradient.input3")
    g.addEdge("PathTracer4.color", "SimpleGradient.input4")

    AccumulatePassX = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePassX, "AccumulatePassX")
    AccumulatePassY = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePassY, "AccumulatePassY")
    g.addEdge("SimpleGradient.outputX", "AccumulatePassX.input")
    g.addEdge("SimpleGradient.outputY", "AccumulatePassY.input")

    g.markOutput("AccumulatePassX.output")
    g.markOutput("AccumulatePassY.output")

    ReconstructionPass = createPass("ReconstructionPass", {})
    g.addPass(ReconstructionPass, "ReconstructionPass")
    g.addEdge("AccumulatePass.output", "ReconstructionPass.base")
    g.addEdge("AccumulatePassX.output", "ReconstructionPass.inputX")
    g.addEdge("AccumulatePassY.output", "ReconstructionPass.inputY")
    g.markOutput("ReconstructionPass.output")

    ErrorMeasurePass = createPass("ErrorMeasurePass", {'ReferenceImagePath': 'E:\\GDPT\\output\\reference.exr', 'UseLoadedReference': True, 'SelectedOutputId': 'Difference'})
    g.addPass(ErrorMeasurePass, "ErrorMeasurePass")
    g.addEdge("ReconstructionPass.output", "ErrorMeasurePass.Source")
    g.markOutput("ErrorMeasurePass.Output")

    PTErrorMeasurePass = createPass("ErrorMeasurePass", {'ReferenceImagePath': 'E:\\GDPT\\output\\reference.exr', 'UseLoadedReference': True, 'SelectedOutputId': 'Difference'})
    g.addPass(PTErrorMeasurePass, "PTErrorMeasurePass")
    g.addEdge("AccumulatePass.output", "PTErrorMeasurePass.Source")
    g.markOutput("PTErrorMeasurePass.Output")
    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None

m.clock.exitFrame = 130
m.frameCapture.outputDir = "../../../../output"

m.frameCapture.addFrames(m.activeGraph, [64, 128])
