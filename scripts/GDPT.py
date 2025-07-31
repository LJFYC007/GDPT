from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("PathTracer")
    PathTracer = createPass("MinimalPathTracer", {'maxBounces': 1})
    g.addPass(PathTracer, "PathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.markOutput("PathTracer.color")
    g.markOutput("AccumulatePass.output")

    AccumulatePassX = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single', 'maxFrameCount': 1024})
    g.addPass(AccumulatePassX, "AccumulatePassX")
    AccumulatePassY = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single', 'maxFrameCount': 1024})
    g.addPass(AccumulatePassY, "AccumulatePassY")
    g.addEdge("PathTracer.gradientX", "AccumulatePassX.input")
    g.addEdge("PathTracer.gradientY", "AccumulatePassY.input")

    ErrorMeasureXPass = createPass("ErrorMeasurePass", {'ReferenceImagePath': 'E:\\GDPT\\result\\gradientX-1024spp.exr', 'UseLoadedReference': True, 'SelectedOutputId': 'Difference'})
    g.addPass(ErrorMeasureXPass, "ErrorMeasureXPass")
    g.addEdge("AccumulatePassX.output", "ErrorMeasureXPass.Source")
    g.markOutput("ErrorMeasureXPass.Output")

    ErrorMeasureYPass = createPass("ErrorMeasurePass", {'ReferenceImagePath': 'E:\\GDPT\\result\\gradientY-1024spp.exr', 'UseLoadedReference': True, 'SelectedOutputId': 'Difference'})
    g.addPass(ErrorMeasureYPass, "ErrorMeasureYPass")
    g.addEdge("AccumulatePassY.output", "ErrorMeasureYPass.Source")
    g.markOutput("ErrorMeasureYPass.Output")

    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None

# m.clock.exitFrame = 1300
# m.frameCapture.outputDir = "../../../../output"
# m.frameCapture.addFrames(m.activeGraph, [16, 32, 64, 128, 1024])
