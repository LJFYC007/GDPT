/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "PostProcess.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, PostProcess>();
}

namespace
{
    const std::string kInputChannel = "Input";
    const std::string kOutputChannel = "Output";
}

PostProcess::PostProcess(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == "sigma") mSigma = value;
        else if (key == "kernelWidth") mKernelWidth = value;
        else throw std::exception("Unknown property");
    }
}

Properties PostProcess::getProperties() const
{
    Properties props;
    props["sigma"] = mSigma;
    props["kernelWidth"] = mKernelWidth;
    return props;
}

RenderPassReflection PostProcess::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addInput(kInputChannel, "").bindFlags(ResourceBindFlags::ShaderResource);

    reflector.addOutput(kOutputChannel, "").bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource).format(ResourceFormat::RGBA32Float);
    return reflector;
}

void PostProcess::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    ref<Texture> inputTexture = renderData.getTexture(kInputChannel);
    ref<Texture> outputTexture = renderData.getTexture(kOutputChannel);
    if (!inputTexture || !outputTexture) return;

    if (!mpGaussianBlurPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary("RenderPasses/PostProcess/PostProcess.slang");
        desc.csEntry("main");
        mpGaussianBlurPass = ComputePass::create(mpDevice, desc, {});
    }
    pRenderContext->clearUAV(outputTexture->getUAV().get(), float4(0.f));

    auto vars = mpGaussianBlurPass->getRootVar();
    vars["Input"] = inputTexture;
    vars["Output"] = outputTexture;
    vars["PerFrameCB"]["gSigma"] = mSigma;
    vars["PerFrameCB"]["gKernelWidth"] = mKernelWidth;
    uint2 resolution = uint2(inputTexture->getWidth(), inputTexture->getHeight());
    vars["PerFrameCB"]["gResolution"] = resolution;
    mpGaussianBlurPass->execute(pRenderContext, uint3(resolution, 1));
}

void PostProcess::renderUI(Gui::Widgets& widget)
{
    widget.slider("Sigma", mSigma, 0.1f, 10.0f);
    widget.slider("Kernel Width", mKernelWidth, 3, 25);
}
