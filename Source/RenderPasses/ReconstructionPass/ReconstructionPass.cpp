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
#include "ReconstructionPass.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReconstructionPass>();
}

namespace
{
const std::string kBaseChannelEventImage = "base";
const std::string kInputXChannelEventImage = "inputX";
const std::string kInputYChannelEventImage = "inputY";
const std::string kOutputChannelEventImage = "output";
const std::string kTempChannelEventImage = "temp";
const std::string kNum = "num";
}

ReconstructionPass::ReconstructionPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (auto [key, value] : props)
    {
        if (key == kNum) num = value;
    }
}

Properties ReconstructionPass::getProperties() const
{
    Properties props;
    props[kNum] = num;
    return props;
}

RenderPassReflection ReconstructionPass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addInput(kBaseChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInputXChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInputYChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput(kTempChannelEventImage, "")
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputChannelEventImage, "")
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Float);
    return reflector;
}

void ReconstructionPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mpScene == nullptr)
        return;

    if (!mpComputePass)
    {
        DefineList defines;
        mpScene->getShaderDefines(defines);
        ProgramDesc desc;
        mpScene->getShaderModules(desc.shaderModules);
        desc.addShaderLibrary("RenderPasses/ReconstructionPass/ReconstructionPass.slang");
        desc.csEntry("main");
        mpScene->getTypeConformances(desc.typeConformances);
        mpComputePass = ComputePass::create(mpDevice, desc, defines);
    }

    if (!mpClearPass)
    {
        DefineList defines;
        mpScene->getShaderDefines(defines);
        ProgramDesc desc;
        mpScene->getShaderModules(desc.shaderModules);
        desc.addShaderLibrary("RenderPasses/ReconstructionPass/Clear.slang");
        desc.csEntry("main");
        mpScene->getTypeConformances(desc.typeConformances);
        mpClearPass = ComputePass::create(mpDevice, desc, defines);
    }

    if (!mpCopyPass)
    {
        DefineList defines;
        mpScene->getShaderDefines(defines);
        ProgramDesc desc;
        mpScene->getShaderModules(desc.shaderModules);
        desc.addShaderLibrary("RenderPasses/ReconstructionPass/Copy.slang");
        desc.csEntry("main");
        mpScene->getTypeConformances(desc.typeConformances);
        mpCopyPass = ComputePass::create(mpDevice, desc, defines);
    }

    ref<Texture> baseTexture = renderData.getTexture(kBaseChannelEventImage);
    const uint2 resolution = uint2(baseTexture->getWidth(), baseTexture->getHeight());

    auto vars = mpCopyPass->getRootVar();
    vars["input"] = baseTexture;
    vars["output"] = renderData.getTexture(kTempChannelEventImage);
    vars["PerFrameCB"]["gResolution"] = resolution;
    mpCopyPass->execute(pRenderContext, uint3(resolution, 1));

    vars = mpComputePass->getRootVar();
    vars["base"] = renderData.getTexture(kTempChannelEventImage);
    vars["inputX"] = renderData.getTexture(kInputXChannelEventImage);
    vars["inputY"] = renderData.getTexture(kInputYChannelEventImage);
    vars["output"] = renderData.getTexture(kOutputChannelEventImage);
    vars["PerFrameCB"]["gResolution"] = resolution;

    vars = mpClearPass->getRootVar();
    vars["base"] = renderData.getTexture(kTempChannelEventImage);
    vars["output"] = renderData.getTexture(kOutputChannelEventImage);
    vars["PerFrameCB"]["gResolution"] = resolution;

    for ( int i = 0; i < num; ++ i)
    {
        mpComputePass->execute(pRenderContext, uint3(resolution, 1));
        mpClearPass->execute(pRenderContext, uint3(resolution, 1));
    }
}

void ReconstructionPass::renderUI(Gui::Widgets& widget)
{
    widget.var("Number of iterations", num, 1, 100);
}
