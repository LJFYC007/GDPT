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
#include "SimpleGradient.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, SimpleGradient>();
}

namespace
{
const std::string kBaseChannelEventImage = "base";
const std::string kInput1ChannelEventImage = "input1";
const std::string kInput2ChannelEventImage = "input2";
const std::string kInput3ChannelEventImage = "input3";
const std::string kInput4ChannelEventImage = "input4";
const std::string kOutputXChannelEventImage = "outputX";
const std::string kOutputYChannelEventImage = "outputY";
} // namespace

SimpleGradient::SimpleGradient(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {}

Properties SimpleGradient::getProperties() const
{
    return {};
}

RenderPassReflection SimpleGradient::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    reflector.addInput(kBaseChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInput1ChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInput2ChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInput3ChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInput4ChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);

    reflector.addOutput(kOutputXChannelEventImage, "")
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kOutputYChannelEventImage, "")
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Float);

    return reflector;
}

void SimpleGradient::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mpScene == nullptr)
        return;

    if (!mpComputePass)
    {
        DefineList defines;
        mpScene->getShaderDefines(defines);
        ProgramDesc desc;
        mpScene->getShaderModules(desc.shaderModules);
        desc.addShaderLibrary("RenderPasses/SimpleGradient/SimpleGradient.slang");
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
        desc.addShaderLibrary("RenderPasses/SimpleGradient/Clear.slang");
        desc.csEntry("main");
        mpScene->getTypeConformances(desc.typeConformances);
        mpClearPass = ComputePass::create(mpDevice, desc, defines);
    }

    ref<Texture> baseTexture = renderData.getTexture(kBaseChannelEventImage);
    const uint2 resolution = uint2(baseTexture->getWidth(), baseTexture->getHeight());

    auto vars = mpClearPass->getRootVar();
    vars["outputX"] = renderData.getTexture(kOutputXChannelEventImage);
    vars["outputY"] = renderData.getTexture(kOutputYChannelEventImage);
    vars["PerFrameCB"]["gResolution"] = resolution;
    mpClearPass->execute(pRenderContext, uint3(resolution, 1));

    vars = mpComputePass->getRootVar();
    vars["base"] = baseTexture;
    vars["input1"] = renderData.getTexture(kInput1ChannelEventImage);
    vars["input2"] = renderData.getTexture(kInput2ChannelEventImage);
    vars["input3"] = renderData.getTexture(kInput3ChannelEventImage);
    vars["input4"] = renderData.getTexture(kInput4ChannelEventImage);
    vars["outputX"] = renderData.getTexture(kOutputXChannelEventImage);
    vars["outputY"] = renderData.getTexture(kOutputYChannelEventImage);
    vars["PerFrameCB"]["gResolution"] = resolution;

    mpComputePass->execute(pRenderContext, uint3(resolution, 1));
}

void SimpleGradient::renderUI(Gui::Widgets& widget) {}
