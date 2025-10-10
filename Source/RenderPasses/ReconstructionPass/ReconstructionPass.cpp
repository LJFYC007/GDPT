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
#include <cmath>
#include <chrono>

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReconstructionPass>();
}

namespace
{
const std::string kBaseChannelEventImage = "base";
const std::string kVarianceChannelEventImage = "variance";
const std::string kGradXChannelEventImage = "gradX";
const std::string kGradYChannelEventImage = "gradY";
const std::string kVarXChannelEventImage = "varX";
const std::string kVarYChannelEventImage = "varY";
const std::string kOutputChannelEventImage = "output";
const std::string kNum = "num";
const float kMinDivisor = 1e-12f;
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
    RenderPassReflection reflector;
    reflector.addInput(kBaseChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kVarianceChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kGradXChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kGradYChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kVarXChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kVarYChannelEventImage, "").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput(kOutputChannelEventImage, "")
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Float);
    return reflector;
}

void ReconstructionPass::prepareComputePasses()
{
    if (!mpInitPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary("RenderPasses/ReconstructionPass/ReconstructionPass.slang");
        desc.csEntry("initSystem");
        mpInitPass = ComputePass::create(mpDevice, desc);
    }
    if (!mpApplyAPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary("RenderPasses/ReconstructionPass/ReconstructionPass.slang");
        desc.csEntry("applyA");
        mpApplyAPass = ComputePass::create(mpDevice, desc);
    }
    if (!mpUpdateSolutionPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary("RenderPasses/ReconstructionPass/ReconstructionPass.slang");
        desc.csEntry("updateSolutionResidual");
        mpUpdateSolutionPass = ComputePass::create(mpDevice, desc);
    }
    if (!mpUpdateDirectionPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary("RenderPasses/ReconstructionPass/ReconstructionPass.slang");
        desc.csEntry("updateDirection");
        mpUpdateDirectionPass = ComputePass::create(mpDevice, desc);
    }
    if (!mpReduction)
    {
        mpReduction = std::make_unique<ParallelReduction>(mpDevice);
    }
}

void ReconstructionPass::allocateInternalTextures(uint32_t width, uint32_t height)
{
    auto createTexture = [&](ref<Texture>& texture)
    {
        if (texture && texture->getWidth() == width && texture->getHeight() == height)
            return;

        texture = mpDevice->createTexture2D(
            width,
            height,
            ResourceFormat::RGBA32Float,
            1,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
    };

    createTexture(mpResidual);
    createTexture(mpDirection);
    createTexture(mpAp);
    createTexture(mpDotBuffer);
    createTexture(mpPreconditioner);
}

void ReconstructionPass::bindCommonResources(ref<ComputePass> pPass, const ref<Texture>& pSolution, const ref<Texture>& pBase,
    const ref<Texture>& pVariance, const ref<Texture>& pGradX, const ref<Texture>& pGradY,
    const ref<Texture>& pVarX, const ref<Texture>& pVarY)
{
    if (!pPass)
        return;

    auto vars = pPass->getRootVar();
    const uint2 resolution = {pSolution->getWidth(), pSolution->getHeight()};
    vars["FrameCB"]["gResolution"] = resolution;
    vars["FrameCB"]["gEpsilon"] = mEpsilon;
    vars["Base"] = pBase;
    vars["Variance"] = pVariance;
    vars["GradX"] = pGradX;
    vars["GradY"] = pGradY;
    vars["VarX"] = pVarX;
    vars["VarY"] = pVarY;
    vars["Solution"] = pSolution;
    vars["Residual"] = mpResidual;
    vars["Direction"] = mpDirection;
    vars["Ap"] = mpAp;
    vars["DotBuffer"] = mpDotBuffer;
    vars["Preconditioner"] = mpPreconditioner;
}

float ReconstructionPass::reduceDotProduct(RenderContext* pRenderContext)
{
    // Flush pending Falcor/CUDA work so the latest UAV writes are visible before reading.
    mpDevice->wait();
    pRenderContext->waitForFalcor();

    // Run the reduction on the GPU and read back the accumulated value.
    pRenderContext->resourceBarrier(mpDotBuffer.get(), Resource::State::ShaderResource);
    float4 reductionResult = float4(0.f);
    mpReduction->execute<float4>(pRenderContext, mpDotBuffer, ParallelReduction::Type::Sum, &reductionResult);
    pRenderContext->resourceBarrier(mpDotBuffer.get(), Resource::State::UnorderedAccess);

    return reductionResult.x;
}

void ReconstructionPass::placeUavBarriers(RenderContext* pRenderContext, const ref<Texture>& pSolution)
{
    pRenderContext->uavBarrier(pSolution.get());
    pRenderContext->uavBarrier(mpResidual.get());
    pRenderContext->uavBarrier(mpDirection.get());
    pRenderContext->uavBarrier(mpAp.get());
    pRenderContext->uavBarrier(mpDotBuffer.get());
    pRenderContext->uavBarrier(mpPreconditioner.get());  // Add barrier for preconditioner
}

void ReconstructionPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;
    auto startTime = std::chrono::high_resolution_clock::now();

    auto pOutput = renderData.getTexture(kOutputChannelEventImage);
    auto pBase = renderData.getTexture(kBaseChannelEventImage);

    auto pVariance = renderData.getTexture(kVarianceChannelEventImage);
    auto pGradX = renderData.getTexture(kGradXChannelEventImage);
    auto pGradY = renderData.getTexture(kGradYChannelEventImage);
    auto pVarX = renderData.getTexture(kVarXChannelEventImage);
    auto pVarY = renderData.getTexture(kVarYChannelEventImage);
    const uint32_t width = pOutput->getWidth();
    const uint32_t height = pOutput->getHeight();

    prepareComputePasses();
    allocateInternalTextures(width, height);

    auto bindPass = [&](ref<ComputePass> pPass)
    {
        bindCommonResources(pPass, pOutput, pBase, pVariance, pGradX, pGradY, pVarX, pVarY);
    };

    bindPass(mpInitPass);
    bindPass(mpApplyAPass);
    bindPass(mpUpdateSolutionPass);
    bindPass(mpUpdateDirectionPass);

    const uint3 dispatchDims = uint3(width, height, 1u);

    mpInitPass->getRootVar()["IterationCB"]["gAlpha"] = 0.f;
    mpInitPass->getRootVar()["IterationCB"]["gBeta"] = 0.f;
    mpInitPass->execute(pRenderContext, dispatchDims);
    placeUavBarriers(pRenderContext, pOutput);
    float residualNorm = reduceDotProduct(pRenderContext);
    if (!std::isfinite(residualNorm) || residualNorm <= mTolerance)
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        std::cout << "Execute function took: " << duration.count() << " microseconds (early return: residual converged)" << std::endl;
        return;
    }

    float prevResidualNorm = residualNorm;

    ++ frame;
    std::cout << "Frame: " << frame << std::endl;
    for (int iteration = 0; iteration < num; ++iteration)
    {
        mpApplyAPass->execute(pRenderContext, dispatchDims);
        placeUavBarriers(pRenderContext, pOutput);
        float dotPAp = reduceDotProduct(pRenderContext);
        if (!std::isfinite(dotPAp) || std::fabs(dotPAp) < kMinDivisor)
            break;

        const float alpha = prevResidualNorm / dotPAp;
        auto updateVars = mpUpdateSolutionPass->getRootVar();
        updateVars["IterationCB"]["gAlpha"] = alpha;
        updateVars["IterationCB"]["gBeta"] = 0.f;
        mpUpdateSolutionPass->execute(pRenderContext, dispatchDims);
        placeUavBarriers(pRenderContext, pOutput);

        float newResidualNorm = reduceDotProduct(pRenderContext);
        if (!std::isfinite(newResidualNorm) || newResidualNorm <= mTolerance)
            break;

        const float beta = newResidualNorm / prevResidualNorm;
        auto dirVars = mpUpdateDirectionPass->getRootVar();
        dirVars["IterationCB"]["gAlpha"] = 0.f;
        dirVars["IterationCB"]["gBeta"] = beta;
        mpUpdateDirectionPass->execute(pRenderContext, dispatchDims);
        placeUavBarriers(pRenderContext, pOutput);

        // std::cout << "Iteration " << iteration << ", prev residual norm = " << std::sqrt(prevResidualNorm) << ", new residual norm = " << std::sqrt(newResidualNorm) << ", alpha = " << alpha << ", beta = " << beta << std::endl;
        prevResidualNorm = newResidualNorm;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "Execute function took: " << duration.count() << " microseconds (" << duration.count() / 1000.0 << " ms)" << std::endl;
}


void ReconstructionPass::renderUI(Gui::Widgets& widget)
{
    widget.var("Number of iterations", num, 1, 100);
}
