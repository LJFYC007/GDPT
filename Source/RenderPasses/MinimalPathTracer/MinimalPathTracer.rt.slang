/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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

/**
 * Minimal path tracer.
 *
 * The purpose is to use it for validation of more complex renderers.
 * The implementation here should be kept as simple/naive as possible.
 *
 * At each hit point (including the primary hit loaded from the V-buffer),
 * analytic light sources (point, directional) are sampled uniformly using
 * 1 shadow ray, and 1 scatter ray is traced to sample the hemisphere.
 * At hit/miss the scatter ray includes light from emissive surface and
 * the environment map, respectively. Traversal stops at a fixed path length.
 *
 * Each type of light (analytic, emissive, env map) can be individually
 * enabled/disabled from the host. This clutters the code a bit, but it is
 * important as not all other renderes may support all three light types.
 *
 * The host sets the following defines:
 *
 * MAX_BOUNCES             Maximum number of indirect bounces (0 means no indirect).
 * COMPUTE_DIRECT          Nonzero if direct illumination should be included.
 * USE_IMPORTANCE_SAMPLING Nonzero if importance sampling should be used for sampling materials.
 * USE_ANALYTIC_LIGHTS     Nonzero if Falcor's analytic lights should be used.
 * USE_EMISSIVE_LIGHTS     Nonzero if emissive geometry should be used as lights.
 * USE_ENV_LIGHT           Nonzero if env map is available and should be used as light source.
 * USE_ENV_BACKGROUND      Nonzero if env map is available and should be used as background.
 * is_valid_<name>         1 if optional I/O buffer with this name should be used.
 */

#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;

cbuffer CB
{
    float3 gNormal;
    float3 gPoint1;
    float3 gPoint2;
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
    EmissiveLightSampler emissiveSampler;
}

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW; // Optional

// Outputs
RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gOutputGradientX;
RWTexture2D<float4> gOutputGradientY;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const uint kMaxBounces = MAX_BOUNCES;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

/**
 * Payload for shadow ray.
 */
struct ShadowRayData
{
    bool visible;
};

/**
 * Payload for scatter ray in gradient domain path tracing (up to 88B).
 */
struct ScatterRayData
{
    float3 radiance;        ///< Accumulated outgoing radiance from path.
    bool terminated;        ///< Set to true when path is terminated.
    float3 throughput;      ///< Current path throughput. This is updated at each path vertex.
    uint pathLength;        ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3 origin;          ///< Next path segment origin.
    float pdf;              ///< Path probability density function value.
    float3 direction;       ///< Next path segment direction.

    // Gradient domain specific fields
    bool isOffsetPath;      ///< True if this is an offset path for gradient computation.
    float3 targetPoint;     ///< Target point for offset path connection.
    uint instanceID;        ///< Instance ID of the hit geometry.
    float3 normal;          ///< Surface normal at hit point.
    float lastPdf;          ///< Last bsdf pdf value for the scattered ray
    float3 hitPoint;        ///< World space hit point position.
    float lightPdf;
    float3 lightDir;
    float correctPdf;
    float3 lightRadiance;

    SampleGenerator sg;     ///< Per-ray state for the sample generator (up to 16B).

    /**
     * Initializes ray payload with default parameters.
     */
    __init(SampleGenerator sg)
    {
        this.terminated = false;
        this.pathLength = 0;
        this.radiance = float3(0, 0, 0);
        this.throughput = float3(1, 1, 1);
        this.pdf = 1.f;
        this.isOffsetPath = false;
        this.origin = float3(0, 0, 0);
        this.normal = float3(1, 1, 1);
        this.direction = float3(0, 0, 0);
        this.sg = sg;
    }
};

/**
 * Setup ShadingData based on loaded vertex/material attributes for a hit point.
 * @param[in] hit Hit information.
 * @param[in] rayOrigin Ray origin.
 * @param[in] rayDir Normalized ray direction.
 * @return ShadingData struct.
 */
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir)
{
    VertexData v = {};
    uint materialID = {};
    uint instanceID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
        instanceID = triangleHit.instanceID.index;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
        instanceID = displacedTriangleHit.instanceID.index;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
        instanceID = curveHit.instanceID.index;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
        instanceID = sdfGridHit.instanceID.index;
    }
#endif

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
    sd.instanceID = instanceID;

    return sd;
}

/**
 * Returns the primary ray's direction.
 */
float3 getPrimaryRayDir(uint2 launchIndex, uint2 launchDim, const Camera camera)
{
    if (is_valid(gViewW))
    {
        // If we have the view vector bound as a buffer, just fetch it. No need to compute anything.
        return -gViewW[launchIndex].xyz;
    }
    else
    {
        // Compute the view vector. This must exactly match what the G-buffer pass is doing (jitter etc.).
        // Note that we do not take depth-of-field into account as it would require exactly matching the
        // sample generator between the passes, which is error prone. The host side will issue a warning instead.
        return camera.computeRayPinhole(launchIndex, launchDim).dir;
    }
}

/**
 * Traces a shadow ray towards a light source.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
 * @return True if light is visible, false otherwise.
 */
bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    ShadowRayData rayData;
    rayData.visible = false; // Set to true by miss shader if ray is not terminated before
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff /* instanceInclusionMask */,
        1 /* hitIdx */,
        rayTypeCount,
        1 /* missIdx */,
        ray,
        rayData
    );

    return rayData.visible;
}

/**
 * Traces a scatter ray based on ray parameters stored in the ray payload.
 * @param[in] rayData Describes the ray parameters. The struct is modified based on the result.
 */
void traceScatterRay(inout ScatterRayData rayData)
{
    RayDesc ray;
    ray.Origin = rayData.origin;
    ray.Direction = rayData.direction;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    uint rayFlags = 0; // TODO: Set cull mode from the app
    TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
}

/** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
    \param[in] n0 Number of samples taken from the first sampling strategy.
    \param[in] p0 Pdf for the first sampling strategy.
    \param[in] n1 Number of samples taken from the second sampling strategy.
    \param[in] p1 Pdf for the second sampling strategy.
    \return Weight for the contribution from the first strategy (p0).
*/
float evalMIS(float n0, float p0, float n1, float p1)
{
    // Power two heuristic
    float q0 = (n0 * p0) * (n0 * p0);
    float q1 = (n1 * p1) * (n1 * p1);
    return q0 / (q0 + q1);
}

/**
 * Evaluates the direct illumination from analytic lights.
 * This function samples Falcor's light list uniformly with one shadow ray.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in,out] sg SampleGenerator object.
 * @return Outgoing radiance in view direction.
 */
float3 evalDirectEmissive(const ShadingData sd, const IMaterialInstance mi, inout ScatterRayData rayData, out TriangleLightHit hit)
{
    rayData.lightPdf = 0.f;
    TriangleLightSample tls;
    if (!emissiveSampler.sampleLight(sd.posW, sd.faceN, false, rayData.sg, tls))
        return float3(0.f);

    // Reject sample if not in the hemisphere of a BSDF lobe.
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasReflection = lobeTypes & uint(LobeType::Reflection);
    const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
    float NdotL = dot(sd.getOrientedFaceNormal(), tls.dir);
    if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return float3(0.f);

    // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
    const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, tls.dir) >= 0.f ? sd.faceN : -sd.faceN);
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    float3 toLight = lightPos - origin;

    // Test visibility by tracing a shadow ray.
    bool V = traceShadowRay(origin, normalize(toLight), length(toLight));
    if (!V)
        return float3(0.f);

    hit.triangleIndex = tls.triangleIndex;
    hit.normalW = tls.normalW;
    hit.posW = tls.posW;

    // Evaluate contribution.
    rayData.lightDir = tls.dir;
    rayData.lightPdf = tls.pdf;
    return tls.Le;
}

/**
 * Generate a new scatter ray or terminate for gradient domain path tracing.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in] isCurveHit True if on curve hit.
 * @param[in] rayOrigin Ray origin for the new ray.
 * @param[in,out] rayData Ray payload.
 * @return True if the path continues.
 */
bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, bool isCurveHit, float3 rayOrigin, inout ScatterRayData rayData)
{
    // Sample material.
    BSDFSample bsdfSample;
    if (rayData.isOffsetPath || mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling))
    {
        rayData.origin = rayOrigin;
        if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
            rayData.origin = sd.computeRayOrigin(false);

        if (rayData.isOffsetPath)
        {
            rayData.direction = normalize(rayData.targetPoint - rayData.origin);
            rayData.throughput *= mi.eval(sd, rayData.direction, rayData.sg);
            rayData.lastPdf = mi.evalPdf(sd, rayData.direction);
            rayData.pdf *= rayData.lastPdf;
        }
        else
        {
            rayData.direction = bsdfSample.wo;
            // If deltaReflection or deltaTransmission, the returned bsdfSample.pdf is zero
            if (bsdfSample.pdf == 0.f)
            {
                rayData.throughput *= bsdfSample.weight;
                rayData.lastPdf = 0.f;
            }
            else
            {
                rayData.throughput *= mi.eval(sd, rayData.direction, rayData.sg);
                rayData.lastPdf = mi.evalPdf(sd, rayData.direction);
                rayData.pdf *= rayData.lastPdf;
            }
        }
        return any(rayData.throughput > 0.f);
    }

    rayData.pdf = 0.f;
    return false;
}

/**
 * Process a hit.
 * Loads the shading data, samples analytic lights and samples a new scatter ray.
 * Terminates the path if maximum number of bounces is reached.
 * @param[in] hit Hit info.
 * @param[in,out] rayData Ray payload.
 *
 */
void handleHit(const HitInfo hit, inout ScatterRayData rayData)
{
    const bool isCurveHit = hit.getType() == HitType::Curve;

    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction);
    rayData.hitPoint = sd.posW;
    rayData.normal = sd.frame.N;
    rayData.instanceID = sd.instanceID;
    rayData.radiance = float3(0.f);

    // Create material instance.
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    // If the material is emissive, we accumulate it only on the first bounce, since we enabled NEE.
    if (kUseEmissiveLights && any(mi.getProperties(sd).emission > 0.f))
    {
        float misWeight = 1.f;
        // if (rayData.lastPdf > 0.f) // delta materials have zero pdf, so we skip MIS weight computation
        // {
        //     TriangleLightHit lightHit;
        //     TriangleHit triangleHit = hit.getTriangleHit();
        //     lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
        //     lightHit.posW = sd.posW;
        //     lightHit.normalW = sd.getOrientedFaceNormal();

        //     float bsdfPdf = rayData.lastPdf;
        //     float lightPdf = emissiveSampler.evalPdf(rayData.origin, rayData.normal, false, lightHit);
        //     misWeight = evalMIS(1, bsdfPdf, 1, lightPdf);
        // }
        rayData.radiance = misWeight * rayData.throughput * mi.getProperties(sd).emission;
        rayData.lightPdf = 1.f;
        rayData.terminated = true;
        return;
    }

    // Check whether to terminate based on max depth.
    if (rayData.pathLength >= kMaxBounces)
    {
        rayData.terminated = true;
        return;
    }

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin;
    if (isCurveHit)
    {
        // For curves, we set the new origin at the sphere center.
        rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
    }
    else
    {
        rayOrigin = sd.computeRayOrigin();
    }

    // Here we implement the NEE
    {
        // TriangleLightHit tlh;
        // if (rayData.isOffsetPath == false)
        //     rayData.lightRadiance = evalDirectEmissive(sd, mi, rayData, tlh);
        // if (any(rayData.lightRadiance > 0.f))
        // {
        //     float bsdfPdf = mi.evalPdf(sd, rayData.lightDir);
        //     // delta materials have zero pdf, so we skip MIS weight computation
        //     float misWeight = bsdfPdf > 0.f ? evalMIS(1, rayData.lightPdf, 1, bsdfPdf) : 0.f;
        //     rayData.radiance = rayData.throughput * misWeight * mi.eval(sd, rayData.lightDir, rayData.sg) * rayData.lightRadiance;
        // }
    }

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, rayData))
    {
        rayData.terminated = true;
        return;
    }

    rayData.pathLength++;
}

// Compute geometric Jacobian for gradient domain path tracing
float computeJacobian(float3 basePoint1, float3 basePoint2, float3 offsetPoint1, float3 normalPoint2)
{
    float baseDist = length(basePoint2 - basePoint1);
    float offsetDist = length(basePoint2 - offsetPoint1);
    float baseCosine = dot(normalize(normalPoint2), normalize(basePoint1 - basePoint2));
    float offsetCosine = dot(normalize(normalPoint2), normalize(offsetPoint1 - basePoint2));
    if (baseCosine == 0.f || offsetCosine == 0.f)
        return 0.f;
    return abs(offsetCosine * baseDist * baseDist) / (FLT_EPSILON + abs(baseCosine * offsetDist * offsetDist));
}


/**
 * This is the main entry point for the gradient domain path tracer.
 *
 * For each pixel, traces both a base path and offset paths to compute gradients.
 * The base path follows standard path tracing, while offset paths are connected
 * to corresponding vertices in the base path to estimate image gradients.
 *
 * @param[in] pixel Pixel to trace a path for.
 * @param[in] frameDim Dimension of the frame in pixels.
 */
void tracePath(const uint2 pixel, const uint2 frameDim)
{
    float3 basePathRadiance[10], offsetPathRadiance[10];
    float basePathPdf[10] = { 0 }, offsetPathPdf[10] = { 0 };
    float3 basePathVertices[10], basePathNormals[10];
    uint basePathInstanceIDs[10];
    float geometricJacobian[10];

    float3 lightRadiance[10], lightDir[10];
    float lightPdf[10] = { 0 } ;
    float offsetLightPdf = 0.f; // this is only for first bounce
    TriangleLightHit baseLightHit;
    uint lightLength = 1;

    // Trace base path from the primary hit point
    {
        const float3 primaryRayOrigin = gScene.camera.getPosition();
        const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);

        const HitInfo hit = HitInfo(gVBuffer[pixel]);
        // Pixel represents a valid primary hit. Compute its contribution.

        const bool isCurveHit = hit.getType() == HitType::Curve;

        // Load shading data.
        ShadingData sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir);

        // Create material instance at shading point.
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

        // Compute ray origin for new rays spawned from the G-buffer.
        float3 rayOrigin;
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
        }
        else
        {
            rayOrigin = sd.computeRayOrigin();
        }

        // Prepare ray payload for base path.
        ScatterRayData baseRayData = ScatterRayData(sg);

        // Here we implement the NEE
        {
            // baseRayData.lightRadiance = evalDirectEmissive(sd, mi, baseRayData, baseLightHit);
            // if (any(baseRayData.lightRadiance > 0.f))
            // {
            //     float bsdfPdf = mi.evalPdf(sd, baseRayData.lightDir);
            //     // delta materials have zero pdf, so we skip MIS weight computation
            //     float misWeight = bsdfPdf > 0.f ? evalMIS(1, baseRayData.lightPdf, 1, bsdfPdf) : 0.f;
            //     basePathRadiance[1] = baseRayData.throughput * misWeight * mi.eval(sd, baseRayData.lightDir, baseRayData.sg) * baseRayData.lightRadiance;
            // }
            // else
            //     basePathRadiance[1] = float3(0.f);
        }

        // Generate scatter ray.
        if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, baseRayData))
            baseRayData.terminated = true;

        // Store first vertex of base path
        basePathVertices[1] = sd.posW;
        basePathNormals[1] = sd.frame.N;
        basePathPdf[0] = 1.f;
        basePathPdf[1] = baseRayData.pdf;
        lightRadiance[1] = baseRayData.lightRadiance;
        lightPdf[1] = baseRayData.lightPdf;

        // Follow base path into the scene and compute its total contribution.
        for (uint depth = 0; depth <= kMaxBounces && !baseRayData.terminated; depth++)
        {
            // Trace scatter ray. If it hits geometry, the closest hit shader samples
            // direct illumination and generates the next scatter ray.
            traceScatterRay(baseRayData);

            basePathVertices[depth + 2] = baseRayData.hitPoint;
            basePathNormals[depth + 2] = baseRayData.normal;
            basePathInstanceIDs[depth + 2] = baseRayData.instanceID;
            basePathPdf[depth + 2] = baseRayData.pdf;
            basePathRadiance[depth + 2] = baseRayData.radiance;
            lightRadiance[depth + 2] = baseRayData.lightRadiance;
            lightDir[depth + 2] = baseRayData.lightDir;
            lightPdf[depth + 2] = baseRayData.lightPdf;
            lightLength = depth + 2;
        }
    }

    // Early exit if base path has no contribution
    // if (!any(baseRadiance > 0.f) || basePdf == 0.f)
    //     return;

    // Determine offset direction for gradient estimation (cycles through 4 directions)
    uint2 pixelOffset;
    if (gFrameCount % 4 == 0) pixelOffset = uint2(1, 0);
    else if (gFrameCount % 4 == 1) pixelOffset = uint2(0, 1);
    else if (gFrameCount % 4 == 2) pixelOffset = uint2(-1, 0);
    else pixelOffset = uint2(0, -1);

    // Clamp offset pixel to frame boundaries
    if (pixel.x + pixelOffset.x < 0) pixelOffset.x = -pixelOffset.x;
    if (pixel.y + pixelOffset.y < 0) pixelOffset.y = -pixelOffset.y;
    if (pixel.x + pixelOffset.x >= frameDim.x) pixelOffset.x = -pixelOffset.x;
    if (pixel.y + pixelOffset.y >= frameDim.y) pixelOffset.y = -pixelOffset.y;

    // Trace offset path for gradient computation
    {
        const float3 primaryRayOrigin = gScene.camera.getPosition();
        const float3 primaryRayDir = getPrimaryRayDir(pixel + pixelOffset, frameDim, gScene.camera);

        const HitInfo hit = HitInfo(gVBuffer[pixel + pixelOffset]);
        // Pixel represents a valid primary hit. Compute its contribution.

        const bool isCurveHit = hit.getType() == HitType::Curve;

        // Load shading data.
        ShadingData sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir);

        // Create material instance at shading point.
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

        // Compute ray origin for new rays spawned from the G-buffer.
        float3 rayOrigin;
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
        }
        else
        {
            rayOrigin = sd.computeRayOrigin();
        }

        // Prepare ray payload for offset path.
        ScatterRayData offsetRayData = ScatterRayData(sg);
        offsetRayData.isOffsetPath = true;
        offsetRayData.targetPoint = basePathVertices[2];  // Connect to first bounce of base path
        offsetRayData.instanceID = sd.instanceID;

        // Here we implement the NEE
        {
            // if (any(lightRadiance[1] > 0.f))
            // {
            //     offsetLightPdf = emissiveSampler.evalPdf(rayOrigin, sd.frame.N, false, baseLightHit);
            //     float3 dir = normalize(baseLightHit.posW - rayOrigin);
            //     float3 lightPos = computeRayOrigin(baseLightHit.posW, baseLightHit.normalW);
            //     float3 toLight = lightPos - rayOrigin;
            //     bool V = traceShadowRay(rayOrigin, normalize(toLight), length(toLight));
            //     if (V)
            //     {
            //         float bsdfPdf = mi.evalPdf(sd, dir);
            //         // delta materials have zero pdf, so we skip MIS weight computation
            //         float misWeight = bsdfPdf > 0.f ? evalMIS(1, offsetLightPdf, 1, bsdfPdf) : 0.f;

            //         geometricJacobian[1] = computeJacobian(basePathVertices[1], baseLightHit.posW, sd.posW, baseLightHit.normalW);
            //         offsetPathRadiance[1] = offsetRayData.throughput * misWeight * mi.eval(sd, dir, offsetRayData.sg) * lightRadiance[1];
            //     }
            //     else
            //         offsetPathRadiance[1] = float3(0.f);
            // }
            // else
            //     offsetPathRadiance[1] = float3(0.f);
        }

        // Generate scatter ray.
        if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, offsetRayData))
            offsetRayData.terminated = true;

        geometricJacobian[2] = computeJacobian(basePathVertices[1], basePathVertices[2], sd.posW, basePathNormals[2]);
        if (geometricJacobian[2] == 0.f)
        {
            offsetRayData.terminated = true;
            offsetRayData.pdf = 0.f;
        }
        offsetPathPdf[0] = 1.f;
        offsetPathPdf[1] = offsetRayData.pdf;

        // Follow offset path into the scene with connection to base path vertices
        for (uint depth = 0; depth <= kMaxBounces && !offsetRayData.terminated; depth++)
        {
            // Set next target point for offset path connection
            offsetRayData.targetPoint = basePathVertices[depth + 3];
            offsetRayData.lightRadiance = lightRadiance[depth + 2];
            offsetRayData.lightDir = lightDir[depth + 2];
            offsetRayData.lightPdf = lightPdf[depth + 2];

            traceScatterRay(offsetRayData);

            // Terminate if hit different geometry instance (path diverged)
            if (offsetRayData.instanceID != basePathInstanceIDs[depth + 2])
            {
                offsetRayData.terminated = true;
                offsetRayData.pdf = 0.f;
            }

            offsetPathRadiance[depth + 2] = offsetRayData.radiance;
            offsetPathPdf[depth + 2] = offsetRayData.pdf;
        }
    }

    // Compute importance weight for gradient estimation
    float3 baseColor = float3(0.f);
    float3 offsetColor = float3(0.f);
    float3 gradient = float3(0.f);
    for (uint i = 1; i <= lightLength; ++i)
    {
        float basePdf = basePathPdf[i - 1] * lightPdf[i];
        if (basePdf == 0.f)
            continue;
        float offsetPdf = offsetPathPdf[i - 1] * (i == 1 ? offsetLightPdf : lightPdf[i]);
        float geomJac = i == 1 ? geometricJacobian[1] : geometricJacobian[2];
        baseColor += basePathRadiance[i] / basePdf;

        // it seems bathPathRadiance[2] sometimes will be nan,
        if (offsetPdf == 0.f || geomJac == 0.f || any(isnan(geomJac)))
        {
            gradient += basePathRadiance[i] / basePdf;
        }
        else
        {
            offsetColor += geomJac * offsetPathRadiance[i] / basePdf;
            float weight = basePdf / (basePdf + offsetPdf * geomJac);
            gradient += weight * (basePathRadiance[i] / basePdf - geomJac * offsetPathRadiance[i] / basePdf);
        }
    }

    // Output base color
    gOutputColor[pixel] = float4(baseColor, 1.f);
    // gOutputGradientX[pixel] = float4(offsetColor - baseColor, 1.f);
    // gOutputGradientY[pixel] = float4(offsetColor, 1.f);

    // Output gradients based on offset direction
    // Scale factor of 4 accounts for the fact that we only compute gradients 1/4 of the time
    if (pixelOffset.x == 1 && pixelOffset.y == 0)
        gOutputGradientX[pixel] += 4 * float4(-gradient, 1.f);
    if (pixelOffset.x == 0 && pixelOffset.y == 1)
        gOutputGradientY[pixel] += 4 * float4(-gradient, 1.f);
    if (pixelOffset.x == -1 && pixelOffset.y == 0)
        gOutputGradientX[pixel + pixelOffset] += 4 * float4(gradient, 1.f);
    if (pixelOffset.x == 0 && pixelOffset.y == -1)
        gOutputGradientY[pixel + pixelOffset] += 4 * float4(gradient, 1.f);
}

//
// Shader entry points for miss shaders.
//

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the ray as terminated.
    rayData.terminated = true;
    rayData.pdf = 0.f;
    rayData.radiance = float3(0.f);

    // Add contribution from distant light (env map) in this direction.
    // if (kUseEnvLight && (kComputeDirect || rayData.pathLength > 0))
    // {
    //     float3 Le = gScene.envMap.eval(WorldRayDirection());
    //     rayData.radiance += rayData.thp * Le;
    // }
}

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void scatterTriangleMeshAnyHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void scatterTriangleMeshClosestHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(triangleHit), rayData);
}

[shader("anyhit")]
void shadowTriangleMeshAnyHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

//
// Shader entry points for DisplacedTriangleMesh hit groups.
//

[shader("intersection")]
void displacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterDisplacedTriangleMeshClosestHit(inout ScatterRayData rayData, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;
    handleHit(HitInfo(displacedTriangleHit), rayData);
}

//
// Shader entry points for Curve hit groups.
//

[shader("intersection")]
void curveIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    CurveIntersector::Attribs attribs;
    float t;
    if (CurveIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterCurveClosestHit(inout ScatterRayData rayData, CurveIntersector::Attribs attribs)
{
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(curveHit), rayData);
}

//
// Shader entry points for SDFGrid hit groups.
//

[shader("intersection")]
void sdfGridIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    SDFGridHitData sdfGridHitData;
    float t;
    if (SDFGridIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), sdfGridHitData, t))
    {
        ReportHit(t, 0, sdfGridHitData);
    }
}

[shader("closesthit")]
void scatterSdfGridClosestHit(inout ScatterRayData rayData, SDFGridHitData sdfGridHitData)
{
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;
    handleHit(HitInfo(sdfGridHit), rayData);
}

//
// Shader entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    tracePath(pixel, frameDim);
}
