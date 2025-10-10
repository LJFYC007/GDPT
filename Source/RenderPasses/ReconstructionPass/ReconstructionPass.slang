cbuffer FrameCB
{
    uint2 gResolution;
    float gEpsilon;
};

cbuffer IterationCB
{
    float gAlpha;
    float gBeta;
};

Texture2D<float4> Base;
Texture2D<float4> Variance;
Texture2D<float4> GradX;
Texture2D<float4> GradY;
Texture2D<float4> VarX;
Texture2D<float4> VarY;

RWTexture2D<float4> Solution;
RWTexture2D<float4> Residual;
RWTexture2D<float4> Direction;
RWTexture2D<float4> Ap;
RWTexture2D<float4> DotBuffer;

struct Neighborhood

{

    float3 center;

    float3 left;

    float3 right;

    float3 up;

    float3 down;

    uint hasLeft;

    uint hasRight;

    uint hasUp;

    uint hasDown;

};



// Convert raw variance samples into inverse variance weights (F^{-1}).

float3 loadInvVariance(float3 variance)
{
    float3 v = max(variance, float3(0.0));
    float3 invVar = 1.0 / (v + gEpsilon);
    return invVar;
}



Neighborhood loadBaseNeighborhood(uint2 pixel)

{

    Neighborhood nb;

    nb.center = Base[pixel].xyz;

    nb.left = nb.center;

    nb.right = nb.center;

    nb.up = nb.center;

    nb.down = nb.center;

    nb.hasLeft = pixel.x > 0 ? 1u : 0u;

    if (nb.hasLeft != 0u) nb.left = Base[uint2(pixel.x - 1, pixel.y)].xyz;

    nb.hasRight = pixel.x + 1 < gResolution.x ? 1u : 0u;

    if (nb.hasRight != 0u) nb.right = Base[uint2(pixel.x + 1, pixel.y)].xyz;

    nb.hasUp = pixel.y + 1 < gResolution.y ? 1u : 0u;

    if (nb.hasUp != 0u) nb.up = Base[uint2(pixel.x, pixel.y + 1)].xyz;

    nb.hasDown = pixel.y > 0 ? 1u : 0u;

    if (nb.hasDown != 0u) nb.down = Base[uint2(pixel.x, pixel.y - 1)].xyz;

    return nb;

}



Neighborhood loadDirectionNeighborhood(uint2 pixel)

{

    Neighborhood nb;

    nb.center = Direction[pixel].xyz;

    nb.left = nb.center;

    nb.right = nb.center;

    nb.up = nb.center;

    nb.down = nb.center;

    nb.hasLeft = pixel.x > 0 ? 1u : 0u;

    if (nb.hasLeft != 0u) nb.left = Direction[uint2(pixel.x - 1, pixel.y)].xyz;

    nb.hasRight = pixel.x + 1 < gResolution.x ? 1u : 0u;

    if (nb.hasRight != 0u) nb.right = Direction[uint2(pixel.x + 1, pixel.y)].xyz;

    nb.hasUp = pixel.y + 1 < gResolution.y ? 1u : 0u;

    if (nb.hasUp != 0u) nb.up = Direction[uint2(pixel.x, pixel.y + 1)].xyz;

    nb.hasDown = pixel.y > 0 ? 1u : 0u;

    if (nb.hasDown != 0u) nb.down = Direction[uint2(pixel.x, pixel.y - 1)].xyz;

    return nb;

}



// Apply A = S F^{-1} S^T to the vector encoded by the supplied neighborhood.

float3 applyWeightedSystem(Neighborhood nb, uint2 pixel)

{

    float3 accum = loadInvVariance(Variance[pixel].xyz) * nb.center;



    if (nb.hasRight != 0u)

    {

        float3 grad = nb.right - nb.center;

        float3 weighted = loadInvVariance(VarX[pixel].xyz) * grad;

        accum -= weighted;

    }

    if (nb.hasLeft != 0u)

    {

        uint2 leftCoord = uint2(pixel.x - 1, pixel.y);

        float3 grad = nb.center - nb.left;

        float3 weighted = loadInvVariance(VarX[leftCoord].xyz) * grad;

        accum += weighted;

    }

    if (nb.hasUp != 0u)

    {

        float3 grad = nb.up - nb.center;

        float3 weighted = loadInvVariance(VarY[pixel].xyz) * grad;

        accum -= weighted;

    }

    if (nb.hasDown != 0u)

    {

        uint2 downCoord = uint2(pixel.x, pixel.y - 1);

        float3 grad = nb.center - nb.down;

        float3 weighted = loadInvVariance(VarY[downCoord].xyz) * grad;

        accum += weighted;

    }



    return accum;

}



float3 computeB(uint2 pixel)

{

    // b = S F^{-1} c with measurements c = [Base, GradX, GradY].

    float3 result = loadInvVariance(Variance[pixel].xyz) * Base[pixel].xyz;



    if (pixel.x + 1 < gResolution.x)

    {

        float3 invVar = loadInvVariance(VarX[pixel].xyz);

        result -= invVar * GradX[pixel].xyz;

    }

    if (pixel.x > 0)

    {

        uint2 leftEdge = uint2(pixel.x - 1, pixel.y);

        float3 invVar = loadInvVariance(VarX[leftEdge].xyz);

        result += invVar * GradX[leftEdge].xyz;

    }

    if (pixel.y + 1 < gResolution.y)

    {

        float3 invVar = loadInvVariance(VarY[pixel].xyz);

        result -= invVar * GradY[pixel].xyz;

    }

    if (pixel.y > 0)

    {

        uint2 lowerEdge = uint2(pixel.x, pixel.y - 1);

        float3 invVar = loadInvVariance(VarY[lowerEdge].xyz);

        result += invVar * GradY[lowerEdge].xyz;

    }



    return result;

}



float3 applyAFromBase(uint2 pixel)

{

    Neighborhood nb = loadBaseNeighborhood(pixel);

    return applyWeightedSystem(nb, pixel);

}



float3 applyAFromDirection(uint2 pixel)

{

    Neighborhood nb = loadDirectionNeighborhood(pixel);

    return applyWeightedSystem(nb, pixel);

}

[numthreads(1, 1, 1)]
void initSystem(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gResolution)) return;

    float4 baseSample = Base[pixel];
    float3 b = computeB(pixel);
    float3 x0 = baseSample.xyz;

    float3 Ax = applyAFromBase(pixel);
    float3 r = b - Ax;

    Solution[pixel] = float4(x0, baseSample.w);
    Residual[pixel] = float4(r, 0.0);
    Direction[pixel] = float4(r, 0.0);
    Ap[pixel] = float4(0.0, 0.0, 0.0, 0.0);
    DotBuffer[pixel] = float4(dot(r, r), 0.0, 0.0, 0.0);
}

[numthreads(1, 1, 1)]
void applyA(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gResolution)) return;

    float3 ap = applyAFromDirection(pixel);
    Ap[pixel] = float4(ap, 0.0);

    float3 p = Direction[pixel].xyz;
    DotBuffer[pixel] = float4(dot(p, ap), 0.0, 0.0, 0.0);
}

[numthreads(1, 1, 1)]
void updateSolutionResidual(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gResolution)) return;

    float4 solutionSample = Solution[pixel];
    float3 direction = Direction[pixel].xyz;
    float3 ap = Ap[pixel].xyz;
    float3 residual = Residual[pixel].xyz;

    float3 newSolution = solutionSample.xyz + gAlpha * direction;
    float3 newResidual = residual - gAlpha * ap;

    Solution[pixel] = float4(newSolution, solutionSample.w);
    Residual[pixel] = float4(newResidual, 0.0);
    DotBuffer[pixel] = float4(dot(newResidual, newResidual), 0.0, 0.0, 0.0);
}

[numthreads(1, 1, 1)]
void updateDirection(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gResolution)) return;

    float3 residual = Residual[pixel].xyz;
    float3 direction = Direction[pixel].xyz;
    Direction[pixel] = float4(residual + gBeta * direction, 0.0);
}
