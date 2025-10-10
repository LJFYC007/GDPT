cbuffer FrameCB {
    uint2 gResolution;
    float gEpsilon;
};

cbuffer IterationCB {
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
RWTexture2D<float4> Preconditioner;

// Convert raw variance samples into inverse variance weights (F^{-1})
float3 loadInvVariance(float3 variance) {
    return 1.0 / (max(variance, float3(0.0)) + gEpsilon);
}

// Compute diagonal of A = S F^{-1} S^T for preconditioning
// diag(A) = sum of squared S entries times corresponding F^{-1} weights
float3 computeDiagonalA(uint2 pixel) {
    // Start with contribution from pixel itself (S[i,i] = 1)
    float3 diag = loadInvVariance(Variance[pixel].xyz);

    // Contribution from x-gradients
    // Right edge: S[i, offsetX + y*(W-1) + x] = -1
    if (pixel.x + 1 < gResolution.x) {
        diag += loadInvVariance(VarX[pixel].xyz);
    }
    // Left edge: S[i, offsetX + y*(W-1) + (x-1)] = +1
    if (pixel.x > 0) {
        diag += loadInvVariance(VarX[uint2(pixel.x - 1, pixel.y)].xyz);
    }

    // Contribution from y-gradients
    // Down edge: S[i, offsetY + y*W + x] = -1
    if (pixel.y + 1 < gResolution.y) {
        diag += loadInvVariance(VarY[pixel].xyz);
    }
    // Up edge: S[i, offsetY + (y-1)*W + x] = +1
    if (pixel.y > 0) {
        diag += loadInvVariance(VarY[uint2(pixel.x, pixel.y - 1)].xyz);
    }

    // Add small regularization and return inverse for preconditioning
    return 1.0 / (max(diag, 1e-12) + 1e-8);
}

// Apply diagonal preconditioner: M^{-1} v = v / diag(A)
float3 applyPreconditioner(uint2 pixel, float3 v) {
    return Preconditioner[pixel].xyz * v;
}

// Apply A = S F^{-1} S^T to a texture at the given pixel
float3 applyWeightedSystem(Texture2D<float4> tex, uint2 pixel) {
    float3 center = tex[pixel].xyz;
    float3 accum = loadInvVariance(Variance[pixel].xyz) * center;

    // Right neighbor - combine variance load with computation
    if (pixel.x + 1 < gResolution.x) {
        float3 invVar = loadInvVariance(VarX[pixel].xyz);
        accum -= invVar * (tex[uint2(pixel.x + 1, pixel.y)].xyz - center);
    }

    // Left neighbor
    if (pixel.x > 0) {
        uint2 leftCoord = uint2(pixel.x - 1, pixel.y);
        float3 invVar = loadInvVariance(VarX[leftCoord].xyz);
        accum += invVar * (center - tex[leftCoord].xyz);
    }

    // Down neighbor
    if (pixel.y + 1 < gResolution.y) {
        float3 invVar = loadInvVariance(VarY[pixel].xyz);
        accum -= invVar * (tex[uint2(pixel.x, pixel.y + 1)].xyz - center);
    }

    // Up neighbor
    if (pixel.y > 0) {
        uint2 upCoord = uint2(pixel.x, pixel.y - 1);
        float3 invVar = loadInvVariance(VarY[upCoord].xyz);
        accum += invVar * (center - tex[upCoord].xyz);
    }

    return accum;
}

// Overload for RWTexture2D
float3 applyWeightedSystem(RWTexture2D<float4> tex, uint2 pixel) {
    float3 center = tex[pixel].xyz;
    float3 accum = loadInvVariance(Variance[pixel].xyz) * center;

    // Right neighbor - combine variance load with computation
    if (pixel.x + 1 < gResolution.x) {
        float3 invVar = loadInvVariance(VarX[pixel].xyz);
        accum -= invVar * (tex[uint2(pixel.x + 1, pixel.y)].xyz - center);
    }

    // Left neighbor
    if (pixel.x > 0) {
        uint2 leftCoord = uint2(pixel.x - 1, pixel.y);
        float3 invVar = loadInvVariance(VarX[leftCoord].xyz);
        accum += invVar * (center - tex[leftCoord].xyz);
    }

    // Down neighbor
    if (pixel.y + 1 < gResolution.y) {
        float3 invVar = loadInvVariance(VarY[pixel].xyz);
        accum -= invVar * (tex[uint2(pixel.x, pixel.y + 1)].xyz - center);
    }

    // Up neighbor
    if (pixel.y > 0) {
        uint2 upCoord = uint2(pixel.x, pixel.y - 1);
        float3 invVar = loadInvVariance(VarY[upCoord].xyz);
        accum += invVar * (center - tex[upCoord].xyz);
    }

    return accum;
}

// Compute b = S F^{-1} c with measurements c = [Base, GradX, GradY]
float3 computeB(uint2 pixel) {
    float3 result = loadInvVariance(Variance[pixel].xyz) * Base[pixel].xyz;

    if (pixel.x + 1 < gResolution.x) {
        result -= loadInvVariance(VarX[pixel].xyz) * GradX[pixel].xyz;
    }
    if (pixel.x > 0) {
        result += loadInvVariance(VarX[uint2(pixel.x - 1, pixel.y)].xyz) * GradX[uint2(pixel.x - 1, pixel.y)].xyz;
    }
    if (pixel.y + 1 < gResolution.y) {
        result -= loadInvVariance(VarY[pixel].xyz) * GradY[pixel].xyz;
    }
    if (pixel.y > 0) {
        result += loadInvVariance(VarY[uint2(pixel.x, pixel.y - 1)].xyz) * GradY[uint2(pixel.x, pixel.y - 1)].xyz;
    }

    return result;
}

[numthreads(16, 16, 1)]
void initSystem(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gResolution)) return;

    // Compute and store diagonal preconditioner
    float3 diagInv = computeDiagonalA(pixel);
    Preconditioner[pixel] = float4(diagInv, 0.0);

    float4 baseSample = Base[pixel];
    float3 b = computeB(pixel);
    float3 x0 = baseSample.xyz;
    float3 Ax = applyWeightedSystem(Base, pixel);
    float3 r = b - Ax;

    // Apply preconditioner to initial residual for direction
    float3 z = applyPreconditioner(pixel, r);

    Solution[pixel] = float4(x0, baseSample.w);
    Residual[pixel] = float4(r, 0.0);
    Direction[pixel] = float4(z, 0.0);  // Use preconditioned residual
    Ap[pixel] = float4(0.0, 0.0, 0.0, 0.0);
    DotBuffer[pixel] = float4(dot(r, z), 0.0, 0.0, 0.0);  // r^T z for preconditioned CG
}

[numthreads(16, 16, 1)]
void applyA(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gResolution)) return;

    float3 p = Direction[pixel].xyz;
    float3 ap = applyWeightedSystem(Direction, pixel);
    Ap[pixel] = float4(ap, 0.0);
    DotBuffer[pixel] = float4(dot(p, ap), 0.0, 0.0, 0.0);
}

[numthreads(16, 16, 1)]
void updateSolutionResidual(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gResolution)) return;

    float4 solutionSample = Solution[pixel];
    float3 direction = Direction[pixel].xyz;
    float3 ap = Ap[pixel].xyz;
    float3 residual = Residual[pixel].xyz;

    float3 newSolution = solutionSample.xyz + gAlpha * direction;
    float3 newResidual = residual - gAlpha * ap;

    // Apply preconditioner for next iteration
    float3 z = applyPreconditioner(pixel, newResidual);

    Solution[pixel] = float4(newSolution, solutionSample.w);
    Residual[pixel] = float4(newResidual, 0.0);
    DotBuffer[pixel] = float4(dot(newResidual, z), 0.0, 0.0, 0.0);  // r^T z for preconditioned CG
}

[numthreads(16, 16, 1)]
void updateDirection(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gResolution)) return;

    float3 residual = Residual[pixel].xyz;
    float3 direction = Direction[pixel].xyz;

    // Apply preconditioner to residual
    float3 z = applyPreconditioner(pixel, residual);

    // Update direction with preconditioned residual: z + gBeta * direction
    Direction[pixel] = float4(mad(direction, gBeta, z), 0.0);
}
