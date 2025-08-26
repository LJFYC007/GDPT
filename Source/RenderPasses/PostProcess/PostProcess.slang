cbuffer PerFrameCB
{
    float gSigma;
    int gKernelWidth;
    uint2 gResolution;
};

Texture2D Input;
RWTexture2D<float4> Output;

float gaussian(float x, float sigma)
{
    return exp(-x * x / (2.0 * sigma * sigma));
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 coord = dispatchThreadId.xy;
    if (coord.x >= gResolution.x || coord.y >= gResolution.y) return;

    int halfWidth = gKernelWidth / 2;
    float4 color = float4(0, 0, 0, 0);
    float weightSum = 0.0;

    for (int dy = -halfWidth; dy <= halfWidth; ++dy)
    {
        for (int dx = -halfWidth; dx <= halfWidth; ++dx)
        {
            int2 sampleCoord = int2(coord) + int2(dx, dy);
            sampleCoord = clamp(sampleCoord, int2(0,0), int2(gResolution)-1);
            float w = gaussian(dx, gSigma) * gaussian(dy, gSigma);
            color += Input[sampleCoord] * w;
            weightSum += w;
        }
    }
    Output[coord] = color / max(weightSum, 1e-6);
}
