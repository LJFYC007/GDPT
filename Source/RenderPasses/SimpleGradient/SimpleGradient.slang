Texture2D<float4> base;
Texture2D<float4> input1;
Texture2D<float4> input2;
Texture2D<float4> input3;
Texture2D<float4> input4;
RWTexture2D<float4> outputX;
RWTexture2D<float4> outputY;

cbuffer PerFrameCB
{
    uint2 gResolution;
}

[numthreads(32, 32, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint index = dispatchThreadID.y * gResolution.x + dispatchThreadID.x;
    if (dispatchThreadID.x >= gResolution.x && dispatchThreadID.y >= gResolution.y)
        return;

    uint x = dispatchThreadID.x;
    uint y = dispatchThreadID.y;
    if (x >= 1)
        outputX[uint2(x - 1, y)] += 0.5 * (base[uint2(x, y)] - base[uint2(x - 1, y)]);
    if (y >= 1)
        outputY[uint2(x, y - 1)] += 0.5 * (base[uint2(x, y)] - base[uint2(x, y - 1)]);
    if (x + 1 < gResolution.x)
        outputX[uint2(x, y)] += 0.5 * (base[uint2(x + 1, y)] - base[uint2(x, y)]);
    if (y + 1 < gResolution.y)
        outputY[uint2(x, y)] += 0.5 * (base[uint2(x, y + 1)] - base[uint2(x, y)]);
}
