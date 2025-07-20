Texture2D<float4> base;
Texture2D<float4> inputX;
Texture2D<float4> inputY;
RWTexture2D<float4> output;

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
    float4 value = base[uint2(x, y)];
    uint num = 1;
    if (x >= 1)
    {
        value += base[uint2(x - 1, y)] + inputX[uint2(x - 1, y)];
        num++;
    }
    if (y >= 1)
    {
        value += base[uint2(x, y - 1)] + inputY[uint2(x, y - 1)];
        num++;
    }
    if (x + 1 < gResolution.x)
    {
        value += base[uint2(x + 1, y)] - inputX[uint2(x, y)];
        num++;
    }
    if (y + 1 < gResolution.y)
    {
        value += base[uint2(x, y + 1)] - inputY[uint2(x, y)];
        num++;
    }
    output[uint2(x, y)] = value / float(num);
}
