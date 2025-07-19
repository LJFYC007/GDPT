Texture2D<float4> input;
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
    output[uint2(x, y)] = input[uint2(x, y)];
}
