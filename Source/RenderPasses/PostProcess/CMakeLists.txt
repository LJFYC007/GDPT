add_plugin(PostProcess)

target_sources(PostProcess PRIVATE
    PostProcess.cpp
    PostProcess.h
    PostProcess.slang
)

target_copy_shaders(PostProcess RenderPasses/PostProcess)

target_source_group(PostProcess "RenderPasses")
