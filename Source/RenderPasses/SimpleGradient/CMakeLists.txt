add_plugin(SimpleGradient)

target_sources(SimpleGradient PRIVATE
    SimpleGradient.cpp
    SimpleGradient.h
    SimpleGradient.slang
    Clear.slang
)

target_copy_shaders(SimpleGradient RenderPasses/SimpleGradient)

target_source_group(SimpleGradient "RenderPasses")
