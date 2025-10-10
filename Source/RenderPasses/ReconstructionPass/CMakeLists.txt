add_plugin(ReconstructionPass)

target_sources(ReconstructionPass PRIVATE
    ReconstructionPass.cpp
    ReconstructionPass.h
    ReconstructionPass.slang
)

target_copy_shaders(ReconstructionPass RenderPasses/ReconstructionPass)

target_source_group(ReconstructionPass "RenderPasses")
