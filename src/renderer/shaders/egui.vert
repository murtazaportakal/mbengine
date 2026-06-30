#version 450

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
} push_constants;

void main() {
    // Convert from [0, screen_size] to [-1, 1]
    vec2 pos = in_pos;
    gl_Position = vec4(
        2.0 * pos.x / push_constants.screen_size.x - 1.0,
        2.0 * pos.y / push_constants.screen_size.y - 1.0,
        0.0,
        1.0
    );
    out_uv = in_uv;
    
    // Egui color is sRGB, but for simplicity we just pass it as linear-ish if we aren't doing strict color management
    // Actually Egui passes linear colors to the vertex shader
    out_color = in_color;
}
