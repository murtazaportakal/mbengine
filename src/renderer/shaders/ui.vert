#version 450

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
} pc;

void main() {
    out_uv = in_uv;
    out_color = in_color;
    
    // Convert from pixel space to normalized device coordinates
    vec2 pos = (in_pos / pc.screen_size) * 2.0 - 1.0;
    // Vulkan Y is down, so we keep pos.y as is for window coordinates, wait, window Y down?
    // Usually NDC Y is down in Vulkan, so if in_pos.y is 0 (top), it should map to -1.
    // If in_pos.y is screen_size.y, it should map to +1.
    // (in_pos / screen_size) * 2.0 - 1.0 does exactly this!
    gl_Position = vec4(pos, 0.0, 1.0);
}
