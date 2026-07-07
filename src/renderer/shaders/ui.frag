#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform sampler2D tex;

void main() {
    vec4 tex_color = texture(tex, in_uv);
    out_color = in_color * tex_color;
}
