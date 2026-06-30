#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D font_texture;

void main() {
    // Egui font texture stores coverage in the R channel, but egui also provides color in the vertex color.
    // Actually, in egui 0.27+, the font texture is a typical RGBA texture (or R depending on FontImage config).
    // Let's sample as RGBA.
    vec4 tex_color = texture(font_texture, in_uv);
    out_color = in_color * tex_color;
}
