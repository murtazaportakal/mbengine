#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D inputTexture;

layout(push_constant) uniform PushConstants {
    vec2 direction; // (1, 0) for horizontal, (0, 1) for vertical
    float radius; // ignored in this optimized 5-tap linear sampling, but kept for alignment
    float padding;
} pc;

void main() {
    vec2 tex_offset = 1.0 / textureSize(inputTexture, 0); // gets size of single texel
    vec3 result = texture(inputTexture, inUV).rgb * 0.227027; // current fragment's contribution
    
    vec2 offset1 = tex_offset * pc.direction * 1.3846153846;
    vec2 offset2 = tex_offset * pc.direction * 3.2307692308;
    
    result += texture(inputTexture, inUV + offset1).rgb * 0.3162162162;
    result += texture(inputTexture, inUV - offset1).rgb * 0.3162162162;
    
    result += texture(inputTexture, inUV + offset2).rgb * 0.0702702703;
    result += texture(inputTexture, inUV - offset2).rgb * 0.0702702703;

    outColor = vec4(result, 1.0);
}
