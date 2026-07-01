#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D srcTexture;

layout(push_constant) uniform PushConstants {
    vec2 inv_resolution;
    float filter_radius;
} pc;

void main() {
    vec2 texelSize = pc.inv_resolution;
    vec2 uv = fragUV;
    
    float x = pc.filter_radius;
    float y = pc.filter_radius;

    // 9-tap filter
    vec3 a = texture(srcTexture, vec2(uv.x - x * texelSize.x, uv.y + y * texelSize.y)).rgb;
    vec3 b = texture(srcTexture, vec2(uv.x,                   uv.y + y * texelSize.y)).rgb;
    vec3 c = texture(srcTexture, vec2(uv.x + x * texelSize.x, uv.y + y * texelSize.y)).rgb;

    vec3 d = texture(srcTexture, vec2(uv.x - x * texelSize.x, uv.y)).rgb;
    vec3 e = texture(srcTexture, vec2(uv.x,                   uv.y)).rgb;
    vec3 f = texture(srcTexture, vec2(uv.x + x * texelSize.x, uv.y)).rgb;

    vec3 g = texture(srcTexture, vec2(uv.x - x * texelSize.x, uv.y - y * texelSize.y)).rgb;
    vec3 h = texture(srcTexture, vec2(uv.x,                   uv.y - y * texelSize.y)).rgb;
    vec3 i = texture(srcTexture, vec2(uv.x + x * texelSize.x, uv.y - y * texelSize.y)).rgb;

    vec3 color = e * 0.25;
    color += (b + d + f + h) * 0.125;
    color += (a + c + g + i) * 0.0625;

    outColor = vec4(color, 1.0);
}
