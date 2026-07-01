#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D srcTexture;

layout(push_constant) uniform PushConstants {
    vec2 inv_resolution;
    float threshold;
    float is_first_pass;
} pc;

void main() {
    vec2 texelSize = pc.inv_resolution;
    vec2 uv = fragUV;

    // 13-tap filter
    vec3 a = texture(srcTexture, vec2(uv.x - 2.0 * texelSize.x, uv.y + 2.0 * texelSize.y)).rgb;
    vec3 b = texture(srcTexture, vec2(uv.x,                     uv.y + 2.0 * texelSize.y)).rgb;
    vec3 c = texture(srcTexture, vec2(uv.x + 2.0 * texelSize.x, uv.y + 2.0 * texelSize.y)).rgb;

    vec3 d = texture(srcTexture, vec2(uv.x - 2.0 * texelSize.x, uv.y)).rgb;
    vec3 e = texture(srcTexture, vec2(uv.x,                     uv.y)).rgb;
    vec3 f = texture(srcTexture, vec2(uv.x + 2.0 * texelSize.x, uv.y)).rgb;

    vec3 g = texture(srcTexture, vec2(uv.x - 2.0 * texelSize.x, uv.y - 2.0 * texelSize.y)).rgb;
    vec3 h = texture(srcTexture, vec2(uv.x,                     uv.y - 2.0 * texelSize.y)).rgb;
    vec3 i = texture(srcTexture, vec2(uv.x + 2.0 * texelSize.x, uv.y - 2.0 * texelSize.y)).rgb;

    vec3 j = texture(srcTexture, vec2(uv.x - texelSize.x, uv.y + texelSize.y)).rgb;
    vec3 k = texture(srcTexture, vec2(uv.x + texelSize.x, uv.y + texelSize.y)).rgb;
    vec3 l = texture(srcTexture, vec2(uv.x - texelSize.x, uv.y - texelSize.y)).rgb;
    vec3 m = texture(srcTexture, vec2(uv.x + texelSize.x, uv.y - texelSize.y)).rgb;

    vec3 color = e * 0.125;
    color += (a + c + g + i) * 0.03125;
    color += (b + d + f + h) * 0.0625;
    color += (j + k + l + m) * 0.125;

    if (pc.is_first_pass > 0.5) {
        float brightness = max(color.r, max(color.g, color.b));
        float soft = brightness - pc.threshold;
        soft = clamp(soft, 0.0, 1.0);
        
        if (brightness < pc.threshold) {
            color = vec3(0.0);
        } else {
            // Smoothly ease in the glow over the threshold
            color = color * soft;
        }
    }

    outColor = vec4(color, 1.0);
}
