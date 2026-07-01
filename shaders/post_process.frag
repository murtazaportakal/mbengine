#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D colorSampler;
layout(binding = 1) uniform sampler2D bloomSampler;

// ACES Tonemapping
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
    vec3 hdrColor = texture(colorSampler, fragUV).rgb;
    vec3 bloomColor = texture(bloomSampler, fragUV).rgb;

    // Add bloom
    hdrColor += bloomColor * 1.0; // Bloom intensity multiplier

    // Apply ACES Tonemapping
    vec3 mapped = ACESFilm(hdrColor);
    
    // Apply Gamma Correction (2.2)
    mapped = pow(mapped, vec3(1.0 / 2.2));
    
    outColor = vec4(mapped, 1.0);
}
