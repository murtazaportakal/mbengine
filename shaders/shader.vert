#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragUV;

layout(binding = 0) uniform GlobalUbo {
    mat4 viewProj;
    vec4 lightDir;
    vec4 lightColor;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 world;
} pc;

void main() {
    gl_Position = ubo.viewProj * pc.world * vec4(inPosition, 1.0);
    // Transform normal to world space. 
    // Assumes uniform scaling! (If non-uniform, use inverse transpose)
    fragNormal = mat3(pc.world) * inNormal;
    fragUV = inUV;
}
