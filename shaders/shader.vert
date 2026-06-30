#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragPos;
layout(location = 3) out vec4 fragPosLightSpace;

struct PointLight {
    vec4 position;
    vec4 color;
};

layout(binding = 0) uniform GlobalUbo {
    mat4 viewProj;
    mat4 lightSpaceMatrix;
    vec4 cameraPos;
    vec4 lightDir;
    vec4 lightColor;
    PointLight pointLights[4];
    uint numPointLights;
    uvec3 _padding;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 world;
    float metallic;
    float roughness;
    vec2 _padding;
} pc;

void main() {
    vec4 worldPos = pc.world * vec4(inPosition, 1.0);
    fragPos = worldPos.xyz;
    
    // Transform normal to world space. 
    // In a real engine, we'd use inverse(transpose(mat3(pc.world))) if scale is non-uniform.
    fragNormal = mat3(pc.world) * inNormal;
    fragUV = inUV;
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;

    gl_Position = ubo.viewProj * worldPos;
}
