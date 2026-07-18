#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in uvec4 inJointIds;     // Bone indices (unused in vertex shader — skinning done in compute)
layout(location = 4) in vec4 inJointWeights;  // Bone weights (unused in vertex shader — skinning done in compute)

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragPos;
layout(location = 3) out vec4 fragPosLightSpace;

struct PointLight {
    vec4 position;
    vec4 color;
};

layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 viewProj;
    mat4 lightSpaceMatrix;
    vec4 cameraPos;
    vec4 lightDir;
    vec4 lightColor;
    PointLight pointLights[4];
    uint numPointLights;
    uvec3 _padding;
} ubo;

struct InstanceData {
    mat4 world;
    vec4 aabbMin;
    vec4 aabbMax;
    vec4 color;
    vec4 pbr;
    uvec4 geometry;
    uvec4 geometry2;
};

layout(set = 0, binding = 3) readonly buffer InstanceBuffer {
    InstanceData instances[];
};

layout(location = 4) out vec4 fragColor;
layout(location = 5) out float fragMetallic;
layout(location = 6) out float fragRoughness;
layout(location = 7) flat out uint fragTextureIndex;
layout(location = 8) flat out uint fragNormalTextureIndex;
layout(location = 9) flat out uint fragMRTextureIndex;
layout(location = 10) flat out uint fragEmissiveTextureIndex;

void main() {
    InstanceData inst = instances[gl_InstanceIndex];
    vec4 worldPos = inst.world * vec4(inPosition, 1.0);
    fragPos = worldPos.xyz;
    
    // Transform normal to world space. 
    // In a real engine, we'd use inverse(transpose(mat3(inst.world))) if scale is non-uniform.
    fragNormal = mat3(inst.world) * inNormal;
    fragUV = inUV;
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;

    fragColor = vec4(inst.color.rgb, 1.0);
    fragMetallic = inst.pbr.x;
    fragRoughness = inst.pbr.y;
    fragTextureIndex = floatBitsToUint(inst.color.a);
    fragNormalTextureIndex = floatBitsToUint(inst.pbr.z);
    fragMRTextureIndex = floatBitsToUint(inst.pbr.w);
    fragEmissiveTextureIndex = inst.geometry.w;

    gl_Position = ubo.viewProj * worldPos;
}
