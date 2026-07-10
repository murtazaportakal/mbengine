#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in uvec4 inJointIds;
layout(location = 4) in vec4 inJointWeights;

layout(location = 0) out vec3 fragNormal;

layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 viewProj;
    mat4 lightSpaceMatrix;
    vec4 cameraPos;
    vec4 lightDir;
    vec4 lightColor;
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
};

layout(set = 0, binding = 3) readonly buffer InstanceBuffer {
    InstanceData instances[];
};

void main() {
    InstanceData instance = instances[gl_InstanceIndex];
    mat4 world = instance.world;
    
    vec4 worldPos = world * vec4(inPosition, 1.0);
    gl_Position = ubo.viewProj * worldPos;

    mat3 normalMatrix = transpose(inverse(mat3(world)));
    fragNormal = normalize(normalMatrix * inNormal);
}
