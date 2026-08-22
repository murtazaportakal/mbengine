#version 450
#extension GL_ARB_shader_draw_parameters : require

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

layout(set = 0, binding = 7) readonly buffer AnimBoneMatrices {
    mat4 boneMatrices[];
};

layout(location = 4) out vec4 fragColor;
layout(location = 5) out float fragMetallic;
layout(location = 6) out float fragRoughness;
layout(location = 7) flat out uint fragTextureIndex;
layout(location = 8) flat out uint fragNormalTextureIndex;
layout(location = 9) flat out uint fragMRTextureIndex;
layout(location = 10) flat out uint fragEmissiveTextureIndex;
layout(location = 11) flat out vec3 fragMeshletColor;

void main() {
    InstanceData inst = instances[gl_InstanceIndex];
    uint animInstanceId = inst.geometry2.z;
    
    vec3 localPos = inPosition;
    vec3 localNormal = inNormal;
    
    if (animInstanceId != 0xFFFFFFFF) {
        float totalWeight = inJointWeights.x + inJointWeights.y + inJointWeights.z + inJointWeights.w;
        if (totalWeight > 0.0001) {
            uint offset = animInstanceId * 128;
            mat4 skinMat = 
                inJointWeights.x * boneMatrices[offset + inJointIds.x] +
                inJointWeights.y * boneMatrices[offset + inJointIds.y] +
                inJointWeights.z * boneMatrices[offset + inJointIds.z] +
                inJointWeights.w * boneMatrices[offset + inJointIds.w];
                
            localPos = (skinMat * vec4(inPosition, 1.0)).xyz;
            localNormal = mat3(skinMat) * inNormal;
        }
    }
    
    vec4 worldPos = inst.world * vec4(localPos, 1.0);
    fragPos = worldPos.xyz;
    
    fragNormal = mat3(inst.world) * localNormal;
    fragUV = inUV;
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;

    fragColor = vec4(inst.color.rgb, 1.0);
    fragMetallic = inst.pbr.x;
    fragRoughness = inst.pbr.y;
    fragTextureIndex = uint(inst.color.w);
    fragNormalTextureIndex = uint(inst.pbr.z);
    fragMRTextureIndex = uint(inst.pbr.w);
    fragEmissiveTextureIndex = inst.geometry.w;

    uint id = gl_DrawIDARB;
    float r = float((id * 137 + 59) % 256) / 255.0;
    float g = float((id * 73 + 17) % 256) / 255.0;
    float b = float((id * 251 + 101) % 256) / 255.0;
    fragMeshletColor = vec3(r, g, b);

    gl_Position = ubo.viewProj * worldPos;
}
