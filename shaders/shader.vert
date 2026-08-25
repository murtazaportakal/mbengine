#version 450
#extension GL_ARB_shader_draw_parameters : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct Vertex {
    vec3 pos;
    uint _pad0;
    vec3 normal;
    uint _pad1;
    vec2 uv;
    uvec2 _pad2;
    uvec4 jointIds;
    vec4 jointWeights;
};

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer VertexBuffer {
    Vertex vertices[];
};

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
    mat4 prevViewProj;
    mat4 view;
    mat4 proj;
    mat4 inverseProj;
    mat4 lightSpaceMatrix;
    vec4 cameraPos;
    vec4 lightDir;
    vec4 lightColor;
    vec2 screenSize;
    float zNear;
    float zFar;
    uint numPointLights;
    uint debugMeshlets;
    uvec2 _pad0;
    uint64_t vertexBufferAddr;
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
    
    VertexBuffer vb = VertexBuffer(ubo.vertexBufferAddr);
    Vertex v = vb.vertices[gl_VertexIndex];
    
    vec3 localPos = v.pos;
    vec3 localNormal = v.normal;
    
    if (animInstanceId != 0xFFFFFFFF) {
        float totalWeight = v.jointWeights.x + v.jointWeights.y + v.jointWeights.z + v.jointWeights.w;
        if (totalWeight > 0.0001) {
            uint offset = animInstanceId * 128;
            mat4 skinMat = 
                v.jointWeights.x * boneMatrices[offset + v.jointIds.x] +
                v.jointWeights.y * boneMatrices[offset + v.jointIds.y] +
                v.jointWeights.z * boneMatrices[offset + v.jointIds.z] +
                v.jointWeights.w * boneMatrices[offset + v.jointIds.w];
                
            localPos = (skinMat * vec4(v.pos, 1.0)).xyz;
            localNormal = mat3(skinMat) * v.normal;
        }
    }
    
    vec4 worldPos = inst.world * vec4(localPos, 1.0);
    fragPos = worldPos.xyz;
    
    fragNormal = mat3(inst.world) * localNormal;
    fragUV = v.uv;
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
