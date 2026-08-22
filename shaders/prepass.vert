#version 450
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

struct PointLight {
    vec4 position;
    vec4 color;
};

layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 viewProj;
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
};

layout(set = 0, binding = 3) readonly buffer InstanceBuffer {
    InstanceData instances[];
};

layout(location = 0) out vec3 fragNormal;

void main() {
    InstanceData instance = instances[gl_InstanceIndex];
    mat4 world = instance.world;
    
    VertexBuffer vb = VertexBuffer(ubo.vertexBufferAddr);
    Vertex v = vb.vertices[gl_VertexIndex];
    
    vec4 worldPos = world * vec4(v.pos, 1.0);
    gl_Position = ubo.viewProj * worldPos;

    mat3 normalMatrix = transpose(inverse(mat3(world)));
    fragNormal = normalize(normalMatrix * v.normal);
}
