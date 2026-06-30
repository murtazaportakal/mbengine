#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragUV;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 lightDir;
    vec4 lightColor;
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    // Since we don't have a separate normal matrix yet (no non-uniform scaling/rotations implemented on normals),
    // we just pass the object space normal for now to get basic lighting working.
    // In a real engine, we'd pass the world-space normal: mat3(model) * inNormal.
    fragNormal = inNormal;
    fragUV = inUV;
}
