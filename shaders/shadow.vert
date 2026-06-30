#version 450

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform PushConstants {
    mat4 lightSpaceMatrix; // Projection * View of the light
    mat4 modelMatrix;
} pc;

void main() {
    gl_Position = pc.lightSpaceMatrix * pc.modelMatrix * vec4(inPosition, 1.0);
}
