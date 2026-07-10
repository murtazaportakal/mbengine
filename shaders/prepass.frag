#version 450

layout(location = 0) in vec3 fragNormal;

layout(location = 0) out vec4 outNormal;

void main() {
    outNormal = vec4(normalize(fragNormal), 1.0);
}
