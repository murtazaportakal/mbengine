#version 450

layout(location = 0) out vec2 fragUV;

void main() {
    // Generate a full screen triangle based on gl_VertexIndex
    fragUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(fragUV * 2.0f - 1.0f, 0.0f, 1.0f);
}
