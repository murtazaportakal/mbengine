#version 450
layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D depthSampler;
void main() {
    float d = texture(depthSampler, fragUV).r;
    outColor = vec4(d, 0.0, 0.0, 1.0);
}
