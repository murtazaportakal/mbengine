#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform GlobalUbo {
    mat4 viewProj;
    vec4 lightDir;
    vec4 lightColor;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;

void main() {
    // Sample texture
    vec4 texColor = texture(texSampler, fragUV);
    vec3 objectColor = texColor.rgb;
    
    // Normalize normal and light direction
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(ubo.lightDir.xyz);
    
    // Diffuse impact
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * ubo.lightColor.rgb;
    
    // Ambient light
    vec3 ambient = vec3(0.2);
    
    // Final composite
    vec3 result = (ambient + diffuse) * objectColor;
    
    outColor = vec4(result, 1.0);
}
