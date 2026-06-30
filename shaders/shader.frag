#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D texSampler;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 lightDir;
    vec4 lightColor;
} pc;

void main() {
    // Sample texture
    vec4 texColor = texture(texSampler, fragUV);
    vec3 objectColor = texColor.rgb;
    
    // Normalize normal and light direction
    vec3 norm = normalize(fragNormal);
    
    // lightDir is defined as the direction TO the light, or direction OF the light.
    // If it's the direction OF the light, we need to invert it for the dot product.
    // Let's assume lightDir is the vector pointing TO the light source.
    vec3 lightDir = normalize(pc.lightDir.xyz);
    
    // Diffuse impact
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * pc.lightColor.rgb;
    
    // Ambient light
    vec3 ambient = vec3(0.2);
    
    // Final composite
    vec3 result = (ambient + diffuse) * objectColor;
    
    outColor = vec4(result, 1.0);
}
