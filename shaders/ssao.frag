#version 450

layout(location = 0) in vec2 fragUV;

layout(set = 0, binding = 0) uniform sampler2D depthMap;
layout(set = 0, binding = 1) uniform sampler2D normalMap;
layout(set = 0, binding = 2) uniform sampler2D noiseMap;

layout(set = 0, binding = 3) uniform SsaoUbo {
    mat4 viewProj;
    mat4 invViewProj;
    vec4 samples[64];
    vec2 resolution;
    float radius;
    float bias;
} ubo;

layout(location = 0) out float outOcclusion;

vec3 getPosition(vec2 uv) {
    float depth = texture(depthMap, uv).r;
    vec4 clipSpace = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 worldSpace = ubo.invViewProj * clipSpace;
    return worldSpace.xyz / worldSpace.w;
}

void main() {
    float depth = texture(depthMap, fragUV).r;
    if (depth >= 1.0) {
        outOcclusion = 1.0;
        return;
    }

    vec3 fragPos = getPosition(fragUV);
    vec3 normal = normalize(texture(normalMap, fragUV).xyz);

    vec2 noiseScale = ubo.resolution / 4.0;
    vec3 randomVec = normalize(texture(noiseMap, fragUV * noiseScale).xyz * 2.0 - 1.0);

    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 tbn = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    int kernelSize = 32;

    for (int i = 0; i < kernelSize; ++i) {
        vec3 samplePos = tbn * ubo.samples[i].xyz; 
        samplePos = fragPos + samplePos * ubo.radius;

        vec4 offset = vec4(samplePos, 1.0);
        offset = ubo.viewProj * offset;
        offset.xyz /= offset.w;
        offset.xy = offset.xy * 0.5 + 0.5;

        float sampleDepth = texture(depthMap, offset.xy).r;
        vec4 sampleClip = vec4(offset.xy * 2.0 - 1.0, sampleDepth, 1.0);
        vec4 sampleWorld = ubo.invViewProj * sampleClip;
        vec3 sampleWorldPos = sampleWorld.xyz / sampleWorld.w;
        
        // Depth test: if the actual depth is closer to the camera than our sample point depth
        // We use viewProj depth direction. Usually closer means smaller depth in Vulkan?
        // Wait, world space distance is better.
        float rangeCheck = smoothstep(0.0, 1.0, ubo.radius / length(fragPos - sampleWorldPos));
        
        // Let's use projection depth directly.
        // In Vulkan depth goes 0 to 1 (0 is near, 1 is far).
        // A smaller depth value means it's closer to camera.
        // We occlusion if sampleDepth is closer than our hemisphere sample offset.z
        if (sampleDepth < offset.z - ubo.bias) {
            occlusion += 1.0 * rangeCheck;
        }
    }

    outOcclusion = 1.0 - (occlusion / float(kernelSize));
}
