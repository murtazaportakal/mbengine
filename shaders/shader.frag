#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragPos;
layout(location = 3) in vec4 fragPosLightSpace;
layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position;
    vec4 color;
};

layout(binding = 0) uniform GlobalUbo {
    mat4 viewProj;
    mat4 lightSpaceMatrix;
    vec4 cameraPos;
    vec4 lightDir;
    vec4 lightColor;
    PointLight pointLights[4];
    uint numPointLights;
    uvec3 _padding;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 world;
    float metallic;
    float roughness;
    vec2 _padding;
} pc;

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D envSampler;
layout(binding = 3) uniform sampler2D shadowMap;

vec2 sampleEquirectangular(vec3 v) {
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= vec2(0.1591, 0.3183);
    uv += 0.5;
    return uv;
}

// ACES tonemapping moved to post_process.frag

float ShadowCalculation(vec4 fragPosLightSpace, vec3 N, vec3 L) {
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range (Vulkan Z is already [0,1], XY is [-1,1])
    projCoords.xy = projCoords.xy * 0.5 + 0.5;
    
    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if(projCoords.z > 1.0)
        return 0.0;
        
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    float currentDepth = projCoords.z;
    
    // calculate bias (based on depth map resolution and slope)
    float bias = max(0.005 * (1.0 - dot(N, L)), 0.0005);
    
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth  ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    
    return shadow;
}

const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / max(denom, 0.0000001);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    vec4 texColor = texture(texSampler, fragUV);
    vec3 albedo = pow(texColor.rgb, vec3(2.2));

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(ubo.cameraPos.xyz - fragPos);

    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, pc.metallic);
    
    vec3 Lo = vec3(0.0);
    
    // Directional Light
    {
        vec3 L = normalize(ubo.lightDir.xyz);
        vec3 H = normalize(V + L);
        vec3 radiance = ubo.lightColor.rgb;
        
        float NDF = DistributionGGX(N, H, pc.roughness);   
        float G   = GeometrySmith(N, V, L, pc.roughness);    
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular     = numerator / denominator;
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - pc.metallic;
        
        float NdotL = max(dot(N, L), 0.0);
        float shadow = ShadowCalculation(fragPosLightSpace, N, L);
        Lo += (1.0 - shadow) * (kD * albedo / PI + specular) * radiance * NdotL;
    }
    
    // Point Lights
    for (uint i = 0; i < ubo.numPointLights; i++) {
        vec3 L = normalize(ubo.pointLights[i].position.xyz - fragPos);
        vec3 H = normalize(V + L);
        
        float distance = length(ubo.pointLights[i].position.xyz - fragPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = ubo.pointLights[i].color.rgb * ubo.pointLights[i].color.w * attenuation;
        
        float NDF = DistributionGGX(N, H, pc.roughness);   
        float G   = GeometrySmith(N, V, L, pc.roughness);    
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular     = numerator / denominator;
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - pc.metallic;
        
        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }
    
    // Ambient IBL
    vec3 R = reflect(-V, N);
    
    // Approximate Irradiance by sampling the highest LOD (most blurry)
    vec2 irradianceUV = sampleEquirectangular(N);
    vec3 irradiance = textureLod(envSampler, irradianceUV, 10.0).rgb; // Hardcoded high LOD
    vec3 diffuseIBL = irradiance * albedo;
    
    // Approximate Prefiltered Specular
    const float MAX_REFLECTION_LOD = 8.0;
    vec2 prefilteredUV = sampleEquirectangular(R);
    vec3 prefilteredColor = textureLod(envSampler, prefilteredUV, pc.roughness * MAX_REFLECTION_LOD).rgb;
    
    // Approximate BRDF LUT
    vec2 brdfApprox = vec2(F0.x, 1.0 - pc.roughness); // simplistic approximation
    vec3 specularIBL = prefilteredColor * (F0 * brdfApprox.x + brdfApprox.y);
    
    vec3 kS_ambient = fresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 kD_ambient = vec3(1.0) - kS_ambient;
    kD_ambient *= 1.0 - pc.metallic;
    
    vec3 ambient = (kD_ambient * diffuseIBL) + specularIBL;
    vec3 color = ambient + Lo;
    
    outColor = vec4(color, texColor.a);
}
