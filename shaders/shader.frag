#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragPos;
layout(location = 3) in vec4 fragPosLightSpace;
layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position;
    vec4 color;
};

layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 viewProj;
    mat4 prevViewProj;
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
    uvec2 _padding;
} ubo;

layout(std430, set = 0, binding = 4) readonly buffer PointLightBuffer {
    PointLight lights[];
} pointLights;

layout(std430, set = 0, binding = 5) readonly buffer LightGridBuffer {
    uvec2 lightGrid[]; // x = offset, y = count
};

layout(std430, set = 0, binding = 6) readonly buffer LightIndexBuffer {
    uint lightIndices[];
};

layout(location = 4) in vec4 fragColor;
layout(location = 5) in float fragMetallic;
layout(location = 6) in float fragRoughness;
layout(location = 7) flat in uint fragTextureIndex;
layout(location = 8) flat in uint fragNormalTextureIndex;
layout(location = 9) flat in uint fragMRTextureIndex;
layout(location = 10) flat in uint fragEmissiveTextureIndex;
layout(location = 11) flat in vec3 fragMeshletColor;

#extension GL_EXT_nonuniform_qualifier : enable
layout(set = 1, binding = 0) uniform sampler2D textures[];
layout(set = 0, binding = 1) uniform sampler2D envSampler;
layout(set = 0, binding = 2) uniform sampler2D shadowMap;

vec2 sampleEquirectangular(vec3 v) {
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= vec2(0.1591, 0.3183);
    uv += 0.5;
    return uv;
}

mat3 computeTBN(vec3 normal, vec3 pos, vec2 uv) {
    vec3 dp1 = dFdx(pos);
    vec3 dp2 = dFdy(pos);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    
    vec3 dp2perp = cross(dp2, normal);
    vec3 dp1perp = cross(normal, dp1);
    
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
    return mat3(T * invmax, B * invmax, normal);
}

float ShadowCalculation(vec4 fragPosLightSpace, vec3 N, vec3 L) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords.xy = projCoords.xy * 0.5 + 0.5;
    
    if(projCoords.z > 1.0)
        return 0.0;
        
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    float currentDepth = projCoords.z;
    float bias = max(0.005 * (1.0 - dot(N, L)), 0.0005);
    
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
    vec4 texColor = texture(textures[nonuniformEXT(fragTextureIndex)], fragUV);
    vec3 baseColor = pow(fragColor.rgb, vec3(2.2));
    vec3 albedo = pow(texColor.rgb, vec3(2.2)) * baseColor;
    
    float metallic = fragMetallic;
    float roughness = fragRoughness;
    if (fragMRTextureIndex != 0) {
        vec4 mrSample = texture(textures[nonuniformEXT(fragMRTextureIndex)], fragUV);
        // GLTF specifies: B is metallic, G is roughness
        roughness = mrSample.g * fragRoughness;
        metallic = mrSample.b * fragMetallic;
    }

    vec3 N = normalize(fragNormal);
    if (!gl_FrontFacing) {
        N = -N;
    }
    
    if (fragNormalTextureIndex != 0) {
        mat3 tbn = computeTBN(N, fragPos, fragUV);
        vec3 normalSample = texture(textures[nonuniformEXT(fragNormalTextureIndex)], fragUV).rgb;
        normalSample = normalize(normalSample * 2.0 - 1.0);
        N = normalize(tbn * normalSample);
    }
    
    vec3 V = normalize(ubo.cameraPos.xyz - fragPos);

    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);
    
    vec3 Lo = vec3(0.0);
    
    // Directional Light
    {
        vec3 L = normalize(-ubo.lightDir.xyz);
        vec3 H = normalize(V + L);
        vec3 radiance = ubo.lightColor.rgb;
        
        float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);    
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular     = numerator / denominator;
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        
        float NdotL = max(dot(N, L), 0.0);
        float shadow = ShadowCalculation(fragPosLightSpace, N, L);
        Lo += (1.0 - shadow) * (kD * albedo / PI + specular) * radiance * NdotL;
    }
    
    // Point Lights via Forward+ Clustering
    vec4 viewPos = ubo.view * vec4(fragPos, 1.0);
    float viewZ = -viewPos.z;
    
    uint zTile = uint(max(log2(viewZ / ubo.zNear) * 24.0 / log2(ubo.zFar / ubo.zNear), 0.0));
    zTile = min(zTile, 23); // clamp to max 24 slices
    
    uvec2 tile = uvec2(gl_FragCoord.xy / vec2(ubo.screenSize.x / 16.0, ubo.screenSize.y / 9.0));
    tile.x = min(tile.x, 15);
    tile.y = min(tile.y, 8);
    
    uint clusterIndex = tile.x + (tile.y * 16) + (zTile * 16 * 9);
    
    uvec2 gridData = lightGrid[clusterIndex];
    uint offset = gridData.x;
    uint count = gridData.y;
    
    for (uint i = 0; i < count; i++) {
        uint lightIdx = lightIndices[offset + i];
        PointLight light = pointLights.lights[lightIdx];
        
        vec3 L = normalize(light.position.xyz - fragPos);
        vec3 H = normalize(V + L);
        
        float distance = length(light.position.xyz - fragPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = light.color.rgb * light.color.w * attenuation;
        
        float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);    
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular     = numerator / denominator;
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        
        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }
    
    // Ambient IBL
    vec3 R = reflect(-V, N);
    
    vec2 irradianceUV = sampleEquirectangular(N);
    vec3 irradiance = textureLod(envSampler, irradianceUV, 10.0).rgb;
    irradiance = pow(irradiance, vec3(2.2)) * 2.0;
    vec3 diffuseIBL = irradiance * albedo;
    
    const float MAX_REFLECTION_LOD = 8.0;
    vec2 prefilteredUV = sampleEquirectangular(R);
    vec3 prefilteredColor = textureLod(envSampler, prefilteredUV, roughness * MAX_REFLECTION_LOD).rgb;
    prefilteredColor = pow(prefilteredColor, vec3(2.2)) * 2.0;
    
    vec2 brdfApprox = vec2(F0.x, 1.0 - roughness);
    vec3 specularIBL = prefilteredColor * (F0 * brdfApprox.x + brdfApprox.y);
    
    vec3 kS_ambient = fresnelSchlick(max(dot(N, V), 0.0), F0);
    vec3 kD_ambient = vec3(1.0) - kS_ambient;
    kD_ambient *= 1.0 - metallic;

    // SSAO pass not yet implemented — use full ambient
    float ao = 1.0;
    
    // Fake IBL since we don't have an environment map yet
    vec3 ambient = vec3(0.03) * albedo * ao;
    if (metallic > 0.0) {
        ambient += albedo * metallic * 0.15 * ao;
    }
    
    vec3 emissive = vec3(0.0);
    if (fragEmissiveTextureIndex != 0) {
        emissive = pow(texture(textures[nonuniformEXT(fragEmissiveTextureIndex)], fragUV).rgb, vec3(2.2));
    }
    
    vec3 color = ambient + Lo + emissive;
    
    if (ubo.debugMeshlets > 0) {
        color = fragMeshletColor;
    }
    
    outColor = vec4(color, 1.0);
}
