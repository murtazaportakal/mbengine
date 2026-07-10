@echo off
echo Compiling Shaders...

for %%f in (*.vert) do (
    glslc "%%f" -o "%%~nf_vert.spv"
    IF !ERRORLEVEL! NEQ 0 (
        echo Failed to compile %%f
        exit /b 1
    )
)

for %%f in (*.frag) do (
    glslc "%%f" -o "%%~nf_frag.spv"
    IF !ERRORLEVEL! NEQ 0 (
        echo Failed to compile %%f
        exit /b 1
    )
)

glslc bloom_downsample.frag -o bloom_downsample_frag.spv
glslc bloom_upsample.frag -o bloom_upsample_frag.spv
glslc ssao.frag -o ssao_frag.spv

for %%f in (*.comp) do (
    glslc "%%f" -o "%%~nf.spv"
    IF !ERRORLEVEL! NEQ 0 (
        echo Failed to compile %%f
        exit /b 1
    )
)

:: Backwards compatibility for hardcoded names in engine
copy shader_vert.spv vert.spv > nul
copy shader_frag.spv frag.spv > nul
copy shadow_vert.spv shadow.spv > nul
:: Note: shadow_vert is compiled to shadow.spv manually for backwards compat

echo Success!
pause
