@echo off
echo Compiling Shaders...

glslc shader.vert -o vert.spv
IF %ERRORLEVEL% NEQ 0 (
  echo Failed to compile shader.vert
  exit /b %ERRORLEVEL%
)

glslc shader.frag -o frag.spv
IF %ERRORLEVEL% NEQ 0 (
  echo Failed to compile shader.frag
  exit /b %ERRORLEVEL%
)

echo Success!
pause
