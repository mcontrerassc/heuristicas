@echo off
setlocal

REM Recorre X desde 1 hasta 10
for /L %%X in (1,1,10) do (
  echo Graficando X=%%X...
  python plot_route.py %%X
  if errorlevel 1 (
    echo [ERROR] El comando fallo para X=%%X
    exit /b %errorlevel%
  )
)

endlocal
