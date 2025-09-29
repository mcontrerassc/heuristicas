@echo off
setlocal

REM Recorre X desde 1 hasta 10
for /L %%X in (1,1,10) do (
  echo Ejecutando X=%%X...
  python main.py grafo.csv instancia%%X.csv
  if errorlevel 1 (
    echo [ERROR] El comando fallo para X=%%X
    exit /b %errorlevel%
  )
)

endlocal
