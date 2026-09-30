<#
Prepara o ambiente Python do projeto e instala as dependências.

Uso (PowerShell, na raiz do projeto):
    powershell -ExecutionPolicy Bypass -File setup.ps1                     # CPU / NPU / GPU Intel (.venv)
    powershell -ExecutionPolicy Bypass -File setup.ps1 -Alvo cuda          # GPU NVIDIA (.venv)
    powershell -ExecutionPolicy Bypass -File setup.ps1 -Alvo quantizacao   # gerar modelo INT8 (.venv-quant)

Nos alvos intel/cuda, ao final baixa os modelos e verifica os dispositivos.
#>
param(
    [ValidateSet("intel", "cuda", "quantizacao")]
    [string]$Alvo = "intel"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked {
    param([string]$Descricao, [scriptblock]$Comando)
    Write-Host "`n>> $Descricao" -ForegroundColor Cyan
    & $Comando
    if ($LASTEXITCODE -ne 0) { throw "Falhou: $Descricao (código $LASTEXITCODE)" }
}

# A quantização usa um ambiente separado (o nncf exige outra versão do numpy)
$venv = if ($Alvo -eq "quantizacao") { ".venv-quant" } else { ".venv" }

# 1. Python 3.12 (versão usada no projeto)
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        Invoke-Checked "Criando $venv com Python 3.12" { py -3.12 -m venv $venv }
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        Invoke-Checked "Criando $venv com o python do PATH" { python -m venv $venv }
    } else {
        throw "Python não encontrado. Instale com: winget install Python.Python.3.12"
    }
}
$py = Join-Path $PSScriptRoot "$venv\Scripts\python.exe"

Invoke-Checked "Atualizando pip" { & $py -m pip install --upgrade pip }

# 2. Dependências
if ($Alvo -eq "quantizacao") {
    Invoke-Checked "Instalando requirements-quantizacao.txt" { & $py -m pip install -r requirements-quantizacao.txt }
    Write-Host "`nAmbiente de quantização pronto. Para gerar o modelo INT8:" -ForegroundColor Green
    Write-Host "  $venv\Scripts\python.exe teste_desempenho\quantizar_int8.py"
    exit 0
} elseif ($Alvo -eq "cuda") {
    # Remove o onnxruntime de CPU caso já exista (conflita com onnxruntime-gpu)
    Invoke-Checked "Removendo onnxruntime de CPU (se houver)" { & $py -m pip uninstall -y onnxruntime }
    Invoke-Checked "Instalando requirements-cuda.txt" { & $py -m pip install -r requirements-cuda.txt }
    Invoke-Checked "Instalando rtmlib sem dependências" { & $py -m pip install --no-deps rtmlib==0.0.16 }
} else {
    Invoke-Checked "Instalando requirements.txt" { & $py -m pip install -r requirements.txt }
}

# 3. Modelos ONNX para o cache local (o download não entra no tempo medido)
Invoke-Checked "Baixando modelos" { & $py teste_desempenho\run_benchmark.py --baixar-modelos }

# 4. Verificação (não aborta o setup: é normal faltar algum dispositivo)
Write-Host "`n>> Verificando dispositivos disponíveis" -ForegroundColor Cyan
& $py teste_desempenho\run_benchmark.py --verificar

Write-Host "`nAmbiente pronto. Para rodar o teste de desempenho:" -ForegroundColor Green
Write-Host "  .venv\Scripts\python.exe teste_desempenho\run_benchmark.py"
