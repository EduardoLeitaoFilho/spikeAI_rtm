## Instalação

### Ambiente testado
| Item | Versão |
|---|---|
| SO | Windows 11 |
| Python | 3.13.1 |
| rtmlib | 0.0.16 |
| onnxruntime | 1.30.0 (CPU) |
| opencv-python | 5.0.0.93 |
| opencv-contrib-python | 5.0.0.93 |
| numpy | 2.5.3 |
| GPU/CUDA | não utilizada (CPU only) |

As versões completas, incluindo dependências indiretas, estão em `requirements.lock.txt`.

### Passo a passo
1. Clonar o repositório
2. Criar e ativar o ambiente virtual:
   - Windows: `python -m venv venv` e `venv\Scripts\Activate.ps1`
   - Linux/macOS: `python -m venv venv` e `source venv/bin/activate`
3. Instalar as dependências: `pip install -r requirements.txt`
4. Verificar o ambiente: `python scripts/check_env.py` (deve terminar com "Ambiente OK")

### Observações
- O rtmlib exige `opencv-python` e `opencv-contrib-python`. Os dois devem ficar
  na mesma versão (5.0.0.93); o `check_env.py` avisa se divergirem.
- Na primeira execução, a rtmlib baixa os modelos ONNX (detector e pose) e os
  mantém em cache local. É necessária conexão com a internet nessa primeira vez.
- Para usar GPU, substituir `onnxruntime` por `onnxruntime-gpu` (nunca os dois).