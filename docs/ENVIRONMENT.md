## Instalação

### Ambiente testado
| Item | Versão |
|---|---|
| SO | Windows 11 |
| Python | 3.12.10 |
| rtmlib | 0.0.16 |
| onnxruntime | 1.30.0 (CPU) ou onnxruntime-gpu 1.30.0 (CUDA) |
| openvino | 2026.4.0 (CPU, GPU Intel e NPU Intel) |
| opencv-python | 5.0.0.93 |
| opencv-contrib-python | 5.0.0.93 |
| numpy | 2.5.3 |
| pyyaml | 6.0.3 |

As versões completas, incluindo dependências indiretas, estão em `requirements.lock.txt`.

### Arquivos de dependências
| Arquivo | Uso |
|---|---|
| `requirements-base.txt` | Comum a todos os ambientes (não instalar sozinho) |
| `requirements.txt` | Padrão: CPU, GPU Intel e NPU Intel (onnxruntime + OpenVINO) |
| `requirements-cuda.txt` | GPU NVIDIA (onnxruntime-gpu, CUDA 13 e cuDNN 9 via pip) |
| `requirements-quantizacao.txt` | Ambiente separado `.venv-quant` para gerar o modelo INT8 da NPU |

### Passo a passo
1. Clonar o repositório
2. Na raiz do projeto, rodar o setup (cria o `.venv`, instala, baixa os modelos e verifica os dispositivos):
   - CPU / GPU Intel / NPU Intel: `powershell -ExecutionPolicy Bypass -File setup.ps1`
   - GPU NVIDIA: `powershell -ExecutionPolicy Bypass -File setup.ps1 -Alvo cuda`
3. Verificar o ambiente: `.venv\Scripts\python.exe scripts\check_env.py` (deve terminar com "Ambiente OK")
4. Rodar o pipeline: `.venv\Scripts\python.exe run_pose.py --video input\video.mp4 --config config\pose_config.yaml`

Instalação manual (sem o setup): `python -m venv .venv` e `pip install -r requirements.txt`.

### Observações
- O rtmlib exige `opencv-python` e `opencv-contrib-python`. Os dois devem ficar
  na mesma versão (5.0.0.93); o `check_env.py` avisa se divergirem.
- `onnxruntime` e `onnxruntime-gpu` nunca devem ser instalados juntos. Como a rtmlib
  declara `onnxruntime` como dependência, no alvo `cuda` ela é instalada com `--no-deps`
  (o `setup.ps1` faz isso).
- Na primeira execução, a rtmlib baixa os modelos ONNX (detector e pose) e os
  mantém em cache local. É necessária conexão com a internet nessa primeira vez
  (o `setup.ps1` já faz esse download).
- Backend e dispositivo são escolhidos no YAML (`pose.runtime`), não no código.
