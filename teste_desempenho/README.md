# Teste de desempenho: CPU × NPU × GPU

Roda o pipeline de pose em cada dispositivo e compara os tempos. Todas as
configs usam o mesmo vídeo, o mesmo detector (YOLOX-x) e os mesmos limiares.
Só mudam o modelo de pose, o dispositivo e a pasta de saída.

| Grupo | Configs | O que compara |
|---|---|---|
| RTMPose-x | `x_cpu`, `x_gpu_intel`, `x_npu`, `x_gpu_cuda` | modelo mais preciso em cada dispositivo (`x_npu` tem **só a `Conv_323` em INT8**, veja abaixo) |
| RTMPose-m | `m_cpu`, `m_gpu_intel`, `m_npu`, `m_gpu_cuda` | modelo menor, que roda em **todos** os dispositivos, inclusive a NPU |
| Híbrido | `hibrido_x_npu_gpu` | detector na NPU + RTMPose-x na GPU Intel |

| Dispositivo | Backend | Precisão efetiva | Onde roda |
|---|---|---|---|
| `cpu` | onnxruntime | FP32 | qualquer máquina |
| `gpu_intel` | openvino | FP16 (padrão do OpenVINO em GPU) | GPU Intel |
| `npu` | openvino | FP16 (a NPU não roda FP32); no `x_npu`, a `Conv_323` em INT8 | Intel Core Ultra |
| `gpu_cuda` | onnxruntime | FP32 | GPU NVIDIA (ambiente `cuda`) |

A precisão efetiva de cada modelo é gravada no `run_info.json` e aparece nas
colunas `precisao_det` e `precisao_pose` do comparativo. Parte do ganho da
GPU Intel e da NPU vem de rodarem em FP16. Para saber se isso afetou o resultado,
compare a coluna `confianca_media` com a da CPU (FP32).

**Métricas por pessoa:** o RTMPose roda uma vez para cada pessoa detectada (no
vídeo de teste são ~28 por frame), então `pose_ms` é a soma de todas elas. Para
comparar testes com números diferentes de pessoas, o comparativo traz
`pessoas_por_frame` (contada no `landmarks.csv`), `pose_ms_por_pessoa`, e uma
estimativa para um frame com uma pessoa: `inferencia_1_pessoa_ms` (detector +
uma pose) e `fps_1_pessoa`. A coluna `cpu_maquina` indica em qual máquina cada
teste rodou.

**Modo híbrido:** no YAML, `pose.detector.runtime` é opcional. Sem ele, o
detector usa o mesmo dispositivo da pose. Com ele, cada modelo roda num
dispositivo diferente (veja `configs/hibrido_x_npu_gpu.yaml`).

## Preparar uma máquina do zero (Windows)

1. Instalar as ferramentas (PowerShell):
   ```powershell
   winget install Python.Python.3.12
   winget install Microsoft.VisualStudioCode
   winget install Git.Git
   ```
   Na máquina NVIDIA, instale também o **driver NVIDIA** mais recente (com
   suporte a CUDA 13). **Não** é preciso instalar o CUDA Toolkit nem o cuDNN:
   eles vêm via pip.

2. Abrir a pasta do projeto no VS Code e aceitar as extensões recomendadas.

3. Criar o ambiente e instalar tudo. No VS Code: *Terminal > Run Task...*, ou pelo terminal:
   ```powershell
   # Máquina Intel (CPU / NPU / GPU Intel)
   powershell -ExecutionPolicy Bypass -File setup.ps1

   # Máquina NVIDIA
   powershell -ExecutionPolicy Bypass -File setup.ps1 -Alvo cuda
   ```
   O setup cria o `.venv`, instala os pacotes, baixa os modelos e mostra quais
   dispositivos estão disponíveis.

> Os dois ambientes não se misturam: `onnxruntime` (CPU) e `onnxruntime-gpu`
> ocupam o mesmo módulo Python. Por isso existem `requirements.txt` e
> `requirements-cuda.txt`. No ambiente `cuda`, as configs `npu` e `gpu_intel`
> aparecem como indisponíveis, o que é esperado.

## Rodar

```powershell
.venv\Scripts\python.exe teste_desempenho\run_benchmark.py              # todos os disponíveis
.venv\Scripts\python.exe teste_desempenho\run_benchmark.py x_cpu m_npu  # só alguns
.venv\Scripts\python.exe teste_desempenho\run_benchmark.py --verificar  # só checa dispositivos
```

No VS Code também dá para usar *Run and Debug* (F5): escolha "todos os testes"
ou "escolher um teste...".

Cada teste termina com um destes status no comparativo:

| Status | Significado |
|---|---|
| `completed` | rodou até o fim |
| `indisponível` | o dispositivo não existe nesta máquina (ex.: CUDA sem GPU NVIDIA) |
| `incompatível` | o dispositivo existe, mas o modelo não compila nele. O motivo fica na coluna `observacao` |
| `... (execução anterior)` | o teste não foi rodado agora; o valor vem do último `run_info.json` salvo |

Dispositivos indisponíveis ou incompatíveis não quebram a execução. Se o CUDA for
pedido mas o onnxruntime cair para CPU silenciosamente, a execução **falha de
propósito**, para não gerar um resultado "de GPU" medido na CPU.

## Resultados

```
resultados/
  <config>/run_info.json   detalhes completos da execução (ou o motivo da falha)
  <config>/landmarks.csv   keypoints (para comparar a precisão entre dispositivos)
  comparativo.csv          uma linha por dispositivo
  comparativo.md           mesma tabela, pronta para relatório
```

Principais métricas do `run_info.json`:

| Campo                                     | Significado |
|-------------------------------------------|-------------|
| `performance.model_load_s`                | carregar e compilar os modelos (em GPU/NPU inclui a compilação) |
| `performance.inference_ms.detector/pose/total` | tempo por frame (média, mediana, p95, mín, máx) sem os frames de aquecimento |
| `performance.inference_ms.total.warmup_mean` | média dos primeiros frames (custo de aquecimento) |
| `performance.inference_fps`               | frames/s considerando só a inferência |
| `performance.pipeline_fps`                | frames/s de ponta a ponta (leitura, inferência, desenho, CSV) |
| `detection.detection_rate_pct`            | % de frames com pessoa detectada |
| `detection.mean_keypoint_confidence`      | confiança média dos keypoints. Se cair muito na NPU, a precisão reduzida está afetando o resultado |
| `system`, `runtime`                       | hardware, sistema operacional e versões das bibliotecas usadas |

## RTMPose-x na NPU com a `Conv_323` em INT8 (`x_npu`)

Em FP16, o RTMPose-x não compila na NPU: a camada `Conv_323` não cabe na
memória interna (veja "Limitações conhecidas"). A solução é quantizar **só
essa camada** para INT8. Assim ela passa a caber, e o resto do modelo roda em
FP16, que a NPU executa corretamente.

O modelo é gerado uma vez, num ambiente separado, porque o `nncf` exige
`numpy<2.5` e o ambiente dos testes usa 2.5.3:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1 -Alvo quantizacao   # cria .venv-quant
.venv-quant\Scripts\python.exe teste_desempenho\quantizar_int8.py        # gera modelos\rtmpose-x_int8.xml
.venv\Scripts\python.exe teste_desempenho\run_benchmark.py x_npu         # roda o teste
```

A calibração usa 300 recortes de pessoas de 30 frames do próprio vídeo,
preparados exatamente como a rtmlib prepara a entrada do RTMPose. A camada
quantizada é escolhida com `--camadas`, e o padrão é `Conv_323/WithoutBiases`.

**Resultado (driver NPU 32.0.100.5540, OpenVINO 2026.4.0):** no vídeo inteiro,
0,96 s por frame (1,04 FPS), erro de 0,38% do tamanho da pessoa, 99,6% dos
keypoints concordando com o FP32 e eventos nos mesmos frames. A confiança média
sobe um pouco (0,668 contra 0,643), mas as posições praticamente não mudam.

As tentativas descartadas (FP16 puro e o modelo inteiro em INT8), com o
diagnóstico, estão em `resultados_arquivo/HISTORICO_NPU.md`.

## Métricas de precisão

Não há anotação manual do vídeo. Por isso, a precisão é medida como
**concordância com a referência FP32** (`x_cpu`: RTMPose-x, onnxruntime, CPU).
Ela mostra quanto FP16, INT8 ou o modelo menor desviam da referência, não a
precisão absoluta em relação à pose real. Detalhes em `comparar_precisao.py`.

| Coluna | Significado |
|---|---|
| `nme_pct` | erro médio dos keypoints, em % do tamanho da pessoa (menor = melhor) |
| `pck_5pct` | % de keypoints a menos de 5% do tamanho da pessoa da referência (maior = melhor) |
| `erro_medio_px` | erro médio em pixels |
| `delta_confianca` | diferença média de confiança em relação à referência |
| `pessoas_pareadas_pct` | % das pessoas da referência encontradas no teste |
| `eventos` | se take-off e landing caíram nos mesmos frames da referência |

Só entram keypoints com confiança ≥ 0.40 na referência. O mesmo cálculo avulso:
`python teste_desempenho/comparar_precisao.py [config ...]`.

## Limitações conhecidas

- **NPU:** a NPU só compila modelos com formato de entrada fixo. O pipeline
  converte o modelo sozinho (batch = 1, salvo como `*_static_b1.xml` no cache da
  rtmlib). O RTMPose-x também precisa ter a `Conv_323` em INT8 para caber na
  memória interna da NPU (driver 32.0.100.5540), o que o `x_npu` já faz. O
  detector YOLOX-x e o RTMPose-m rodam na NPU em FP16, sem ajustes.

**Leitura dos tempos:** o tempo de `pose` cresce com o número de pessoas no
frame, porque o RTMPose roda uma vez por pessoa detectada. Compare dispositivos
sempre com o mesmo vídeo.
