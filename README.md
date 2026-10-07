# Spike.AI: pipeline de Pose Estimation (RTMPose)

Pipeline reproduzível que extrai os keypoints do atleta ao longo de um vídeo. Todo o comportamento
(modelo, tamanho de entrada, limiares de confiança, dispositivo, saídas) vem de um arquivo YAML.

```
config.yaml + video.mp4  ->  detector (YOLOX) + RTMPose  ->  keypoints.json + vídeo com o esqueleto
```

## 1. Instalação

Resumo (detalhes, versões testadas e opções de GPU/NPU em [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)):

```powershell
# Na raiz do projeto: cria o .venv, instala as dependências e baixa os modelos
powershell -ExecutionPolicy Bypass -File setup.ps1

# Verifica o ambiente (deve terminar com "Ambiente OK")
.venv\Scripts\python.exe scripts\check_env.py
```

Para GPU NVIDIA, use `setup.ps1 -Alvo cuda`. Na primeira execução a `rtmlib` baixa os modelos ONNX
(é preciso internet nessa vez).

Notas sobre os modelos:

- O `setup.ps1` pré-baixa só os modelos das configs do benchmark (`teste_desempenho/configs`, COCO-17). Os
  checkpoints dos candidatos em `config/candidatos/` (Halpe26 e WholeBody, de 50 a 204 MB cada) são baixados
  na primeira vez que o candidato roda.
- A config `x_npu` depende de um modelo INT8 gerado localmente (`teste_desempenho/modelos/`, fora do Git).
  Num clone novo ela é pulada no download, e só funciona depois de rodar `setup.ps1 -Alvo quantizacao` e
  `quantizar_int8.py` (veja `teste_desempenho/README.md`). Não afeta o uso do `run_pose.py`.

## 2. Como executar

```powershell
.venv\Scripts\python.exe run_pose.py --video input\video.mp4 --config config\pose_config.yaml
```

| Argumento | Descrição |
|---|---|
| `--config` | YAML com modelo, limiares e saídas. Padrão: `config/pose_config.yaml` |
| `--video` | Vídeo de entrada (ou índice da webcam, ex.: `0`). Opcional: substitui `input.source` do YAML |

Trocar de modelo ou candidato é só trocar o `--config`, sem mexer no código:

```powershell
.venv\Scripts\python.exe run_pose.py --video input\video.mp4 --config config\candidatos\candidato_a.yml
```

O YAML é validado **antes** de qualquer modelo ser carregado. Se algo estiver errado, a execução para com
uma mensagem que diz qual chave corrigir (por exemplo, `'uncertain' deve ser menor que 'valid'`).
Um `--video` inexistente também para logo no início.

## 3. Estrutura do YAML

Exemplo comentado em [`config/pose_config.yaml`](config/pose_config.yaml). Caminhos relativos são
resolvidos a partir da raiz do projeto. Chaves desconhecidas são recusadas (isso pega erro de digitação).

```yaml
input:
  source: "input/video.mp4"
  show_preview: false

pose:
  model: "rtmpose_x_halpe26"
  checkpoint: "https://.../rtmpose-x_...zip"   # URL ou caminho local (.onnx/.zip)
  input_size: {width: 288, height: 384}        # atenção: "384x288" do nome do modelo é altura x largura
  keypoint_layout: "halpe26"                   # coco17 | halpe26 | coco_wholebody
  landmarks: [right_shoulder, right_elbow, right_wrist, right_hip,
              right_knee, right_ankle, right_heel, right_toe]
  detector:
    model: "yolox_x"
    checkpoint: "https://.../yolox_x_...zip"
    input_size: {width: 640, height: 640}
  runtime: {backend: "openvino", device: "cpu"}
  confidence_thresholds: {valid: 0.70, uncertain: 0.40}
  output:
    dir: "output/meu_teste"
    landmarks_csv: "landmarks.csv"
    keypoints_json: "keypoints.json"
    events_json: "events_ambiguous.json"
    run_info: "run_info.json"
    include_skeleton_video: true
    skeleton_video: "skeleton.mp4"
    highlight_athlete: true

metrics:
  warmup_frames: 5
```

| Chave | Obrigatória | O que controla |
|---|---|---|
| `input.source` | sim | Vídeo de entrada (ou índice da webcam). `--video` tem prioridade |
| `input.show_preview` | sim | Mostra uma janela durante o processamento (`q` fecha) |
| `pose.model`, `pose.checkpoint` | sim | Modelo de pose (nome para logs e checkpoint) |
| `pose.input_size` | sim | Largura e altura de entrada do modelo (inteiros positivos) |
| `pose.keypoint_layout` | não (`coco17`) | Layout dos keypoints do modelo. **Precisa bater com o checkpoint** |
| `pose.landmarks` | não (8 do MVP, lado direito) | Quais landmarks padronizados entram no `keypoints.json`, e em que ordem |
| `pose.detector` | sim | Detector de pessoas (`model`, `checkpoint`, `input_size`, `runtime` opcional) |
| `pose.runtime` | sim | `backend` e `device` (veja abaixo) |
| `pose.confidence_thresholds` | sim | Limiares `valid` e `uncertain`, com `0 <= uncertain < valid <= 1` |
| `pose.output.*` | sim (alguns opcionais) | Pasta e nomes dos arquivos de saída |
| `pose.output.frame_number` | não | Número do frame no canto do vídeo (`show`, `start_at: 0 ou 1`) |
| `pose.output.highlight_athlete` | não (`true`) | Destaca o atleta no vídeo; as outras pessoas saem esmaecidas |
| `pose.note` | não | Observação livre, gravada no `run_info.json` |
| `metrics.warmup_frames` | sim | Frames iniciais fora das estatísticas de tempo |

Combinações de `backend` e `device`: `onnxruntime` (cpu, cuda, cuda:N, rocm, mps), `openvino`
(cpu, gpu, npu) e `opencv` (cpu, cuda). O detector pode ter um `runtime` próprio (modo híbrido).

## 4. Como o YAML afeta o `keypoints.json`

**Status de cada landmark** (a regra está em `ConfidenceThresholds.classify`, em `src/config.py`):

| Condição | Status | `x` e `y` |
|---|---|---|
| `confidence >= valid` | `valid` | valores em pixels |
| `uncertain <= confidence < valid` | `uncertain` | valores em pixels |
| `confidence < uncertain` | `missing` | **`null`** (nunca `(0, 0)`) |

Os mesmos limiares valem no vídeo (verde = `valid`, laranja = `uncertain`, `missing` não é desenhado)
e nos eventos ambíguos. Exemplo: a mesma confiança muda de status só trocando o YAML.

| Confiança | `valid: 0.70` / `uncertain: 0.40` | `valid: 0.85` / `uncertain: 0.50` |
|---|---|---|
| 0,82 | `valid` | `uncertain` |
| 0,55 | `uncertain` | `uncertain` |
| 0,45 | `uncertain` | `missing` (x, y = null) |
| 0,12 | `missing` (x, y = null) | `missing` (x, y = null) |

**Layout e nomes padronizados.** O `keypoint_layout` diz como interpretar os pontos do modelo, e os nomes
no JSON são os mesmos para qualquer checkpoint (`right_heel` é o índice 25 no `halpe26` e o 22 no
`coco_wholebody`). O dedão é o `right_toe`. Se o layout não tem o ponto (por exemplo `right_heel` em
`coco17`), ele sai com `x`, `y` e `confidence` = `null` e `status: "missing"`. Se o layout não bater com o número
de keypoints que o modelo devolve, a execução para no 1º frame com status `incompatível` no `run_info.json`.

**Landmarks exportados.** Só os de `pose.landmarks` entram no JSON. O `landmarks.csv` guarda todos os
pontos do modelo, com valores brutos.

## 5. Onde os resultados são salvos

Tudo vai para `pose.output.dir` (cada config pode usar a sua pasta, como `output/candidato_a_halpe26_x/`):

```
output/<pasta_do_yaml>/
├── keypoints.json          landmarks padronizados com status, por frame e por pessoa
├── skeleton.mp4            vídeo com o esqueleto desenhado (nome em pose.output.skeleton_video)
├── landmarks.csv           todos os pontos do modelo, valores brutos, sem status
├── events_ambiguous.json   eventos candidatos (decolagem e aterrissagem) a partir do CSV
└── run_info.json           tempos de execução, hardware e estatísticas
```

Formato do `keypoints.json` (um item em `frames` por frame do vídeo):

```json
{
  "metadata": {"model": "rtmpose_x_halpe26", "keypoint_layout": "halpe26",
               "confidence_thresholds": {"valid": 0.7, "uncertain": 0.4}, "fps": 30.0, "...": "..."},
  "frames": [{
    "frame_index": 2,
    "timestamp_ms": 66.67,
    "people": [{
      "person_index": 0,
      "keypoints": {
        "right_shoulder": {"x": 1650.03, "y": 923.63, "confidence": 0.8193, "status": "valid"},
        "right_heel":     {"x": null, "y": null, "confidence": 0.1273, "status": "missing"}
      }
    }]
  }]
}
```

Os detalhes de cada campo estão em [`docs/TASKS.md`](docs/TASKS.md).

## 6. Limitações conhecidas

- **Seleção do atleta:** o vídeo destaca a pessoa de maior bounding box no frame, e as demais saem
  esmaecidas em cinza. É uma heurística por frame, sem tracking, feita para a inspeção visual; com
  `pose.output.highlight_athlete: false` todas são desenhadas igual. O `keypoints.json` continua
  trazendo **todas** as pessoas, e o `person_index` vale só dentro do frame.
- **Candidato C (WholeBody 384x288):** a confiança sai fora de 0 a 1, então os limiares de 0,70/0,40 não
  fazem sentido para ele. Veja [`docs/CANDIDATOS.md`](docs/CANDIDATOS.md).
- **Limiares:** os valores atuais são arbitrários. Os definitivos vêm do benchmark (Issue #10).

## 7. Outros documentos

| Arquivo | Conteúdo |
|---|---|
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | Instalação, versões testadas, GPU/NPU |
| [`docs/TASKS.md`](docs/TASKS.md) | Como as tasks 2, 4, 5, 6, 7 e 8 foram resolvidas |
| [`docs/CANDIDATOS.md`](docs/CANDIDATOS.md) | Configurações candidatas e seus landmarks |
| [`teste_desempenho/README.md`](teste_desempenho/README.md) | Benchmark de desempenho por hardware |
