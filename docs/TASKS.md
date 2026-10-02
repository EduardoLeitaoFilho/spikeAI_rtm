# Tasks 2, 4, 5 e 6: o que foi feito e como

Resumo das tasks do pipeline de pose (RTMPose via rtmlib) e de como cada uma foi resolvida.
Para instalação, veja `docs/ENVIRONMENT.md`. Para as configs candidatas (Task 3), veja `docs/CANDIDATOS.md`.

## Visão geral do fluxo

```
run_pose.py --video V --config C.yaml
   └─ src/main.py
        ├─ src/config.py            lê e valida o YAML (Task 2)
        ├─ PoseDetector             YOLOX (detector) -> RTMPose, em cada frame
        ├─ CSVExporter              landmarks.csv: todos os pontos do modelo, valores brutos
        ├─ KeypointsJSONExporter    keypoints.json: landmarks padronizados com status (Tasks 5 e 6)
        └─ run_info.json, events_ambiguous.json, annotated.mp4
```

| Arquivo | Papel |
|---|---|
| `run_pose.py` | Comando de execução (Task 4) |
| `src/config.py` | Esquema e leitura do YAML (Task 2) e o Threshold Engine, `ConfidenceThresholds.classify` (Task 5) |
| `src/keypoint_layouts.py` | Layouts de keypoints (COCO-17, Halpe26, WholeBody) e nomes padronizados (Task 5) |
| `src/keypoints_export.py` | Geração do `keypoints.json` (Task 6) |
| `src/main.py` | Pipeline: inferência, desenho, exportação e métricas |
| `config/pose_config.yaml` | Config principal, comentada |

---

## Task 2: analisador de configuração (YAML)

**Pedido:** nenhum parâmetro de modelo ou limiar fixo no código; todo o comportamento vem de um YAML.

**O que foi feito:**

- **Esquema** em `src/config.py`, como dataclasses imutáveis: `PipelineConfig` → `InputConfig`, `PoseConfig` (modelo, checkpoint, `input_size`, detector, runtime, limiares, saídas, layout, landmarks) e `MetricsConfig`.
- **Leitura** com `load_config(path, base_dir)`, que usa `yaml.safe_load`. Caminhos relativos são resolvidos a partir da raiz do projeto.
- **Validação com erro claro** (`ConfigError`), antes de carregar qualquer modelo:
  - chave obrigatória ausente ou chave desconhecida (pega erro de digitação);
  - tipos: string não vazia, booleano, inteiro positivo, número entre 0 e 1;
  - `uncertain < valid`;
  - checkpoint local inexistente (uma URL é mantida e baixada pela rtmlib);
  - combinação backend/device suportada (`onnxruntime`: cpu/cuda/cuda:N/rocm/mps; `openvino`: cpu/gpu/npu; `opencv`: cpu/cuda).
- **Além do exemplo da task:**
  - seção `detector`, com runtime próprio opcional (modo híbrido, por exemplo detector na NPU e pose na GPU);
  - seções `runtime` e `input`;
  - `metrics.warmup_frames`;
  - `note`;
  - chaves opcionais com padrão: `keypoint_layout`, `landmarks`, `output.keypoints_json` e `output.frame_number`. Assim, YAMLs antigos continuam válidos.

**Como verificar:** os 14 YAMLs do repositório (principal, 9 do benchmark e 4 candidatos) carregam sem erro. Um valor inválido para a execução com uma mensagem que diz qual chave está errada, por exemplo `'pose.confidence_thresholds': 'uncertain' deve ser menor que 'valid'`.

---

## Task 4: comando de execução

**Pedido:** um comando simples, `--video` + `--config`; trocar de configuração não pode exigir mudança no código.

**O que foi feito:**

```powershell
python run_pose.py --video input/video.mp4 --config config/pose_config.yaml
python run_pose.py --video input/video.mp4 --config config/candidatos/candidato_a.yml
```

- `run_pose.py`, na raiz, só põe `src/` no `sys.path` e chama `main()`. O `python src/main.py --config ...` continua funcionando; é o que o `run_benchmark.py` usa.
- `--config`: qualquer YAML. O padrão é `config/pose_config.yaml`.
- `--video`: opcional. **Substitui `input.source` do YAML** sem editar o arquivo, assim o mesmo config roda em vários vídeos.
  - um caminho relativo é resolvido a partir do diretório atual (como em qualquer comando de terminal);
  - um número (`--video 0`) é o índice da webcam;
  - um arquivo inexistente para a execução logo no início com `Vídeo não encontrado: <caminho>`.
- O caminho de saída vem do YAML (`pose.output.dir`). Por isso cada candidato grava na própria pasta (`output/candidato_a_halpe26_x/`...).

**Como verificar:** rodar o mesmo comando trocando só o `--config` entre `config/pose_config.yaml` (COCO-17) e `config/candidatos/candidato_a.yml` (Halpe26). Os dois funcionam sem nenhuma mudança no código.

---

## Task 5: Threshold Engine e landmarks padronizados

### Status de cada landmark

**Pedido:** classificar cada landmark com as regras lidas do YAML, antes de salvar o JSON.

**O que foi feito:** `ConfidenceThresholds.classify(conf)` em `src/config.py`. Os limites vêm de `pose.confidence_thresholds` no YAML:

| Regra | Status |
|---|---|
| `confidence >= valid` | `valid` |
| `uncertain <= confidence < valid` | `uncertain` |
| `confidence < uncertain` | `missing` |

A mesma função é usada em todo o pipeline, então a regra existe em um único lugar:

- no `keypoints.json` (campo `status`);
- no vídeo anotado: `valid` é desenhado em verde, `uncertain` em laranja, e `missing` não é desenhado (nem as linhas do esqueleto ligadas a ele);
- nos eventos ambíguos: frames com tornozelo `missing` são descartados, e o `status` do evento usa a mesma regra.

Antes, o terceiro status se chamava `invalid`. Foi renomeado para `missing`, conforme a task.

### Nomes padronizados

**Pedido:** o `keypoints.json` deve usar os mesmos nomes de landmark (`right_shoulder`, ..., `right_heel`, `right_toe`) qualquer que seja o checkpoint.

**Problema:** cada checkpoint entrega um layout diferente, com número de pontos e índices diferentes:

| Layout | Pontos | Modelos | `right_heel` | `right_toe` (dedão) |
|---|---|---|---|---|
| `coco17` | 17 | RTMPose body (configs do benchmark) | não tem | não tem |
| `halpe26` | 26 | candidatos A, B | índice 25 | índice 21 (`right_big_toe`) |
| `coco_wholebody` | 133 | candidatos C, D | índice 22 | índice 20 (`right_big_toe`) |

**O que foi feito:**

1. **Layout declarado no YAML:** `pose.keypoint_layout: coco17 | halpe26 | coco_wholebody` (padrão `coco17`). `src/keypoint_layouts.py` define os nomes de todos os pontos e o esqueleto de cada layout.
2. **Conferência:** se o modelo devolver um número de pontos diferente do layout declarado, a execução para no 1º frame com status `incompatível` no `run_info.json`. Antes, o pipeline cortava silenciosamente em 17 pontos, e calcanhar e dedão se perdiam.
3. **Lista de landmarks no YAML:** `pose.landmarks` define quais nomes padronizados entram no `keypoints.json` e em que ordem. O padrão são os 8 do MVP (lado direito).
4. **Do nome ao índice:** `KeypointLayout.index_of(nome)`. Quase todos os nomes são iguais nos três layouts. A diferença é tratada em `STANDARD_ALIASES` (`right_toe` → `right_big_toe`, `left_toe` → `left_big_toe`). Assim `right_heel` é o índice 25 no Halpe26 e o 22 no WholeBody, sem nenhum `if` por checkpoint.
5. **Nome digitado errado é recusado ao ler o YAML** (`landmark(s) desconhecido(s): ['right_wirst']`), assim como nomes repetidos.

```yaml
pose:
  keypoint_layout: "halpe26"
  landmarks: [right_shoulder, right_elbow, right_wrist, right_hip,
              right_knee, right_ankle, right_heel, right_toe]
```

**Landmark que o modelo não fornece** (por exemplo `right_heel` com `coco17`): sai com `x`, `y` e `confidence` = `null` e `status: "missing"`. O JSON tem as mesmas chaves para qualquer checkpoint, e `confidence: null` diferencia "o modelo não mede esse ponto" de "o modelo mediu com confiança baixa". A lista desses pontos também aparece em `metadata.not_provided_by_model`.

---

## Task 6: keypoints.json

**Pedido:** para cada frame, salvar o índice do frame, o timestamp, as coordenadas, a confiança e o status. Um landmark `missing` deve ter `x`/`y` = `null`, nunca (0,0).

**O que foi feito:** `KeypointsJSONExporter` em `src/keypoints_export.py`. O caminho do arquivo é `pose.output.dir` + `pose.output.keypoints_json` (padrão `keypoints.json`).

```json
{
  "metadata": {
    "model": "rtmpose_x_halpe26",
    "keypoint_layout": "halpe26",
    "landmarks": ["right_shoulder", "...", "right_toe"],
    "not_provided_by_model": [],
    "confidence_thresholds": {"valid": 0.7, "uncertain": 0.4},
    "config_file": "...", "video": "...", "fps": 30.0, "width": 1920, "height": 1080
  },
  "frames": [
    {
      "frame_index": 2,
      "timestamp_ms": 66.67,
      "people": [
        {
          "person_index": 0,
          "keypoints": {
            "right_shoulder": {"x": 1650.03, "y": 923.63, "confidence": 0.8193, "status": "valid"},
            "right_knee":     {"x": 1621.44, "y": 1074.26, "confidence": 0.684, "status": "uncertain"},
            "right_heel":     {"x": null, "y": null, "confidence": 0.1273, "status": "missing"}
          }
        }
      ]
    }
  ]
}
```

**Como cada campo é gerado:**

- **`frame_index`:** índice do frame no vídeo, começando em 0. É o mesmo da coluna `frame` do `landmarks.csv`.
- **`timestamp_ms`:** posição do frame lida do vídeo pelo OpenCV (`CAP_PROP_POS_MSEC`, logo após a leitura do frame). Isso respeita vídeos com FPS variável. Na webcam, onde esse valor pode vir zerado, usa `frame_index / fps`.
- **`x`, `y`:** em pixels do vídeo original, arredondados em 2 casas decimais. **`missing` → `null`**, nunca (0,0).
- **`confidence`:** valor bruto do modelo, com 4 casas decimais. Só é `null` quando o modelo não fornece o ponto.
- **`status`:** resultado da Task 5.

**Diferença em relação ao exemplo da task: lista `people`.** O detector encontra várias pessoas por frame (cerca de 24 no vídeo de teste: jogadoras, banco e torcida). Por isso cada frame tem uma lista `people`, e dentro de cada pessoa os `keypoints` seguem exatamente o formato do exemplo. Escolher qual pessoa é o atleta ficou para uma etapa posterior, para não descartar dados com uma heurística ainda não validada.

> **Atenção:** `person_index` é a ordem de detecção **dentro do frame**. Ele **não** identifica a mesma pessoa entre frames, porque ainda não há tracking.

**Outros arquivos de saída** (sem mudança de formato):

- `landmarks.csv`: todos os pontos do modelo (17, 26 ou 133), valores brutos, sem status. Serve para análise e para o `comparar_precisao.py`.
- `events_ambiguous.json`, `run_info.json`, `annotated.mp4`.

---

## Testes realizados

| Teste | Resultado |
|---|---|
| Os 14 YAMLs do repositório carregam | OK |
| `run_pose.py` com um clipe de 10 frames, config `coco17` (RTMPose-m) | OK, `keypoints.json` com 10 frames |
| `run_pose.py` com o mesmo clipe, candidato A (`halpe26`), só trocando o `--config` | OK |
| Conferência de 3.840 landmarks do JSON contra o CSV bruto: status pela regra da Task 5, confiança, coordenadas | 0 erros |
| Todo `missing` tem `x`/`y` = `null`; nenhum (0,0) | OK |
| `coco17`: `right_heel`/`right_toe` saem com tudo `null` e `missing` | OK |
| `timestamp_ms` a 30 fps: 0; 33,33; 66,67; ... | OK |
| Landmark com erro de digitação, landmark repetido, lista vazia | recusados ao ler o YAML |
| `--video` inexistente | erro claro, nada é executado |
| Layout errado (candidato A declarado como `coco17`) | para no 1º frame, status `incompatível` |
| `comparar_precisao.py` nos resultados de 17 pontos que já existem | saída idêntica à versão anterior |

## Limitações conhecidas

- **Seleção do atleta:** ainda não existe. O `keypoints.json` traz todas as pessoas, sem tracking entre frames.
- **Candidato C:** a confiança sai fora de 0 a 1 (entre 4 e 7). Com os limiares de 0,70/0,40, quase tudo vira `valid`, então o `status` dele não é comparável com os outros candidatos. A diferença foi mantida de propósito (veja `docs/CANDIDATOS.md`).
- **Limiares:** os valores atuais (0,70/0,40) são arbitrários. Os definitivos vêm do benchmark (Issue #10).
