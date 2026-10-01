# Configurações candidatas (T3, Issue #9)

Estas configurações só **preparam** os candidatos para o benchmark da Issue #10.
Nenhuma é a escolha final do MVP. Os thresholds (`valid: 0.70`, `uncertain: 0.40`) são arbitrários.

Todos usam `onnxruntime` em CPU e foram carregados com sucesso pelo `load_config` de `src/config.py`.

## Tabela

| Candidato | Arquivo | Pose (checkpoint) | Input size (pose, LxA) | Detector | Keypoints |
|---|---|---|---|---|---|
| A | `candidato_a_halpe26_x.yaml` | RTMPose-x Halpe26 (`rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606`) | 288 x 384 | YOLOX-x, 640 x 640 | 26 |
| B | `candidato_b_halpe26_m.yaml` | RTMPose-m Halpe26 (`rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605`) | 192 x 256 | YOLOX-m, 640 x 640 | 26 |
| C | `candidato_c_wholebody_384.yaml` | RTMW-dw-x-l (`rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122`) | 288 x 384 | YOLOX-m, 640 x 640 | 133 |
| D | `candidato_d_wholebody_256.yaml` | RTMW-dw-x-l (`rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122`) | 192 x 256 | YOLOX-m, 640 x 640 | 133 |

Os checkpoints são URLs do openmmlab (as mesmas que o `rtmlib` usa em `BodyWithFeet` e `Wholebody`).

## Landmarks do MVP (todos disponíveis nos 4 candidatos)

Índices do lado direito em cada layout (o lado esquerdo também existe):

| Landmark | Halpe26 (A, B) | COCO-WholeBody (C, D) |
|---|---|---|
| right_shoulder | 6 | 6 |
| right_elbow | 8 | 8 |
| right_wrist | 10 | 10 |
| right_hip | 12 | 12 |
| right_knee | 14 | 14 |
| right_ankle | 16 | 16 |
| right_heel | 25 | 22 |
| right_toe (dedão) | 21 | 20 |

Calcanhar e dedos têm **índices diferentes** entre os dois layouts. O mapeamento de nomes (T5) precisa ser por candidato.

## Observações da execução de teste (1 frame, CPU, 1920x1080)

Confiança dos landmarks de uma pessoa em um único frame, **sem valor para ranquear modelos** (isso é a Issue #10):

| Candidato | Pessoas detectadas | Tempo (1 frame) | Confiança heel / toe |
|---|---|---|---|
| A | 2 | 1,08 s | 0,67 / 0,74 |
| B | 2 | 0,37 s | 0,72 / 0,69 |
| C | 2 | 0,66 s | 5,32 / 6,26 (fora de 0 a 1) |
| D | 2 | 0,48 s | 0,75 / 0,69 |

- **Candidato C:** a confiança saiu fora do intervalo 0 a 1 (entre 4 e 7). Com thresholds de 0,70 e 0,40, tudo viraria `valid`. Precisa de normalização, ou de uma decisão sobre como tratar esse modelo, antes do benchmark.
- **Candidatos C e D** usam o mesmo modelo (RTMW-dw-x-l) em tamanhos de entrada diferentes. O "balanced" do `rtmlib` não é um modelo menor, só uma entrada menor.
- Os 4 detectaram 2 pessoas no frame de teste, então a seleção do atleta é necessária.