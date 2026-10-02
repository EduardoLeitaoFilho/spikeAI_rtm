# Configurações candidatas (T3, Issue #9)

Estas configurações só **preparam** os candidatos para o benchmark da Issue #10.
Nenhuma é a escolha final do MVP. Os thresholds (`valid: 0.70`, `uncertain: 0.40`) são arbitrários.

Todos usam `onnxruntime` em CPU e foram carregados com sucesso pelo `load_config` de `src/config.py`.

## Tabela

| Candidato | Arquivo | Pose (checkpoint) | Input size (pose, LxA) | Detector | Keypoints |
|---|---|---|---|---|---|
| A | `config/candidatos/candidato_a.yml` | RTMPose-x Halpe26 (`rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606`) | 288 x 384 | YOLOX-x, 640 x 640 | 26 |
| B | `config/candidatos/candidato_b.yml` | RTMPose-m Halpe26 (`rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605`) | 192 x 256 | YOLOX-m, 640 x 640 | 26 |
| C | `config/candidatos/candidato_c.yml` | RTMW-dw-x-l (`rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122`) | 288 x 384 | YOLOX-m, 640 x 640 | 133 |
| D | `config/candidatos/candidato_d.yml` | RTMW-dw-x-l (`rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122`) | 192 x 256 | YOLOX-m, 640 x 640 | 133 |

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

## Layout no pipeline (`pose.keypoint_layout`)

Cada YAML declara o layout do modelo, e o pipeline usa os nomes e o esqueleto definidos em `src/keypoint_layouts.py`:

| Valor | Keypoints | Candidatos | Colunas no `landmarks.csv` |
|---|---|---|---|
| `coco17` (padrão) | 17 | configs do benchmark | 2 + 17 x 3 = 53 |
| `halpe26` | 26 | A, B | 2 + 26 x 3 = 80 |
| `coco_wholebody` | 133 | C, D | 2 + 133 x 3 = 401 (todos os pontos, inclusive rosto e mãos) |

- As colunas do CSV usam **nomes**, não índices (`right_heel_x`, `right_big_toe_conf`...), então quem lê o CSV não precisa saber o índice de cada layout. O dedão é `*_big_toe`.
- Rosto e mãos do WholeBody são `face_0..67`, `left_hand_*` e `right_hand_*` (punho + 4 juntas por dedo).
- Se o layout não bater com o número de keypoints que o modelo devolve, a execução para na 1ª inferência com status `incompatível` no `run_info.json`.
- `teste_desempenho/comparar_precisao.py` compara só os keypoints presentes nos dois CSVs, pelo nome (entre Halpe26 e WholeBody: corpo COCO-17 + 6 pontos dos pés).

## Observações da execução de teste (1 frame, CPU, 1920x1080)

Confiança dos landmarks de uma pessoa em um único frame, **sem valor para ranquear modelos** (isso é a Issue #10):

| Candidato | Pessoas detectadas | Tempo (1 frame) | Confiança heel / toe |
|---|---|---|---|
| A | 2 | 1,08 s | 0,67 / 0,74 |
| B | 2 | 0,37 s | 0,72 / 0,69 |
| C | 2 | 0,66 s | 5,32 / 6,26 (fora de 0 a 1) |
| D | 2 | 0,48 s | 0,75 / 0,69 |

- **Candidatos C e D** usam a mesma arquitetura (RTMW-dw-x-l) em tamanhos de entrada diferentes. O "balanced" do `rtmlib` não é um modelo menor, só uma entrada menor. Mas são **checkpoints distintos**, e isso aparece na escala da confiança (próximo item).

## Candidato C: confiança fora de 0 a 1 (diferença mantida)

O checkpoint do candidato C (`rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122`) devolve confiança **fora do intervalo 0 a 1**: entre 4 e 7 nos frames de teste (heel/toe = 5,70 / 5,34 em outra amostra, 3 frames a partir do frame 30). O D, mesma arquitetura em 256x192, fica entre 0 e 1.

**Decisão:** o candidato C continua no benchmark **sem normalização**, e a diferença fica registrada aqui. Consequências:

- Com `valid: 0.70` / `uncertain: 0.40`, praticamente todos os keypoints do C são classificados como `valid`. A classificação valid/uncertain/missing, o desenho, o `status` do `keypoints.json` e o `status` dos eventos do C **não têm o mesmo significado** dos outros candidatos.
- Colunas `*_conf` do CSV, `mean_keypoint_confidence` do `run_info.json` e `delta_confianca` do `comparar_precisao.py` **não são comparáveis** entre o C e os demais.
- Posição dos keypoints (erro em px, NME, PCK) e tempos **são comparáveis** normalmente: a escala da confiança não afeta as coordenadas.
- Os 4 detectaram 2 pessoas no frame de teste, então a seleção do atleta é necessária.