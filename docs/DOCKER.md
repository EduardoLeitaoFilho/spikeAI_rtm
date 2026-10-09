# Executar com Docker (CPU)

Processa um vídeo sem instalar Python, OpenVINO ou dependências na máquina. Só precisa de
Docker (Docker Desktop no Windows/macOS) e internet na **primeira** execução, para baixar os modelos.

Configuração usada: `config/docker_cpu.yaml` (RTMPose-x Halpe26 + YOLOX-x, OpenVINO em CPU, FP32).
É a adaptação para CPU da execução feita na NPU: mesmos modelos, com resultado idêntico à referência FP32.

## Passo a passo

1. Coloque o vídeo na pasta `input/` (na raiz do repositório).
2. Rode (na raiz do repositório):

   ```bash
   docker compose run --rm pose meu_video.mp4
   ```

   Linux/macOS: use `UID=$(id -u) GID=$(id -g) docker compose run --rm pose meu_video.mp4`
   para os arquivos saírem com o seu usuário.
3. Pegue os resultados em `output/meu_video/`.

Na 1ª vez o Docker constrói a imagem e o programa baixa os modelos (várias centenas de MB, ficam no volume
`rtmlib-cache`). Nas próximas, nada é baixado de novo.

**Tempo:** é processamento em CPU, bem lento: cerca de 2,7 s por frame num Core Ultra 5 (um vídeo de
211 frames levou ~9 min). Teste primeiro com um vídeo curto.

## Resultados (`output/<nome_do_video>/`)

| Arquivo | Conteúdo |
|---|---|
| `keypoints.json` | landmarks por frame e por pessoa, com `timestamp_ms` e `status` |
| `skeleton.mp4` | vídeo de inspeção com o esqueleto desenhado (atleta em destaque, número do frame no canto) |
| `landmarks.csv` | todos os pontos do modelo, valores brutos |
| `events_ambiguous.json` | eventos candidatos (decolagem e aterrissagem) |
| `run_info.json` | tempos, hardware e estatísticas da execução |

### Pontos ausentes ou incertos

Cada landmark do `keypoints.json` tem `status`:

| `status` | Condição (confiança) | `x`, `y` |
|---|---|---|
| `valid` | `>= 0.70` | pixels |
| `uncertain` | `>= 0.40` e `< 0.70` | pixels |
| `missing` | `< 0.40` | `null` |

No vídeo: verde = `valid`, laranja = `uncertain`, `missing` não é desenhado. Os limiares ficam em
`pose.confidence_thresholds` do `config/docker_cpu.yaml` e são provisórios (definitivos: Issue #10).
Mais detalhes em [`README.md`](../README.md), seções 4 e 5.

## Problemas comuns

| Sintoma | Causa / solução |
|---|---|
| `Vídeo não encontrado` | O vídeo precisa estar em `input/`; a mensagem lista os que o container enxerga |
| `Permission denied` em `output/` | Crie a pasta antes (`mkdir output`) ou use o `UID`/`GID` como acima |
| Falha ao baixar modelos | Precisa de internet na 1ª execução; rode de novo |
| Muito lento ou falta de memória | Aumente CPU/memória do Docker Desktop |