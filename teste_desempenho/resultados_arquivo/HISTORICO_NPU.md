# Histórico: RTMPose-x na NPU (tentativas descartadas)

Registro das tentativas que não entraram na versão final. A versão final do
`x_npu` (só a `Conv_323` em INT8, resto em FP16) está descrita no
`teste_desempenho/README.md`.

## 1. RTMPose-x em FP16 na NPU: incompatível

O compilador da NPU (driver 32.0.100.5540) não compila o RTMPose-x em FP16
("FeasibleAllocation: cannot schedule anything"). A camada `Conv_323` precisa de
~2,3 MB de memória interna, e só há ~1,9 MB livres por núcleo. Resultados em
`2026-09-30_rodada_fp32_fp16/`.

## 2. Modelo inteiro em INT8: inválido

A primeira tentativa quantizou o modelo inteiro. Os pesos caíram de 188,5 MB
para 48,1 MB e o modelo compilou, mas **a NPU executou o modelo errado**. O mesmo
arquivo INT8 na CPU concordava com o FP32 (7,8 px de erro no recorte, confiança
0,588 contra 0,593). Na NPU, os pontos saíam praticamente aleatórios (217 px,
confiança 0,269). Esse resultado está arquivado em
`2026-09-30_x_npu_int8_modelo_inteiro_invalido/`.

**Diagnóstico camada a camada (30/09/2026, observação; os scripts do diagnóstico
não ficaram no projeto):** comparamos as saídas intermediárias do modelo INT8 na
CPU e na NPU contra o FP32, em 6 recortes reais de pessoas.

- **A NPU erra desde as primeiras camadas do backbone.** Nas primeiras medições,
  o erro relativo é de 112–120% na NPU e de 24–31% na CPU. O erro não nasce na
  `Conv_323` nem no bloco de atenção (GAU) da cabeça: eles só herdam o erro.
- **Não é estouro numérico:** nenhum NaN/infinito, e os valores máximos da NPU
  ficam na mesma faixa dos do FP32.
- **O erro alterna entre as camadas:** algumas batem com a CPU (ex.: `Mul_120`,
  10% contra 13%) e outras não. Isso indica que camadas quantizadas específicas
  são convertidas incorretamente pelo compilador da NPU.
- **Não depende do tipo de quantização:** quantizado com o preset simétrico
  (`PERFORMANCE`) em vez do `MIXED`, o modelo continua errado na NPU (214 px,
  contra 202 px) e certo na CPU (18 px, contra 11 px).
- A NPU roda corretamente os modelos **FP16** (RTMPose-m e YOLOX-x). O problema
  é específico da execução **INT8** do RTMPose-x nesta NPU e neste driver
  (1005540, OpenVINO 2026.4.0). Não testamos o RTMPose-m em INT8.

- **Não é a versão do OpenVINO:** com o OpenVINO 2026.3.1, a versão validada com
  o driver 5540, o resultado é idêntico. O compilador da NPU fica no driver, e o
  5540 era o driver mais recente da Intel em 30/09/2026.
- **O `quantize_with_accuracy_control()` do NNCF não resolveria sozinho:** ele
  avalia o modelo sempre na CPU (`device_name="CPU"` fixo no código do NNCF), e
  na CPU o INT8 completo funciona. Ele não enxergaria o erro da NPU.

Se um driver futuro corrigir a execução INT8, dá para quantizar mais camadas
com `--camadas` (ou `todas`) e comparar velocidade e precisão. O modelo INT8
completo pode ser gerado de novo com `--camadas todas`. Para reportar o problema
à Intel (repositório do OpenVINO), esse modelo e um recorte de entrada já bastam
para reproduzi-lo.
