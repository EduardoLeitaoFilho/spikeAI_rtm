# Comparativo de desempenho (2026-09-30 16:29)

Tempos de inferência em ms por frame (média após o aquecimento).
Precisão (nme_pct, pck_5pct, eventos): concordância com a referência FP32 `x_cpu`; nme_pct = erro médio em % do tamanho da pessoa (menor = melhor), pck_5pct = % de keypoints a menos de 5% do tamanho da pessoa (maior = melhor).

| config | status | modelo_pose | execucao | precisao_det | precisao_pose | carga_modelos_s | detector_ms | pose_ms | inferencia_ms | inferencia_p95_ms | fps_inferencia | fps_pipeline | taxa_deteccao_pct | confianca_media | nme_pct | pck_5pct | eventos | observacao |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x_cpu | completed | rtmpose_x | onnxruntime/cpu | FP32 | FP32 | 1.571 | 1261.41 | 2999.22 | 4260.63 | 5597.86 | 0.23 | 0.24 | 100.0 | 0.6428 |  |  | referência |  |
| x_gpu_intel | completed | rtmpose_x | openvino/gpu | FP16 | FP16 | 5.25 | 46.54 | 457.55 | 504.09 | 667.18 | 1.98 | 1.89 | 100.0 | 0.6426 | 0.22 | 99.7 | iguais |  |
| x_npu | incompatível | rtmpose_x | openvino/npu |  |  |  |  |  |  |  |  |  |  |  |  |  |  | rtmpose_x (pose) não pôde ser compilado em openvino/npu: Compilation failed |
| m_cpu | completed | rtmpose_m | onnxruntime/cpu | FP32 | FP32 | 1.184 | 1191.09 | 1180.28 | 2371.37 | 2747.69 | 0.42 | 0.42 | 100.0 | 0.609 | 3.11 | 87.3 | iguais |  |
| m_gpu_intel | completed | rtmpose_m | openvino/gpu | FP16 | FP16 | 3.766 | 45.7 | 135.32 | 181.03 | 225.01 | 5.52 | 4.89 | 100.0 | 0.6087 | 3.12 | 87.3 | take_off +1 frames |  |
| m_npu | completed | rtmpose_m | openvino/npu | FP16 | FP16 | 2.186 | 207.97 | 148.35 | 356.32 | 444.12 | 2.81 | 2.73 | 100.0 | 0.6086 | 3.13 | 87.2 | take_off +1 frames |  |
| hibrido_x_npu_gpu | completed | rtmpose_x | det openvino/npu + pose openvino/gpu | FP16 | FP16 | 1.549 | 209.83 | 466.38 | 676.22 | 817.44 | 1.48 | 1.44 | 100.0 | 0.6428 | 0.22 | 99.7 | iguais |  |
