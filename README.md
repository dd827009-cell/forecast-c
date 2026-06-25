# forecast-c

設計 C：**治療條件化 OCT latent 預測** → 厚度 + 積水體積變化（MICCAI/MIA）。

凍結 OCT 表徵上，給「現在 + 治療 + Δt + 病人資訊」預測未來 latent，解碼成厚度µm + 積水。
用 JEPA「方法」（空間+時間），非 V-JEPA「模型」。

## 快速上手
1. 讀 **[CLAUDE.md](CLAUDE.md)**（完整交接：架構、現狀、環境、避坑、setup）。
2. 設計細節：[docs/完整規劃_C.md](docs/完整規劃_C.md)。
3. 程式導覽：[forecast_c/README.md](forecast_c/README.md)。
4. 跨 session 記憶：`memory/`。

## 跑測試（WSL2 + Docker `octcube-dev`）
```bash
docker run --rm -v "$(pwd)":/workspace -w /workspace octcube-dev python -m forecast_c.tests.run_tests
```

## 不在 git 內（需另備，見 CLAUDE.md §6/§9）
- `OCTCubeM-main/`（官方 OCTCube 碼）、`ckpts/`（權重）、`h5_output/`（OCT 資料）、治療 Excel。
