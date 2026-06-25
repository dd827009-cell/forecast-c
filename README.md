# forecast-c


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
