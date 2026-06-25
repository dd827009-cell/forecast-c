"""Stage 0 前處理版本字串 (單一真相來源)。

凡產出的 manifest / norm_stats / shard meta 都記這個版本, 之後改配方就 bump,
讓不同批次資料可被區分、追溯。
"""
# M2 強度轉換配方: sqrt -> 非零值 P15 地板 -> percentile[1,99.9] -> [0,1]
TRANSFORM_VERSION = "m2v1_sqrt_p15floor_p1_99.9"

# M3 幾何: OS翻轉 + FOV置中裁512 + resize256(INTER_AREA);
#          層方案C(native不resize) / pos方案A(raw IR) / mask 256+512
GEOM_VERSION = "m3v1_flip_crop512_resize256_layersC_posA"

# 整體 Stage 0 前處理版本 (供 manifest.transform_version 欄位)
STAGE0_VERSION = TRANSFORM_VERSION + "__" + GEOM_VERSION
