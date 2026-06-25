"""Phase 3 介面 stub — SwinUNETR 密集解碼 + 積水體積變化 + REG-1 配準。

鎖契約，實作待最小版 + Phase 1 完成。接**官方代碼**:
  - SwinUNETR = 官方 **MONAI** `monai.networks.nets.SwinUNETR`（需 `pip install monai`）。
  - 積水分割知識 = MedSAM2 蒸餾（Phase 1a）+ 自標 50-100 張校準（真 GT，解循環 D）。
  - REG-1 = 用 M7b 的 `ascan_pos_ir` + `rpe_bm_y`，baseline IR ↔ 回診 IR 2D 配準（血管）。

完整規劃_C §6/§7。厚度=主（真 GT）、積水=次（pseudo-label，報準度落自標真 GT 子集）。
"""


class SwinUNETRDecoder:
    """官方 MONAI SwinUNETR 多尺度 3D 解碼 → 厚度µm + 積水分割。"""

    def __init__(self, img_size, in_channels, out_channels, feature_size: int = 48):
        self.img_size, self.in_channels, self.out_channels = img_size, in_channels, out_channels
        self.feature_size = feature_size

    def build(self, *args, **kwargs):
        raise NotImplementedError(
            "Phase 3 接點: `from monai.networks.nets import SwinUNETR`（pip install monai）。"
            "多尺度來源見規格 O: 階層式學生 + ScaleKD / plain ViT + ViT-Adapter / 最小版簡單解碼。")


def fluid_volume_change(seg_this, seg_last, *args, **kwargs):
    """積水體積變化 = 這次 − 上次（真實 25 切片 slab @512 加總 × spacing）。

    前提: REG-1 配準 + 對的 spacing。系統取樣誤差在變化量抵消。
    """
    raise NotImplementedError(
        "Phase 3 接點: 每次回診分割→真實 25 切片體積加總（× voxel spacing）→ 這次−上次。"
        "需 REG-1 配準（沒變區域殘差→0 才信變化訊號）+ 自標校準（積水 decoding ceiling）。")


class REG1Registration:
    """REG-1: baseline ↔ 回診 2D 配準（用 ascan_pos_ir / rpe_bm_y），QC 閘門。"""

    def __init__(self, qc_residual_thresh: float = 1.0):
        self.qc_residual_thresh = qc_residual_thresh

    def register(self, *args, **kwargs):
        raise NotImplementedError(
            "Phase 3 接點: ① rpe_bm_y 沿 RPE 拉平去軸向 ② baseline IR↔回診 IR 2D 配準（血管）"
            "③ ascan_pos_ir 傳位移到 B-scan。QC: 沒變區域殘差→0，沒過不進變化訊號。")
