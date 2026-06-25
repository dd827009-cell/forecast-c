"""
Stage 0 - Module 3: 幾何處理 (OS翻轉 + FOV置中裁切512 + resize256)。
只處理幾何, 不碰強度 (強度是 Module 2)。資料流: raw -> M2轉換 -> M3幾何。

決策:
  - OS 水平翻轉 (沿 W) 對齊 OD
  - FOV 置中裁切橫向到 512px (768置中裁中央512=黃斑區); 全資料 lateral=11.25um/px 固定 -> 物理尺度一致
  - resize (D,496,512) -> (D,256,256), 逐切片 cv2.INTER_AREA
  - 層標註 ilm_y/rpe_bm_y 同步翻轉/裁切/resize (座標換算)

凡與 OCT 共用 W 軸的附帶陣列, 一律做「跟 volume 完全相同」的 flip + crop:
  - valid_ascan_mask (D,W) bool:
      flip+crop 後得 native 512 版 (provenance), 再以 all-valid 規則降採到 256
      (256 的每欄是 2 條 A-scan 經 INTER_AREA 面積平均, 2 條皆有效才算有效) 對齊 volume。
  - ascan_pos_ir (D,W,2) float32 (方案A, raw IR 座標):
      只對 W 軸 (axis=1) flip+crop, 「數值不動」。座標活在 IR 768x768 像素空間,
      與 OCT resize 到 256 無關; OCT<->IR 對應靠這張顯式查表, 翻不翻 IR 都正確。
      日後 COEP 若要 IR-aligned 座標, 一條公式 x'=(IR_W-1)-x 即可從 raw 推出。
  - ir 影像本身 M3 不動 (raw 保留), 由 M7 連同 laterality / pos_frame='raw_ir_unflipped' 一起打包。

用法 (驗證):
    python stage0/m3_geometry.py
"""
import numpy as np
import cv2

TARGET_FOV_W = 512   # 共同 FOV 像素寬
OUT_SIZE = 256


def _crop_or_pad_w(arr, target_w, axis=-1):
    """對指定軸置中裁切或補零到 target_w。回傳 (新陣列, crop_start)。

    axis 預設 -1 (vol/層的 W 在最後一軸); pos 的 W 在 axis=1, 需顯式指定。
    """
    W = arr.shape[axis]
    if W == target_w:
        return arr, 0
    if W > target_w:
        start = (W - target_w) // 2
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(start, start + target_w)
        return arr[tuple(sl)], start
    pad = target_w - W
    left = pad // 2
    widths = [(0, 0)] * arr.ndim
    widths[axis] = (left, pad - left)
    return np.pad(arr, widths), -left


def _downsample_mask_all_valid(mask_w, out_size):
    """(D,Wc) bool -> (D,out_size) bool, all-valid 規則。
    out 欄 i 對應來源 [round(i*s), round((i+1)*s)) 全有效才算有效 (s=Wc/out)。"""
    D, Wc = mask_w.shape
    s = Wc / out_size
    out = np.zeros((D, out_size), dtype=bool)
    for i in range(out_size):
        a = int(round(i * s))
        b = max(int(round((i + 1) * s)), a + 1)
        out[:, i] = mask_w[:, a:b].all(axis=1)
    return out


def process_volume(vol01, laterality, ilm=None, rpe=None, valid_mask=None, pos=None,
                   target_fov_w=TARGET_FOV_W, out_size=OUT_SIZE):
    """
    參數:
        vol01: (D,H,W) float32 [0,1] (已 Module2 轉換)
        laterality: 'OD' / 'OS'
        ilm, rpe: (D,W) 層 y 座標 (px) 或 None
        valid_mask: (D,W) bool 每條 A-scan 有效性 或 None
        pos: (D,W,2) float32 ascan_pos_ir (IR 像素座標) 或 None
    回傳 dict:
        volume: (D,out,out) float32 [0,1]
        ilm, rpe: (D,target_fov_w) float32 或 None  (方案C: native 496列空間/512欄, 不 resize)
        ilm_valid, rpe_valid: (D,target_fov_w) bool 或 None  (= isfinite)
        valid_mask_256: (D,out) bool 或 None  (all-valid 降採, 對齊 volume)
        valid_mask_512: (D,target_fov_w) bool 或 None  (flip+crop native, provenance)
        ascan_pos_ir: (D,target_fov_w,2) float32 或 None  (方案A: flip+crop, raw 數值不動)
    """
    D, H, W = vol01.shape
    is_os = (str(laterality) == 'OS')

    # 1) OS 翻轉 (沿 W)。注意用 ascontiguousarray 避免負步長
    if is_os:
        vol01 = np.ascontiguousarray(vol01[:, :, ::-1])
        if ilm is not None:
            ilm = np.ascontiguousarray(ilm[:, ::-1])
        if rpe is not None:
            rpe = np.ascontiguousarray(rpe[:, ::-1])
        if valid_mask is not None:
            valid_mask = np.ascontiguousarray(valid_mask[:, ::-1])
        if pos is not None:
            pos = np.ascontiguousarray(pos[:, ::-1, :])   # 只翻 W 軸(axis=1), 數值不動

    # 2) FOV 置中裁切到 target_fov_w
    vol01, _ = _crop_or_pad_w(vol01, target_fov_w)
    if ilm is not None:
        ilm, _ = _crop_or_pad_w(ilm, target_fov_w)
    if rpe is not None:
        rpe, _ = _crop_or_pad_w(rpe, target_fov_w)
    if valid_mask is not None:
        valid_mask, _ = _crop_or_pad_w(valid_mask, target_fov_w)
    if pos is not None:
        pos, _ = _crop_or_pad_w(pos, target_fov_w, axis=1)
    Wc = target_fov_w

    # 3) resize (H,Wc) -> (out,out), 逐切片
    out_vol = np.empty((D, out_size, out_size), dtype=np.float32)
    for d in range(D):
        out_vol[d] = cv2.resize(vol01[d], (out_size, out_size),
                                interpolation=cv2.INTER_AREA)
    out_vol = np.clip(out_vol, 0.0, 1.0)

    # 層座標 (方案C, native): 只做 flip+crop (上方已完成), 不 resize。
    #   - y 保留 native H(496) 列空間、x 保留 target_fov_w(512) 欄。
    #   - thickness µm = (rpe - ilm) * axial_um_per_px (全域常數 3.87167) 一步到位, 零 resize 耦合。
    #   - 保留 NaN (分割失敗欄位不填假值), valid = isfinite; 疊到 256 volume 時於 viz 端即時換算。
    # 層不過 resize -> 完全不涉及 cv2 pixel-center 慣例; 橫向保留 512 全取樣。
    res = {"volume": out_vol}
    for name, lay in [("ilm", ilm), ("rpe", rpe)]:
        if lay is None:
            res[name] = None
            res[name + "_valid"] = None
            continue
        lay = lay.astype(np.float32)               # (D, Wc) native, 已 flip+crop
        res[name] = lay
        res[name + "_valid"] = np.isfinite(lay)

    # valid_ascan_mask: native 512 (provenance) + all-valid 降採 256 (對齊 volume)
    if valid_mask is not None:
        vm512 = valid_mask.astype(bool)
        res["valid_mask_512"] = vm512
        res["valid_mask_256"] = _downsample_mask_all_valid(vm512, out_size)
    else:
        res["valid_mask_512"] = None
        res["valid_mask_256"] = None

    # ascan_pos_ir: 方案A, flip+crop 後 raw IR 座標 (數值不動, 仍在 IR 768x768 空間)
    res["ascan_pos_ir"] = pos.astype(np.float32) if pos is not None else None
    return res


# ============================ 驗證 ============================
def _verify():
    import sys, io, glob
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    import h5py
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, "stage0")
    from m2_transform import transform_volume

    df = pd.read_parquet("stage0/index.parquet")

    def load(p):
        with h5py.File(p, "r") as f:
            vol = f["volume"][:].astype(np.float32)
            valid = f["valid_ascan_mask"][:]
            ilm = f["ilm_y"][:].astype(np.float32)
            rpe = f["rpe_bm_y"][:].astype(np.float32)
            pos = f["ascan_pos_ir"][:].astype(np.float32)
            lat = f.attrs["laterality"]
        return transform_volume(vol, valid), lat, ilm, rpe, valid, pos

    # 找同病人的一組 OD+OS, 以及那顆 768
    pid = df[df["laterality"] == "OS"].iloc[0]["patient_id"]
    od_p = df[(df.patient_id == pid) & (df.laterality == "OD")].iloc[0]["h5_path"]
    os_p = df[(df.patient_id == pid) & (df.laterality == "OS")].iloc[0]["h5_path"]
    w768_p = df[df.vol_w == 768].iloc[0]["h5_path"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    # --- 列0: OD vs OS(原始,未翻) vs OS(翻轉+處理) -> 驗證翻轉 ---
    v_od, lat_od, ilm_od, rpe_od, vm_od, pos_od = load(od_p)
    v_os, lat_os, ilm_os, rpe_os, vm_os, pos_os = load(os_p)
    r_od = process_volume(v_od, lat_od, ilm_od, rpe_od, vm_od, pos_od)
    r_os = process_volume(v_os, lat_os, ilm_os, rpe_os, vm_os, pos_os)
    mid = r_od["volume"].shape[0] // 2
    axes[0, 0].imshow(r_od["volume"][mid], cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[0, 0].set_title(f"OD processed (pid={pid})"); axes[0, 0].axis("off")
    axes[0, 1].imshow(v_os[v_os.shape[0]//2], cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[0, 1].set_title("OS BEFORE flip (mirror of OD)"); axes[0, 1].axis("off")
    axes[0, 2].imshow(r_os["volume"][mid], cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[0, 2].set_title("OS AFTER flip+process (should orient like OD)"); axes[0, 2].axis("off")

    # --- 列1: 768 裁切+resize ---
    v8, lat8, ilm8, rpe8, vm8, pos8 = load(w768_p)
    r8 = process_volume(v8, lat8, ilm8, rpe8, vm8, pos8)
    m8 = v8.shape[0] // 2
    axes[1, 0].imshow(v8[m8], cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[1, 0].set_title(f"768-wide BEFORE (shape {v8.shape[1]}x{v8.shape[2]})"); axes[1, 0].axis("off")
    axes[1, 1].imshow(r8["volume"][m8], cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[1, 1].set_title(f"AFTER crop512+resize (shape {r8['volume'].shape[1]}x{r8['volume'].shape[2]})"); axes[1, 1].axis("off")
    axes[1, 2].axis("off")

    # --- 列2: 層標註疊圖 (層為 native 496列/512欄, viz端即時換算到256, pixel-center) ---
    H_NATIVE = 496
    for col, (r, tag) in enumerate([(r_od, "OD"), (r_os, "OS flipped"), (r8, "768 cropped")]):
        ax = axes[2, col]
        m = r["volume"].shape[0] // 2
        ax.imshow(r["volume"][m], cmap="gray", vmin=0, vmax=1, aspect="auto")
        Wc_lay = r["ilm"].shape[1]
        x256 = (np.arange(Wc_lay) + 0.5) * (OUT_SIZE / Wc_lay) - 0.5
        ilm256 = (r["ilm"][m] + 0.5) * (OUT_SIZE / H_NATIVE) - 0.5
        rpe256 = (r["rpe"][m] + 0.5) * (OUT_SIZE / H_NATIVE) - 0.5
        ax.plot(x256, ilm256, color="lime", lw=1.0, label="ILM")
        ax.plot(x256, rpe256, color="red", lw=1.0, label="RPE_BM")
        ax.set_title(f"{tag}: native layers -> 256 (viz)"); ax.set_xlim(0, OUT_SIZE); ax.set_ylim(OUT_SIZE, 0)
        ax.legend(fontsize=8); ax.axis("on")

    fig.suptitle("Module 3 驗證: 列0=OS翻轉, 列1=768裁切, 列2=層座標換算疊圖", fontsize=14)
    fig.tight_layout()
    fig.savefig("stage0/m3_verify.png", dpi=100, bbox_inches="tight")

    # 數值驗證 (NaN-aware)
    print("=== 數值驗證 ===")
    print(f"OD 輸出 shape: {r_od['volume'].shape}  值域[{r_od['volume'].min():.2f},{r_od['volume'].max():.2f}]")
    print(f"OS 輸出 shape: {r_os['volume'].shape}")
    print(f"768 輸出 shape: {r8['volume'].shape}  (應為 (25,256,256))")
    print(f"層 shape: ilm={r_od['ilm'].shape} (應為 (25,512) native, 非 256)")
    print(f"層座標 ilm 範圍: [{np.nanmin(r_od['ilm']):.1f}, {np.nanmax(r_od['ilm']):.1f}] (native, 應落在 0~496)")
    print(f"層座標 rpe 範圍: [{np.nanmin(r_od['rpe']):.1f}, {np.nanmax(r_od['rpe']):.1f}] (native, 0~496 且 rpe>ilm)")
    # ILM<RPE 只在兩者皆有效的欄位上檢查
    both = r_od['ilm_valid'][mid] & r_od['rpe_valid'][mid]
    ok = (r_od['ilm'][mid][both] < r_od['rpe'][mid][both])
    print(f"ILM 在 RPE 上方(y較小)? {ok.mean()*100:.0f}% (僅在 {both.sum()} 個雙有效欄位上檢查)")
    # thickness µm 一步到位驗證 (native px * axial 3.87167)
    AXIAL_UM = 3.87167
    th_um = (r_od['rpe'][mid][both] - r_od['ilm'][mid][both]) * AXIAL_UM
    print(f"thickness µm (mid B-scan, ILM-RPE): mean={th_um.mean():.1f} range=[{th_um.min():.1f},{th_um.max():.1f}]"
          f"  (= diff_native_px × 3.87167, 零 resize 耦合)")
    print(f"層有效遮罩比例: OD ilm_valid={r_od['ilm_valid'].mean()*100:.1f}% rpe_valid={r_od['rpe_valid'].mean()*100:.1f}%")
    print(f"NaN 已保留: ilm NaN數={np.isnan(r_od['ilm']).sum()} (= 無效欄位數)")

    # --- valid_ascan_mask 驗證 ---
    print("\n=== valid_ascan_mask 驗證 ===")
    print(f"OD mask512 shape={r_od['valid_mask_512'].shape} 有效率={r_od['valid_mask_512'].mean()*100:.2f}%")
    print(f"OD mask256 shape={r_od['valid_mask_256'].shape} 有效率={r_od['valid_mask_256'].mean()*100:.2f}%"
          f"  (all-valid 降採, 應 <= 512 版)")
    assert r_od['valid_mask_256'].mean() <= r_od['valid_mask_512'].mean() + 1e-9, "256 有效率不應高於 512"
    print(f"768 mask256 shape={r8['valid_mask_256'].shape} (應為 (25,256))")

    # --- ascan_pos_ir 驗證 (方案A: flip 後仍指向同一 IR 像素, 數值不變) ---
    print("\n=== ascan_pos_ir 驗證 (方案A) ===")
    print(f"OD pos shape={r_od['ascan_pos_ir'].shape} (應為 (25,512,2))")
    # OS: 原始最右欄(crop後) 的 pos, 應等於翻轉+crop 後最左欄的 pos
    d = pos_os.shape[0] // 2
    # 取 OS crop 前後對照: r_os 已 flip+crop; 直接驗證 raw-vs-processed 的數值集合一致 (僅 reindex/crop)
    raw_set = np.sort(pos_os[d, :, 0])
    proc_set = np.sort(r_os['ascan_pos_ir'][d, :, 0])
    # crop 會砍掉左右各 (W-512)/2; 512 寬則完全相同
    if pos_os.shape[1] == TARGET_FOV_W:
        same_vals = np.array_equal(raw_set, proc_set)
        print(f"OS(512寬) pos 數值集合 flip+crop 前後相同(未被改動)? {same_vals}")
    print(f"OS pos x 範圍=[{r_os['ascan_pos_ir'][...,0].min():.0f},{r_os['ascan_pos_ir'][...,0].max():.0f}]"
          f"  (仍在 IR 768 像素空間, 未被 resize 影響)")
    print("\n驗證圖已存: stage0/m3_verify.png")


if __name__ == "__main__":
    _verify()
