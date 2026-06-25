import h5py

with h5py.File('20120612T023409_OD.h5', 'r') as f:
    vol = f['volume']
    D, H, W = vol.shape
    print(f"B-scan 數量: {D}")
    print(f"每張解析度: {H} × {W}")
    print(f"深度解析度: {f.attrs['scale_y_um_per_px']:.2f} µm/pixel")
    print(f"橫向解析度: {f.attrs['scale_x_mm_per_px']*1000:.1f} µm/pixel")
    print(f"B-scan間距: {f.attrs['scale_z_mm_per_bscan']*1000:.0f} µm")