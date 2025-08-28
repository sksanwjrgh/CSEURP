import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
import re

# ===== 경로 설정 =====
base   = Path(r'C:\Users\난무적호\Desktop\CSEURP\final')
u_dir  = base / 'u'
v_dir  = base / 'v'
assert u_dir.is_dir() and v_dir.is_dir(), "final/u, final/v 폴더를 확인하세요."

# ===== 파일명에서 시간 추출 (stem 안의 첫 숫자) =====
num_re = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
def t_from(stem: str):
    m = num_re.search(stem)
    return float(m.group(0)) if m else None

# u, v 각각 dict[time] = path
u_dict = {round(t_from(p.stem), 2): p for p in u_dir.glob('*.npy') if t_from(p.stem) is not None}
v_dict = {round(t_from(p.stem), 2): p for p in v_dir.glob('*.npy') if t_from(p.stem) is not None}

# 공통 시간만 사용(이미 0.01 간격으로 저장되어 있다고 했으니 그대로 씀)
times = sorted(set(u_dict) & set(v_dict))
if not times:
    raise FileNotFoundError("u, v 폴더에서 공통 시간이 없습니다.")

# ===== 첫 프레임 로드 & 그리드 =====
t0 = times[0]
U0 = np.load(u_dict[t0]); V0 = np.load(v_dict[t0])
ny, nx = U0.shape
x = np.linspace(0, 1.0, nx); y = np.linspace(0, 1.0, ny)
spd0 = np.hypot(U0, V0)

# (선택) 메모리 여유 있으면 미리 다 로드하면 100fps 훨씬 안정적
PRELOAD = True
if PRELOAD:
    U_all = [np.load(u_dict[t]) for t in times]
    V_all = [np.load(v_dict[t]) for t in times]
    # 컬러 스케일 고정 (첫 프레임이 너무 작으면 전체 최대값으로)
    vmax = max(spd0.max(), max(np.hypot(U_all[i], V_all[i]).max() for i in range(len(times))))
else:
    U_all = V_all = None
    vmax = spd0.max()

# ===== 플롯 셋업 (imshow + quiver: streamplot보다 빠름) =====
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(spd0, origin='lower', extent=[0,1,0,1], cmap='viridis',
               vmin=0, vmax=vmax)
cb = plt.colorbar(im, ax=ax, label='|u|')

# 화살표 다운샘플로 속도↑ (값 클수록 화살표 수 ↓)
step = max(1, nx // 32)
Q = ax.quiver(x[::step], y[::step], U0[::step, ::step], V0[::step, ::step],
              color='white', scale=None)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
title = ax.set_title(f't = {t0:.2f} s')

# ===== 업데이트 함수 =====
def update(k):
    if PRELOAD:
        U = U_all[k]; V = V_all[k]
    else:
        t = times[k]
        U = np.load(u_dict[t]); V = np.load(v_dict[t])

    spd = np.hypot(U, V)
    im.set_data(spd)
    Q.set_UVC(U[::step, ::step], V[::step, ::step])
    title.set_text(f't = {times[k]:.2f} s')
    return [im, Q]

# ===== 애니메이션 (100 fps) =====
ani = FuncAnimation(fig, update, frames=len(times), interval=1, blit=False, repeat=True)

# (선택) mp4 저장 — ffmpeg 필요
#ani.save('cavity_100fps.mp4', writer=FFMpegWriter(fps=100, bitrate=5000))

plt.tight_layout()
plt.show()
