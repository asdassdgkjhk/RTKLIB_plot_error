#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTKLIB SPP 误差项绘图脚本（中文图注，仅 GPS 场景）

统一符号约定：观测值 = 几何距离 + 各误差项 (等号右边)
    P_raw = rho + rcv_clk + sat_clk_corr + sat_rel_corr + tgd_corr
            + iono_corr + trop_corr + v_residual
其中 P_raw = P + tgd_corr (原始伪距，未扣 TGD)

CSV 列说明:
    time           : 历元时间
    sat            : 卫星编号 (G10/G12...)
    sys            : 卫星系统字符 (G/R/E/C/J/I/S)
    el_deg/az_deg  : 高度角 / 方位角 (deg)
    P              : 伪距观测值 (m, prange() 返回, 已扣 TGD)
    rho            : 卫星-接收机几何距离 (m)
    sat_clk_corr   : 纯卫星钟差改正 (m, 已剔除相对论)
    sat_rel_corr   : 相对论效应改正 (m)
    tgd_corr       : TGD 码偏差改正 (m)
    iono_corr      : 电离层延迟 (m)
    trop_corr      : 对流层延迟 (m)
    rcv_clk        : GPS 接收机钟差 (m)
    v_residual     : 伪距残差 (m)

用法:
    python plot_spp_errors.py spp_error_terms.csv --mode all_sats   --col iono_corr
    python plot_spp_errors.py spp_error_terms.csv --mode by_sat     --sat G10 --col sat_clk_corr
    python plot_spp_errors.py spp_error_terms.csv --mode rcv_clock
    python plot_spp_errors.py spp_error_terms.csv --mode residual
"""
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# 中文字体设置（Windows 中文环境）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 系统字符 -> 中文名 + 颜色
SYS_INFO = {
    'G': ('GPS',     'C0'),
    'R': ('GLONASS', 'C1'),
    'E': ('Galileo', 'C2'),
    'C': ('BDS',     'C3'),
    'J': ('QZSS',    'C4'),
    'I': ('NavIC',   'C5'),
    'S': ('SBAS',    'C6'),
    '?': ('未知',    'k'),
}

# 列名 -> 中文显示
COL_CN = {
    'P':              '伪距观测值 P (m)',
    'rho':            '几何距离 ρ (m)',
    'sat_clk_corr':   '卫星钟差改正 (m)',
    'sat_rel_corr':   '相对论效应改正 (m)',
    'tgd_corr':       'TGD 码偏差改正 (m)',
    'iono_corr':      '电离层延迟 (m)',
    'trop_corr':      '对流层延迟 (m)',
    'rcv_clk':        'GPS 接收机钟差 (m)',
    'v_residual':     '伪距残差 (m)',
    'el_deg':         '高度角 (°)',
    'az_deg':         '方位角 (°)',
}


def col_cn(col):
    return COL_CN.get(col, col)


def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    return df


def plot_error_by_sat(csv_file, sat_id="G10", error_col="iono_corr", out=None):
    """绘制指定卫星的某项误差随时间变化"""
    df = load_csv(csv_file)
    sub = df[df['sat'] == sat_id].sort_values('time')
    if sub.empty:
        print(f"[警告] 没有卫星 {sat_id} 的数据")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub['time'], sub[error_col], '-o', markersize=3,
            label=sat_id, color='C0')
    ax.set_title(f"{col_cn(error_col)}   卫星 {sat_id}")
    ax.set_xlabel("时间 (GPST)")
    ax.set_ylabel(col_cn(error_col))
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    fig.tight_layout()
    out = out or f"{sat_id}_{error_col}.png"
    fig.savefig(out, dpi=150)
    print(f"[完成] 已保存 {out}")


def plot_error_all_sats(csv_file, error_col="trop_corr", out=None):
    """绘制所有卫星某项误差随时间变化（每颗卫星独立颜色+线型+标记）
    时间不连续的段之间不连线；数据点缩小为原来 1/2。"""
    df = load_csv(csv_file).sort_values('time')

    # 卫星编号排序，确保图例顺序稳定
    sat_list = sorted(df['sat'].unique(),
                      key=lambda s: (s[0], int(s[1:])))

    # 颜色循环 + 线型循环 + 标记循环，组合保证每颗卫星视觉可区分
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', '>', '<', 'P', '*', 'X', 'h', 'd']

    # 历元间隔阈值：超过此间隔视为数据断段，不连线
    # 默认取数据中位间隔的 5 倍作为阈值
    if len(df) >= 2:
        median_dt = df['time'].sort_values().diff().median()
        gap_threshold = median_dt * 5 if pd.notna(median_dt) else pd.Timedelta(days=1)
    else:
        gap_threshold = pd.Timedelta(days=1)

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, sat in enumerate(sat_list):
        g = df[df['sat'] == sat]
        if g.empty:
            continue
        sys_char = g['sys'].iloc[0]
        sys_name, _ = SYS_INFO.get(sys_char, ('未知', 'k'))
        c = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]
        mk = markers[idx % len(markers)]

        # 按时间排序，按间隔切分为多个连续段
        g_sorted = g.sort_values('time')
        times = g_sorted['time'].values
        values = g_sorted[error_col].values

        # 计算相邻时间差，分段
        if len(times) >= 2:
            dts = np.diff(times).astype('timedelta64[ns]')
            seg_breaks = np.where(dts > np.timedelta64(gap_threshold))[0]
        else:
            seg_breaks = []

        seg_starts = np.concatenate([[0], seg_breaks + 1])
        seg_ends = np.concatenate([seg_breaks + 1, [len(times)]])

        first_seg = True
        for s, e in zip(seg_starts, seg_ends):
            seg_t = times[s:e]
            seg_v = values[s:e]
            # 只在第一段画图例
            lbl = f"{sat} ({sys_name})" if first_seg else None
            ax.plot(seg_t, seg_v,
                    linestyle=ls, marker=mk, markersize=2,
                    linewidth=1.2, color=c, alpha=0.85,
                    label=lbl)
            first_seg = False

    ax.set_title(f"{col_cn(error_col)}   所有卫星（每颗卫星独立图线）")
    ax.set_xlabel("时间 (GPST)")
    ax.set_ylabel(col_cn(error_col))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=7, loc='upper right', framealpha=0.85)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    fig.tight_layout()
    out = out or f"all_{error_col}.png"
    fig.savefig(out, dpi=150)
    print(f"[完成] 已保存 {out}")


def plot_receiver_clock(csv_file, out=None):
    """绘制 GPS 接收机钟差随时间"""
    df = load_csv(csv_file).sort_values('time')
    rcv = df.groupby('time')['rcv_clk'].first()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(rcv.index, rcv.values, '-o', markersize=3, color='red',
            label='GPS 接收机钟差')
    ax.set_title("GPS 接收机钟差")
    ax.set_xlabel("时间 (GPST)")
    ax.set_ylabel("钟差 (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    fig.tight_layout()
    out = out or "接收机钟差.png"
    fig.savefig(out, dpi=150)
    print(f"[完成] 已保存 {out}")


def plot_residual_all(csv_file, out=None):
    """所有卫星伪距残差散点图（每颗卫星独立颜色+标记，点缩小为 1/2）"""
    df = load_csv(csv_file).sort_values('time')

    sat_list = sorted(df['sat'].unique(),
                      key=lambda s: (s[0], int(s[1:])))
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    markers = ['o', 's', '^', 'v', 'D', '>', '<', 'P', '*', 'X', 'h', 'd']

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, sat in enumerate(sat_list):
        g = df[df['sat'] == sat]
        if g.empty:
            continue
        c = colors[idx % len(colors)]
        mk = markers[idx % len(markers)]
        ax.plot(g['time'], g['v_residual'],
                marker=mk, markersize=2, linestyle='',
                color=c, alpha=0.75, label=sat)

    ax.set_title("伪距残差（所有卫星，每颗卫星独立图线）")
    ax.set_xlabel("时间 (GPST)")
    ax.set_ylabel(col_cn('v_residual'))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=7, loc='upper right', framealpha=0.85)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    fig.tight_layout()
    out = out or "伪距残差_所有卫星.png"
    fig.savefig(out, dpi=150)
    print(f"[完成] 已保存 {out}")


def plot_all(csv_file):
    """一键绘制所有误差项图"""
    # 各卫星误差项
    for col in ["sat_clk_corr", "sat_rel_corr", "tgd_corr",
                "iono_corr", "trop_corr"]:
        plot_error_all_sats(csv_file, col)
    # 接收机钟差
    plot_receiver_clock(csv_file)
    # 伪距残差
    plot_residual_all(csv_file)
    print("\n[全部完成] 共生成 7 张图片")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RTKLIB SPP 误差项绘图脚本")
    p.add_argument("csv", nargs="?", default="spp_error_terms.csv",
                   help="spp_error_terms.csv 路径 (默认: spp_error_terms.csv)")
    p.add_argument("--mode", choices=[
        "by_sat", "all_sats", "rcv_clock", "residual", "all"],
        default="all", help="绘图模式 (默认: all 画全部)")
    p.add_argument("--sat", default="G10", help="卫星编号, 如 G10")
    p.add_argument("--col", default="iono_corr",
                   help="误差列名, 如 sat_clk_corr/sat_rel_corr/tgd_corr/"
                        "iono_corr/trop_corr/v_residual")
    p.add_argument("--out", default=None, help="输出 PNG 路径")
    args = p.parse_args()

    if args.mode == "all":
        plot_all(args.csv)
    elif args.mode == "by_sat":
        plot_error_by_sat(args.csv, args.sat, args.col, args.out)
    elif args.mode == "all_sats":
        plot_error_all_sats(args.csv, args.col, args.out)
    elif args.mode == "rcv_clock":
        plot_receiver_clock(args.csv, args.out)
    elif args.mode == "residual":
        plot_residual_all(args.csv, args.out)
