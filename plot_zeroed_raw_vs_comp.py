#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader


def read_wrench_topic(bag_path, topic_name):
    times = []
    data = []

    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == topic_name]

        if not conns:
            raise RuntimeError(f"Topic not found: {topic_name}")

        t0 = None

        for conn, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, conn.msgtype)

            t = timestamp * 1e-9
            if t0 is None:
                t0 = t

            f = msg.wrench.force
            tau = msg.wrench.torque

            times.append(t - t0)
            data.append([
                f.x, f.y, f.z,
                tau.x, tau.y, tau.z,
            ])

    return np.array(times), np.array(data)


def interp_to_time(t_src, y_src, t_ref):
    y_out = np.zeros((len(t_ref), y_src.shape[1]))

    for i in range(y_src.shape[1]):
        y_out[:, i] = np.interp(t_ref, t_src, y_src[:, i])

    return y_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path")
    parser.add_argument("--output", default="zeroed_raw_vs_comp.png")

    # raw를 0 기준으로 맞출 초기 정지구간
    parser.add_argument("--zero_start", type=float, default=2.0)
    parser.add_argument("--zero_end", type=float, default=5.0)

    # 그래프 y축 범위
    parser.add_argument("--force_ylim", type=float, default=8.0)
    parser.add_argument("--torque_ylim", type=float, default=0.5)

    # 선 두께
    parser.add_argument("--linewidth", type=float, default=0.7)

    args = parser.parse_args()

    bag_path = Path(args.bag_path)

    t_raw, raw = read_wrench_topic(bag_path, "/aidin_ft/wrench_raw")
    t_comp, comp = read_wrench_topic(bag_path, "/aidin_ft/wrench_compensated")

    # raw 기준 시간축으로 compensated 보간
    comp_i = interp_to_time(t_comp, comp, t_raw)
    t = t_raw

    # raw를 초기 정지구간 기준으로 zeroing
    zero_mask = (t >= args.zero_start) & (t <= args.zero_end)

    if np.sum(zero_mask) < 10:
        raise RuntimeError(
            "Not enough samples in zeroing interval. "
            "Change --zero_start/--zero_end."
        )

    raw_offset = raw[zero_mask].mean(axis=0)
    raw_zeroed = raw - raw_offset

    print("==== Raw offset used for visualization ====")
    print("force offset [N]  :", raw_offset[:3])
    print("torque offset [Nm]:", raw_offset[3:])

    print()
    print("==== Zeroed raw force mean/std ====")
    print("mean:", raw_zeroed[:, :3].mean(axis=0))
    print("std :", raw_zeroed[:, :3].std(axis=0))

    print()
    print("==== Compensated force mean/std ====")
    print("mean:", comp_i[:, :3].mean(axis=0))
    print("std :", comp_i[:, :3].std(axis=0))

    print()
    print("==== Zeroed raw torque mean/std ====")
    print("mean:", raw_zeroed[:, 3:].mean(axis=0))
    print("std :", raw_zeroed[:, 3:].std(axis=0))

    print()
    print("==== Compensated torque mean/std ====")
    print("mean:", comp_i[:, 3:].mean(axis=0))
    print("std :", comp_i[:, 3:].std(axis=0))

    lw = args.linewidth
    alpha = 0.9

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    # ─────────────────────────────────────
    # w/ Gravity Compensation - Force
    # ─────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t, comp_i[:, 0], label="Force X", linewidth=lw, alpha=alpha)
    ax.plot(t, comp_i[:, 1], label="Force Y", linewidth=lw, alpha=alpha)
    ax.plot(t, comp_i[:, 2], label="Force Z", linewidth=lw, alpha=alpha)
    ax.set_title("w/ Gravity Compensation")
    ax.set_ylabel("Force (N)")
    ax.set_ylim(-args.force_ylim, args.force_ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ─────────────────────────────────────
    # w/o Gravity Compensation - Force
    # raw zeroed at initial pose
    # ─────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t, raw_zeroed[:, 0], label="Force X", linewidth=lw, alpha=alpha)
    ax.plot(t, raw_zeroed[:, 1], label="Force Y", linewidth=lw, alpha=alpha)
    ax.plot(t, raw_zeroed[:, 2], label="Force Z", linewidth=lw, alpha=alpha)
    ax.set_title("w/o Gravity Compensation\n(raw zeroed at initial pose)")
    ax.set_ylabel("Force (N)")
    ax.set_ylim(-args.force_ylim, args.force_ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ─────────────────────────────────────
    # w/ Gravity Compensation - Torque
    # ─────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(t, comp_i[:, 3], label="Torque X", linewidth=lw, alpha=alpha)
    ax.plot(t, comp_i[:, 4], label="Torque Y", linewidth=lw, alpha=alpha)
    ax.plot(t, comp_i[:, 5], label="Torque Z", linewidth=lw, alpha=alpha)
    ax.set_ylabel("Torque (Nm)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-args.torque_ylim, args.torque_ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ─────────────────────────────────────
    # w/o Gravity Compensation - Torque
    # raw zeroed at initial pose
    # ─────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t, raw_zeroed[:, 3], label="Torque X", linewidth=lw, alpha=alpha)
    ax.plot(t, raw_zeroed[:, 4], label="Torque Y", linewidth=lw, alpha=alpha)
    ax.plot(t, raw_zeroed[:, 5], label="Torque Z", linewidth=lw, alpha=alpha)
    ax.set_ylabel("Torque (Nm)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-args.torque_ylim, args.torque_ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print()
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()