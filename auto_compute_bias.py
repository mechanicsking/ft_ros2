#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


def moving_std(x, win):
    if len(x) < win:
        return np.zeros(len(x))
    out = np.full(len(x), np.nan)
    for i in range(win, len(x)):
        out[i] = np.std(x[i-win:i])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path")
    parser.add_argument("--topic", default="/aidin_ft/wrench_compensated")
    parser.add_argument("--window_sec", type=float, default=2.0)
    parser.add_argument("--force_std_th", type=float, default=0.35)
    parser.add_argument("--torque_std_th", type=float, default=0.03)
    parser.add_argument("--min_segment_sec", type=float, default=1.0)
    args = parser.parse_args()

    bag_path = Path(args.bag_path)

    times = []
    forces = []
    torques = []

    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic == args.topic]

        if not conns:
            print(f"Topic not found: {args.topic}")
            print("Available topics:")
            for c in reader.connections:
                print(" ", c.topic)
            return

        t0 = None

        for conn, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, conn.msgtype)
            t = timestamp * 1e-9

            if t0 is None:
                t0 = t

            t_rel = t - t0
            f = msg.wrench.force
            tau = msg.wrench.torque

            times.append(t_rel)
            forces.append([f.x, f.y, f.z])
            torques.append([tau.x, tau.y, tau.z])

    times = np.array(times)
    forces = np.array(forces)
    torques = np.array(torques)

    if len(times) < 10:
        print("Not enough samples.")
        return

    dt = np.median(np.diff(times))
    hz = 1.0 / dt
    win = max(5, int(args.window_sec * hz))
    min_len = max(5, int(args.min_segment_sec * hz))

    force_norm = np.linalg.norm(forces, axis=1)
    torque_norm = np.linalg.norm(torques, axis=1)

    f_std = moving_std(force_norm, win)
    t_std = moving_std(torque_norm, win)

    static_mask = (f_std < args.force_std_th) & (t_std < args.torque_std_th)
    static_mask[np.isnan(f_std)] = False

    segments = []
    start = None

    for i, ok in enumerate(static_mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            end = i
            if end - start >= min_len:
                segments.append((start, end))
            start = None

    if start is not None:
        end = len(static_mask)
        if end - start >= min_len:
            segments.append((start, end))

    print("==== Bag summary ====")
    print(f"samples: {len(times)}")
    print(f"duration: {times[-1] - times[0]:.3f} s")
    print(f"estimated hz: {hz:.1f}")
    print()

    if not segments:
        print("No static segments found.")
        print("Try larger thresholds, for example:")
        print(f"python3 auto_compute_bias.py {bag_path} --force_std_th 0.8 --torque_std_th 0.08")
        return

    print("==== Static segments found ====")
    selected_indices = []

    for idx, (s, e) in enumerate(segments):
        seg_t0 = times[s]
        seg_t1 = times[e - 1]
        seg_force_mean = forces[s:e].mean(axis=0)
        seg_torque_mean = torques[s:e].mean(axis=0)
        seg_force_std = forces[s:e].std(axis=0)
        seg_torque_std = torques[s:e].std(axis=0)

        selected_indices.extend(range(s, e))

        print(f"[{idx}] {seg_t0:.2f} ~ {seg_t1:.2f} s, n={e-s}")
        print(f"    force mean  = {seg_force_mean}")
        print(f"    force std   = {seg_force_std}")
        print(f"    torque mean = {seg_torque_mean}")
        print(f"    torque std  = {seg_torque_std}")

    selected_indices = np.array(selected_indices, dtype=int)

    force_bias = forces[selected_indices].mean(axis=0)
    torque_bias = torques[selected_indices].mean(axis=0)

    print()
    print("==== Recommended bias parameters ====")
    print("force_bias [N] =", force_bias)
    print("torque_bias [Nm] =", torque_bias)
    print()
    print("Add these to gravity_comp_arkit_node:")
    print(f"-p force_bias_x:={force_bias[0]:.6f} \\")
    print(f"-p force_bias_y:={force_bias[1]:.6f} \\")
    print(f"-p force_bias_z:={force_bias[2]:.6f} \\")
    print(f"-p torque_bias_x:={torque_bias[0]:.6f} \\")
    print(f"-p torque_bias_y:={torque_bias[1]:.6f} \\")
    print(f"-p torque_bias_z:={torque_bias[2]:.6f}")


if __name__ == "__main__":
    main()