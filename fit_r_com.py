#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


RAW_TOPIC = "/aidin_ft/wrench_raw"
GRAV_TOPIC = "/aidin_ft/gravity_wrench"


def read_wrench_topic(bag_path, topic_name):
    times = []
    force = []
    torque = []

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
            force.append([f.x, f.y, f.z])
            torque.append([tau.x, tau.y, tau.z])

    return np.array(times), np.array(force), np.array(torque)


def interp_to_time(t_src, y_src, t_ref):
    y_out = np.zeros((len(t_ref), y_src.shape[1]))
    for i in range(y_src.shape[1]):
        y_out[:, i] = np.interp(t_ref, t_src, y_src[:, i])
    return y_out


def moving_std_vec(x, win):
    out = np.full(len(x), np.nan)
    for i in range(win, len(x)):
        chunk = x[i - win:i]
        out[i] = np.linalg.norm(chunk.std(axis=0))
    return out


def fit_r_com(Fg, tau_raw):
    """
    Fit:
        tau_raw = torque_bias + r_com x Fg

    unknown:
        [rx, ry, rz, bx, by, bz]
    """

    A_rows = []
    y_rows = []

    for F, tau in zip(Fg, tau_raw):
        Fx, Fy, Fz = F
        tx, ty, tz = tau

        # tau_x = by? no:
        # r x F = [ ry*Fz - rz*Fy,
        #           rz*Fx - rx*Fz,
        #           rx*Fy - ry*Fx ]
        A_rows.append([0.0, Fz, -Fy, 1.0, 0.0, 0.0])
        y_rows.append(tx)

        A_rows.append([-Fz, 0.0, Fx, 0.0, 1.0, 0.0])
        y_rows.append(ty)

        A_rows.append([Fy, -Fx, 0.0, 0.0, 0.0, 1.0])
        y_rows.append(tz)

    A = np.array(A_rows)
    y = np.array(y_rows)

    sol, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

    r_com = sol[:3]
    torque_bias = sol[3:]

    tau_pred = np.zeros_like(tau_raw)
    for i, F in enumerate(Fg):
        tau_pred[i] = torque_bias + np.cross(r_com, F)

    residual = tau_raw - tau_pred

    return r_com, torque_bias, residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path")
    parser.add_argument("--window_sec", type=float, default=1.0)

    # static segment detection thresholds
    parser.add_argument("--grav_std_th", type=float, default=0.20,
                        help="moving std threshold of gravity force vector [N]")
    parser.add_argument("--tau_std_th", type=float, default=0.035,
                        help="moving std threshold of raw torque vector [Nm]")
    parser.add_argument("--min_segment_sec", type=float, default=1.0)

    args = parser.parse_args()

    bag_path = Path(args.bag_path)

    t_raw, _, tau_raw = read_wrench_topic(bag_path, RAW_TOPIC)
    t_grav, Fg, _ = read_wrench_topic(bag_path, GRAV_TOPIC)

    Fg_i = interp_to_time(t_grav, Fg, t_raw)
    t = t_raw

    dt = np.median(np.diff(t))
    hz = 1.0 / dt
    win = max(5, int(args.window_sec * hz))
    min_len = max(5, int(args.min_segment_sec * hz))

    grav_std = moving_std_vec(Fg_i, win)
    tau_std = moving_std_vec(tau_raw, win)

    static_mask = (
        (grav_std < args.grav_std_th)
        & (tau_std < args.tau_std_th)
    )
    static_mask[np.isnan(grav_std)] = False

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
    print(f"samples: {len(t)}")
    print(f"duration: {t[-1] - t[0]:.3f} s")
    print(f"estimated hz: {hz:.1f}")
    print()

    if not segments:
        print("No static segments found.")
        print("Try looser thresholds, e.g.:")
        print(f"python3 fit_r_com.py {bag_path} --grav_std_th 0.4 --tau_std_th 0.08")
        return

    selected_indices = []

    print("==== Static segments used ====")
    for idx, (s, e) in enumerate(segments):
        selected_indices.extend(range(s, e))

        print(f"[{idx}] {t[s]:.2f} ~ {t[e-1]:.2f} s, n={e-s}")
        print(f"    gravity force mean = {Fg_i[s:e].mean(axis=0)}")
        print(f"    raw torque mean    = {tau_raw[s:e].mean(axis=0)}")
        print(f"    raw torque std     = {tau_raw[s:e].std(axis=0)}")

    selected_indices = np.array(selected_indices, dtype=int)

    Fg_sel = Fg_i[selected_indices]
    tau_sel = tau_raw[selected_indices]

    r_com, torque_bias, residual = fit_r_com(Fg_sel, tau_sel)

    print()
    print("==== Estimated r_com and torque_bias ====")
    print("r_com [m] =", r_com)
    print("r_com [mm] =", r_com * 1000.0)
    print("torque_bias [Nm] =", torque_bias)

    print()
    print("==== Torque residual after fitting ====")
    print("mean [Nm] =", residual.mean(axis=0))
    print("std  [Nm] =", residual.std(axis=0))
    print("max abs [Nm] =", np.max(np.abs(residual), axis=0))

    print()
    print("Add these to gravity_comp_arkit_node:")
    print(f"'r_com_x': {r_com[0]:.6f},")
    print(f"'r_com_y': {r_com[1]:.6f},")
    print(f"'r_com_z': {r_com[2]:.6f},")
    print()
    print(f"'torque_bias_x': {torque_bias[0]:.6f},")
    print(f"'torque_bias_y': {torque_bias[1]:.6f},")
    print(f"'torque_bias_z': {torque_bias[2]:.6f},")

    print()
    print("Or command-line params:")
    print(f"-p r_com_x:={r_com[0]:.6f} \\")
    print(f"-p r_com_y:={r_com[1]:.6f} \\")
    print(f"-p r_com_z:={r_com[2]:.6f} \\")
    print(f"-p torque_bias_x:={torque_bias[0]:.6f} \\")
    print(f"-p torque_bias_y:={torque_bias[1]:.6f} \\")
    print(f"-p torque_bias_z:={torque_bias[2]:.6f}")


if __name__ == "__main__":
    main()