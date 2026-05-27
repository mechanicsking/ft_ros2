#!/usr/bin/env python3
import argparse
import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path")
    parser.add_argument("--topic", default="/aidin_ft/wrench_compensated")
    parser.add_argument("--t_start", type=float, default=None,
                        help="bag 시작 기준 사용할 시작 시간 [s]")
    parser.add_argument("--t_end", type=float, default=None,
                        help="bag 시작 기준 사용할 끝 시간 [s]")
    args = parser.parse_args()

    bag_path = Path(args.bag_path)

    forces = []
    torques = []
    times = []

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

            if args.t_start is not None and t_rel < args.t_start:
                continue
            if args.t_end is not None and t_rel > args.t_end:
                continue

            f = msg.wrench.force
            tau = msg.wrench.torque

            forces.append([f.x, f.y, f.z])
            torques.append([tau.x, tau.y, tau.z])
            times.append(t_rel)

    forces = np.array(forces)
    torques = np.array(torques)

    if len(forces) == 0:
        print("No samples in selected range.")
        return

    f_mean = forces.mean(axis=0)
    f_std = forces.std(axis=0)

    tau_mean = torques.mean(axis=0)
    tau_std = torques.std(axis=0)

    print("Samples:", len(forces))
    print("Time range used:", times[0], "~", times[-1], "s")
    print()
    print("force_bias candidate [N]")
    print("x:", f_mean[0])
    print("y:", f_mean[1])
    print("z:", f_mean[2])
    print("std:", f_std)
    print()
    print("torque_bias candidate [Nm]")
    print("x:", tau_mean[0])
    print("y:", tau_mean[1])
    print("z:", tau_mean[2])
    print("std:", tau_std)


if __name__ == "__main__":
    main()