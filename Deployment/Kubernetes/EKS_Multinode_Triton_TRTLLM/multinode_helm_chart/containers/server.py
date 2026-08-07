# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time

ERROR_EXIT_DELAY = 15
ERROR_CODE_FATAL = 255
ERROR_CODE_USAGE = 253
EXIT_SUCCESS = 0
DELAY_BETWEEN_QUERIES = 2


def die(exit_code: int):
    # 打印提示后延迟退出，给管理员留出抓取日志的时间窗口。
    if exit_code is None:
        exit_code = ERROR_CODE_FATAL

    write_error(f"       Waiting {ERROR_EXIT_DELAY} second before exiting.")
    # 延迟进程终止，让管理员在进程退出并重启前有时间抓取日志。
    # Delay the process' termination to provide a small window for administrators to capture the logs before it exits and restarts.
    time.sleep(ERROR_EXIT_DELAY)

    exit(exit_code)


def parse_arguments():
    # 解析命令行参数：模式（leader/worker）以及 TP/PP、模型仓库目录、GPU 数等选项。
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["leader", "worker"])
    parser.add_argument(
        "--triton_model_repo_dir",
        type=str,
        default=None,
        required=True,
        help="Directory that contains Triton Model Repo to be served",
    )
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism.")
    parser.add_argument("--iso8601", action="count", default=0)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Namespace of the Kubernetes deployment.",
    )
    parser.add_argument(
        "--gpu_per_node",
        type=int,
        help="How many gpus are in each pod/node (We launch one pod per node). Only required in leader mode.",
    )
    parser.add_argument(
        "--stateful_set_group_key",
        type=str,
        default=None,
        help="Value of leaderworkerset.sigs.k8s.io/group-key, Leader uses this to gang schedule and its only needed in leader mode",
    )
    parser.add_argument(
        "--enable_nsys", action="store_true", help="Enable Triton server profiling"
    )

    return parser.parse_args()


def run_command(cmd_args: [str], omit_args: [int] = None):
    # 执行子进程命令并打印命令行，omit_args 指定的参数用 ***** 打码。
    command = ""

    for i, arg in enumerate(cmd_args):
        command += " "
        if omit_args is not None and i in omit_args:
            command += "*****"
        else:
            command += arg

    write_output(f">{command}")
    write_output(" ")

    return subprocess.call(cmd_args, stderr=sys.stderr, stdout=sys.stdout)


def signal_handler(sig, frame):
    # 收到 SIGINT/SIGTERM 时打印信号并正常退出。
    write_output(f"Signal {sig} detected, quitting.")
    exit(EXIT_SUCCESS)


def wait_for_workers(num_total_pod: int, args):
    # 组调度核心：按 LeaderWorkerSet 的 group-key 标签轮询，等待全部 Pod 进入 Running 状态。
    if num_total_pod is None or num_total_pod <= 0:
        raise RuntimeError("Argument `world_size` must be greater than zero.")

    write_output("Begin waiting for worker pods.")

    # 用 kubectl 按 group-key 标签筛选 Running 状态的 Pod。
    cmd_args = [
        "kubectl",
        "get",
        "pods",
        "-n",
        f"{args.namespace}",
        "-l",
        f"leaderworkerset.sigs.k8s.io/group-key={args.stateful_set_group_key}",
        "--field-selector",
        "status.phase=Running",
        "-o",
        "jsonpath='{.items[*].metadata.name}'",
    ]
    command = " ".join(cmd_args)

    workers = []

    # 循环查询，直到 Running 的 Pod 数量达到期望总数。
    while len(workers) < num_total_pod:
        time.sleep(DELAY_BETWEEN_QUERIES)

        if args.verbose:
            write_output(f"> {command}")

        output = subprocess.check_output(cmd_args).decode("utf-8")

        if args.verbose:
            write_output(output)

        output = output.strip("'")

        workers = output.split(" ")

        if len(workers) < num_total_pod:
            write_output(
                f"Waiting for worker pods, {len(workers)} of {num_total_pod} ready."
            )
        else:
            write_output(f"{len(workers)} of {num_total_pod} workers ready.")

    write_output(" ")

    # 对 Pod 名排序，保证各节点上 mpirun 的 host 顺序一致。
    if workers is not None and len(workers) > 1:
        workers.sort()

    return workers


def write_output(message: str):
    # 向标准输出打印信息并立即刷新。
    print(message, file=sys.stdout, flush=True)


def write_error(message: str):
    # 向标准错误输出打印信息并立即刷新。
    print(message, file=sys.stderr, flush=True)


def do_leader(args):
    # leader 入口：等待全部 Pod 就绪后，用 AWS OpenMPI 通过 kubessh 跨节点拉起多 rank 的 tritonserver。
    write_output(
        f"Server is assuming each node has {args.gpu_per_node} GPUs. To change this, use --gpu_per_node"
    )

    world_size = args.tp * args.pp

    if world_size <= 0:
        raise Exception(
            "usage: Options --tp and --pp must both be equal to or greater than 1."
        )

    write_output(f"Executing Leader (world size: {world_size})")

    # 每个节点一个 Pod，因此需要的 Pod 数 = 总 GPU 数 / 每节点 GPU 数。
    workers = wait_for_workers(world_size / args.gpu_per_node, args)

    if len(workers) != (world_size / args.gpu_per_node):
        write_error(
            f"fatal: {len(workers)} found, expected {world_size / args.gpu_per_node}."
        )
        die(ERROR_EXIT_DELAY)

    # 为每个 worker 指定 MPI slot 数（即每节点 GPU 数）。
    workers_with_mpi_slots = [worker + f":{args.gpu_per_node}" for worker in workers]

    # 启用 nsys 时用 nsys profile 包裹 mpirun，输出性能剖析报告。
    if args.enable_nsys:
        cmd_args = [
            "/var/run/models/nsight-systems-cli-DVS/bin/nsys",
            "profile",
            "--force-overwrite",
            "true",
            "-t",
            "cuda,nvtx",
            "--enable",
            "efa_metrics",
            "-o",
            "/var/run/models/nsys_report",
            "/opt/amazon/openmpi/bin/mpirun",
            "--allow-run-as-root",
        ]
    else:
        cmd_args = [
            "/opt/amazon/openmpi/bin/mpirun",
            "--allow-run-as-root",
        ]

    if args.verbose:
        cmd_args += ["--debug-devel"]

    # 通过 kubessh 作为远程 shell 代理，在多个 Pod 上拉起 MPI 进程。
    cmd_args += [
        "--report-bindings",
        "-mca",
        "plm_rsh_agent",
        "kubessh",
        "-np",
        f"{world_size}",
        "--host",
        ",".join(workers_with_mpi_slots),
    ]

    # 为每个节点添加以 ':' 分隔的进程命令行。
    # Add per node command lines separated by ':'.
    for i in range(world_size):
        if i != 0:
            cmd_args += [":"]

        # 每个 rank 独占一个 tritonserver 进程。
        cmd_args += [
            "-n",
            "1",
            "tritonserver",
            "--allow-cpu-metrics=false",
            "--allow-gpu-metrics=false",
            "--disable-auto-complete-config",
            f"--id=rank{i}",
            "--model-load-thread-count=2",
            f"--model-repository={args.triton_model_repo_dir}",
        ]

        # rank 0 需要支持指标采集和网络服务。
        # Rank0 node needs to support metrics collection and web services.
        if i == 0:
            cmd_args += [
                "--allow-metrics=true",
                "--metrics-interval-ms=1000",
            ]

            if args.verbose:
                cmd_args += ["--log-verbose=2"]

            if args.iso8601 > 0:
                cmd_args += ["--log-format=ISO8601"]

        # 其余 rank 可以关闭指标、网络服务和日志。
        # Rank(N) nodes can disable metrics, web services, and logging.
        else:
            cmd_args += [
                "--allow-http=false",
                "--allow-grpc=false",
                "--allow-metrics=false",
                "--model-control-mode=explicit",
                "--load-model=tensorrt_llm",
                "--log-info=false",
                "--log-warning=false",
            ]

    result = run_command(cmd_args)

    if result != 0:
        die(result)

    exit(result)


def do_worker(args):
    # worker 入口：注册信号处理器后挂起等待，实际计算由 leader 通过 mpirun 远程拉起。
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    write_output("Worker paused awaiting SIGINT or SIGTERM.")
    signal.pause()


def main():
    # 启动时先打印系统信息（用户、内存限制、GPU 状态），便于排查环境问题。
    write_output("Reporting system information.")
    run_command(["whoami"])
    run_command(
        ["cgget", "-n", "--values-only", "--variable memory.limit_in_bytes", "/"]
    )
    run_command(["nvidia-smi"])

    args = parse_arguments()
    if args.triton_model_repo_dir is None:
        raise Exception(f"--triton_model_repo_dir is required")

    if args.verbose:
        write_output(f"Triton model repository is at:'{args.triton_model_repo_dir}'")

    # 按模式分发执行；leader 模式要求提供每节点 GPU 数和 group-key。
    if args.mode == "leader":
        if args.gpu_per_node is None:
            raise Exception("--gpu_per_node is required for leader mode")
        if args.stateful_set_group_key is None:
            raise Exception("--stateful_set_group_key is required for leader mode")
        do_leader(args)
    elif args.mode == "worker":
        do_worker(args)
    else:
        write_error(f"usage: server.py <mode> [<options>].")
        write_error(f'       Invalid mode ("{args.mode}") provided.')
        write_error(f'       Supported values are "init" or "exec".')
        die(ERROR_CODE_USAGE)


if __name__ == "__main__":
    main()
