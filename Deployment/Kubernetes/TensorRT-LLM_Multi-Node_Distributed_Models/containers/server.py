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

# 以下常量与 Helm Chart 中的挂载点保持一致，两处修改必须同步。
# These values are expected to match the mount points in the Helm Chart.
# Any changes here must also be made there, and vice versa.
HUGGING_FACE_TOKEN_PATH = "/var/run/secrets/hugging_face/password"

ERROR_EXIT_DELAY = 15
ERROR_CODE_FATAL = 255
ERROR_CODE_USAGE = 253
EXIT_SUCCESS = 0

# 环境变量键名。
# Environment variable keys.
CLI_VERBOSE_KEY = "TRITON_CLI_VERBOSE"
ENGINE_PATH_KEY = "ENGINE_DEST_PATH"
HUGGING_FACE_KEY = "HF_HOME"
MODEL_PATH_KEY = "MODEL_DEST_PATH"

HUGGING_FACE_CLI = "huggingface-cli"
DELAY_BETWEEN_QUERIES = 2


# ---


def create_directory(directory_path: str):
    # 逐级创建目录（类似 mkdir -p），跳过空路径段。
    if directory_path is None or len(directory_path) == 0:
        return

    segments = directory_path.split("/")
    path = ""

    for segment in segments:
        if segment is None or len(segment) == 0:
            continue

        path = f"{path}/{segment}"

        if is_verbose:
            write_output(f"> mkdir {path}")

        if not os.path.exists(path):
            os.mkdir(path)


# ---


def die(exit_code: int):
    # 打印提示后延迟退出，给管理员留出抓取日志的时间窗口。
    if exit_code is None:
        exit_code = ERROR_CODE_FATAL

    write_error(f"       Waiting {ERROR_EXIT_DELAY} second before exiting.")
    # 延迟进程终止，让管理员在进程退出并重启前有时间抓取日志。
    # Delay the process' termination to provide a small window for administrators to capture the logs before it exits and restarts.
    time.sleep(ERROR_EXIT_DELAY)

    exit(exit_code)


# ---


def hugging_face_authenticate(args):
    # 检测并读取挂载的 Hugging Face token 文件，用 CLI 完成登录认证。
    # Validate that `HF_HOME` environment variable was set correctly.
    # 校验 `HF_HOME` 环境变量已正确设置。
    if HUGGING_FACE_HOME is None or len(HUGGING_FACE_HOME) == 0:
        raise Exception(f"Required environment variable '{HUGGING_FACE_KEY}' not set.")

    # 如果挂载了 Hugging Face secret，就用它向 Hugging Face 认证。
    # When a Hugging Face secret has been mounted, we'll use that to authenticate with Hugging Face.
    if os.path.exists(HUGGING_FACE_TOKEN_PATH):
        with open(HUGGING_FACE_TOKEN_PATH) as token_file:
            write_output(
                f"Hugging Face token file '{HUGGING_FACE_TOKEN_PATH}' detected, attempting to authenticate w/ Hugging Face."
            )
            write_output(" ")

            hugging_face_token = token_file.read()

            # 调用 Hugging Face 的 CLI 完成认证。
            # Use Hugging Face's CLI to complete the authentication.
            result = run_command(
                [HUGGING_FACE_CLI, "login", "--token", hugging_face_token], [3]
            )

            if result != 0:
                raise Exception(f"Hugging Face authentication failed. ({result})")

            write_output("Hugging Face authentication successful.")
            write_output(" ")


# ---


def parse_arguments():
    # 解析命令行参数：模式（convert/leader/worker）以及模型、并行度、部署名等选项。
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["convert", "leader", "worker"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--dt",
        type=str,
        default="float16",
        choices=["bfloat16", "float16", "float32"],
        help="Tensor type.",
    )
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism.")
    parser.add_argument("--iso8601", action="count", default=0)
    parser.add_argument("--verbose", action="count", default=0)
    parser.add_argument(
        "--deployment", type=str, help="Name of the Kubernetes deployment."
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Namespace of the Kubernetes deployment.",
    )
    parser.add_argument("--multinode", action="count", default=0)
    parser.add_argument(
        "--noconvert",
        action="count",
        default=0,
        help="Prevents leader waiting for model conversion before inference serving begins.",
    )

    return parser.parse_args()


# ---


def remove_path(path: str):
    # 递归删除指定目录或文件，用于清理失败的转换残留。
    if os.path.exists(path):
        if os.path.isfile(path):
            if is_verbose:
                write_output(f"> rm {path}")

            os.remove(path)
        else:
            if is_verbose:
                write_output(f"> rm -rf {path}")

            shutil.rmtree(path)


# ---


def run_command(cmd_args: [str], omit_args: [int] = None):
    # 执行子进程命令并打印命令行，omit_args 指定的参数用 ***** 打码（如 token）。
    command = ""

    for i, arg in enumerate(cmd_args):
        command += " "
        if omit_args is not None and i in omit_args:
            command += "*****"
        else:
            command += arg

    write_output(f">{command}")
    write_output(" ")

    # 运行 triton_cli 构建 TRT-LLM 引擎和 plan。
    # Run triton_cli to build the TRT-LLM engine + plan.
    return subprocess.call(cmd_args, stderr=sys.stderr, stdout=sys.stdout)


# ---


def signal_handler(sig, frame):
    # 收到 SIGINT/SIGTERM 时打印信号并正常退出。
    write_output(f"Signal {sig} detected, quitting.")
    exit(EXIT_SUCCESS)


# ---


def wait_for_convert(args):
    # 轮询查询模型转换 Job 的状态，直到其成功（或失败）为止。
    if args.noconvert != 0:
        write_output("Leader skip waiting for model-conversion job.")
        return

    write_output("Begin waiting for model-conversion job.")

    # 用 kubectl 查询转换 Job 的 active/failed/succeeded 状态。
    cmd_args = [
        "kubectl",
        "get",
        f"job/{args.deployment}",
        "-n",
        f"{args.namespace}",
        "-o",
        'jsonpath={.status.active}{"|"}{.status.failed}{"|"}{.status.succeeded}',
    ]
    command = " ".join(cmd_args)

    active = 1
    failed = 0
    succeeded = 0

    # 只要 Job 还在运行或尚未成功就持续轮询。
    while active > 0 and succeeded == 0:
        time.sleep(DELAY_BETWEEN_QUERIES)

        if is_verbose:
            write_output(f"> {command}")

        output = subprocess.check_output(cmd_args).decode("utf-8")
        if output is None or len(output) == 0:
            continue

        if is_verbose:
            write_output(output)

        output = output.strip(" ")
        # 解析 "active|failed|succeeded" 三段状态值。
        if len(output) > 0:
            parts = output.split("|")

            if len(parts) > 2 and len(parts[2]) > 0:
                succeeded = int(parts[2])
            else:
                succeeded = 0

            if len(parts) > 1 and len(parts[1]) > 0:
                failed = int(parts[1])
            else:
                failed = 0

            if len(parts) > 0 and len(parts[0]) > 0:
                active = int(parts[0])
            else:
                active = 0

        # 根据状态打印等待进度；Job 失败则直接抛出异常。
        if active > 0:
            write_output("Waiting for model-conversion job.")
        elif succeeded > 0:
            write_output("Model-conversion job succeeded.")
        elif failed > 0:
            write_error("Model-conversion job failed.")
            raise RuntimeError("Model-conversion job failed.")

    write_output(" ")


# ---


def wait_for_workers(world_size: int):
    # 组调度核心：轮询等待全部 worker Pod 就绪，凑齐 world_size 个 Pod 才返回。
    if world_size is None or world_size <= 0:
        raise RuntimeError("Argument `world_size` must be greater than zero.")

    write_output("Begin waiting for worker pods.")

    # 用 kubectl 按 app 标签查询所有相关 Pod 的名称。
    cmd_args = [
        "kubectl",
        "get",
        "pods",
        "-n",
        f"{args.namespace}",
        "-l",
        f"app={args.deployment}",
        "-o",
        "jsonpath='{.items[*].metadata.name}'",
    ]
    command = " ".join(cmd_args)

    workers = []

    # 循环查询，直到检测到的 Pod 数量达到 world_size。
    while len(workers) < world_size:
        time.sleep(DELAY_BETWEEN_QUERIES)

        if is_verbose:
            write_output(f"> {command}")

        output = subprocess.check_output(cmd_args).decode("utf-8")

        if is_verbose:
            write_output(output)

        output = output.strip("'")

        workers = output.split(" ")

        if len(workers) < world_size:
            write_output(
                f"Waiting for worker pods, {len(workers)} of {world_size} ready."
            )
        else:
            write_output(f"{len(workers)} of {world_size} workers ready.")

    write_output(" ")

    # 对 Pod 名排序，保证各节点上 mpirun 的 host 顺序一致。
    if workers is not None and len(workers) > 1:
        workers.sort()

    return workers


# ---


def write_output(message: str):
    # 向标准输出打印信息并立即刷新。
    print(message, file=sys.stdout, flush=True)


# ---


def write_error(message: str):
    # 向标准错误输出打印信息并立即刷新。
    print(message, file=sys.stderr, flush=True)


# ---
# 以下为主要的入口函数。
# Below this line are the primary functions.
# ---


def do_convert(args):
    # 模型转换入口：从 Hugging Face 下载模型并用 triton CLI 生成 TRT-LLM 引擎和 plan，存入共享存储。
    write_output("Initializing Model")

    if args.model is None or len(args.model) == 0:
        write_error("fatal: Model name must be provided.")
        die(ERROR_CODE_FATAL)

    create_directory(ENGINE_DIRECTORY)
    create_directory(MODEL_DIRECTORY)

    hugging_face_authenticate(args)

    # 用 lock/ready 标记文件记录转换状态，供多 Pod 并发检测是否已完成。
    engine_path = ENGINE_DIRECTORY
    engine_lock_file = os.path.join(engine_path, "lock")
    engine_ready_file = os.path.join(engine_path, "ready")
    model_path = MODEL_DIRECTORY
    model_lock_file = os.path.join(model_path, "lock")
    model_ready_file = os.path.join(model_path, "ready")

    # 如果引擎和模型都已就绪且无残留锁文件，直接退出，避免重复转换。
    # When the model and plan already exist, we can exit early, happily.
    if os.path.exists(engine_ready_file) and os.path.exists(model_ready_file):
        everything_exists = True

        # 存在锁文件说明上次转换被中断，需要清理重来。
        if os.path.exists(engine_lock_file):
            write_output("Incomplete engine directory detected, removing.")
            everything_exists = False
            remove_path(engine_path)

        if os.path.exists(model_lock_file):
            write_output("Incomplete model directory detected, removing.")
            everything_exists = False
            remove_path(engine_path)

        if everything_exists:
            write_output(
                f"TensorRT engine and plan detected for {args.model}. No work to do, exiting."
            )
            exit(EXIT_SUCCESS)

    write_output(f"Begin generation of TensorRT engine and plan for {args.model}.")
    write_output(" ")

    create_directory(engine_path)

    # 创建引擎目录的锁文件，标记转换进行中。
    # Create a lock file for the engine directory.
    if is_verbose:
        write_output(f"> echo '{args.model}' > {engine_lock_file}")

    with open(engine_lock_file, "w") as f:
        f.write(args.model)

    create_directory(model_path)

    # 创建模型目录的锁文件，标记转换进行中。
    # Create a lock file for the engine model.
    if is_verbose:
        write_output(f"> echo '{args.model}' > {model_lock_file}")

    with open(model_lock_file, "w") as f:
        f.write(args.model)

    try:
        # 组装 triton CLI 的导入转换命令。
        # Build up a set of args for the subprocess call.
        cmd_args = [
            "triton",
            "import",
            "--model",
            args.model,
            "--model-repository",
            MODEL_DIRECTORY,
        ]

        cmd_args += ["--backend", "tensorrtllm"]

        if args.dt is not None and args.dt in ["bfloat", "float16", "float32"]:
            cmd_args += ["--data-type", args.dt]

        if args.pp > 1:
            cmd_args += ["--pipeline-parallelism", f"{args.pp}"]

        if args.tp > 1:
            cmd_args += ["--tensor-parallelism", f"{args.tp}"]

        # 跨节点多机部署时禁用 custom all-reduce（无 NVLink，需走 NCCL）。
        if args.tp * args.pp > 1 and args.multinode > 0:
            cmd_args += ["--disable-custom-all-reduce"]

        # 开启 verbose 时插入 --verbose 标志。
        # 注意该标志必须紧跟在 `triton` 之后，不能放在其他位置。
        # 这个限制在未来的 triton_cli 版本中可能会移除。
        # When verbose, insert the verbose flag.
        # It is important to note that the flag must immediately follow `triton` and cannot be in another ordering position.
        # This limitation will likely be removed a future release of triton_cli.
        if is_verbose:
            cmd_args.insert(1, "--verbose")

        result = run_command(cmd_args)

        if result == 0:
            # 转换成功：写入 ready 标记文件，并删除锁文件。
            # Create the ready file.
            if is_verbose:
                write_output(f"> echo '{args.model}' > {engine_ready_file}")

            with open(engine_ready_file, "w") as f:
                f.write(args.model)

            # Create the ready file.
            if is_verbose:
                write_output(f"> echo '{args.model}' > {model_ready_file}")

            with open(model_ready_file, "w") as f:
                f.write(args.model)

            # Remove the lock files.
            # 删除锁文件。
            if is_verbose:
                write_output(f"> rm {engine_lock_file}")

            os.remove(engine_lock_file)

            if is_verbose:
                write_output(f"> rm {model_lock_file}")

            os.remove(model_lock_file)
        else:
            # 转换失败：清空模型和引擎目录，下次重新转换。
            # Clean the model and engine directories when the command fails.
            remove_path(engine_path)
            remove_path(model_path)

        exit(result)

    except Exception as exception:
        # 异常时同样清理目录并向上抛出。
        remove_path(engine_path)
        remove_path(model_path)
        raise exception


# ---


def do_leader(args):
    # leader 入口：等待转换 Job 和全部 worker Pod 就绪后，用 mpirun 跨节点拉起多 rank 的 tritonserver。
    world_size = args.tp * args.pp

    if world_size <= 0:
        raise Exception(
            "usage: Options --pp and --pp must both be equal to or greater than 1."
        )

    write_output(f"Executing Leader (world size: {world_size})")

    wait_for_convert(args)

    # 组调度：确保所有 worker Pod 就绪后才启动 MPI。
    workers = wait_for_workers(world_size)

    if len(workers) != world_size:
        write_error(f"fatal: {len(workers)} found, expected {world_size}.")
        die(ERROR_EXIT_DELAY)

    # 组装 mpirun 命令行，通过 kubessh 作为远程 shell 代理在多个 Pod 上拉起进程。
    cmd_args = [
        "mpirun",
        "--allow-run-as-root",
    ]

    if is_verbose > 0:
        cmd_args += ["--debug-devel"]

    cmd_args += [
        "--report-bindings",
        "-mca",
        "plm_rsh_agent",
        "kubessh",
        "-np",
        f"{world_size}",
        "--host",
        ",".join(workers),
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
            f"--model-repository={MODEL_DIRECTORY}",
        ]

        # rank 0 需要支持指标采集和网络服务。
        # Rank0 node needs to support metrics collection and web services.
        if i == 0:
            cmd_args += [
                "--allow-metrics=true",
                "--metrics-interval-ms=1000",
            ]

            if is_verbose > 0:
                cmd_args += ["--log-verbose=1"]

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


# ---


def do_worker(args):
    # worker 入口：注册信号处理器后挂起等待，实际计算由 leader 通过 mpirun 远程拉起。
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    write_output("Worker paused awaiting SIGINT or SIGTERM.")
    signal.pause()


# ---


# 启动时先打印系统信息（用户、内存限制、GPU 状态），便于排查环境问题。
write_output("Reporting system information.")
run_command(["whoami"])
run_command(["cgget", "-n", "--values-only", "--variable memory.limit_in_bytes", "/"])
run_command(["nvidia-smi"])

# 读取环境变量并做必填校验。
ENGINE_DIRECTORY = os.getenv(ENGINE_PATH_KEY)
HUGGING_FACE_HOME = os.getenv(HUGGING_FACE_KEY)
MODEL_DIRECTORY = os.getenv(MODEL_PATH_KEY)

is_verbose = os.getenv(CLI_VERBOSE_KEY) is not None

# Validate that `ENGINE_PATH_KEY` isn't empty.
# 校验 `ENGINE_PATH_KEY` 非空。
if ENGINE_DIRECTORY is None or len(ENGINE_DIRECTORY) == 0:
    raise Exception(f"Required environment variable '{ENGINE_PATH_KEY}' not set.")

# Validate that `MODEL_PATH_KEY` isn't empty.
# 校验 `MODEL_PATH_KEY` 非空。
if MODEL_DIRECTORY is None or len(MODEL_DIRECTORY) == 0:
    raise Exception(f"Required environment variable '{MODEL_PATH_KEY}' not set.")

# 解析传入的选项。
# Parse options provided.
args = parse_arguments()

# 用命令行参数更新 is_verbose 标志。
# Update the is_verbose flag with values passed in by options.
is_verbose = is_verbose or args.verbose > 0

if is_verbose:
    write_output(f"{ENGINE_PATH_KEY}='{ENGINE_DIRECTORY}'")
    write_output(f"{HUGGING_FACE_KEY}='{HUGGING_FACE_HOME}'")
    write_output(f"{MODEL_PATH_KEY}='{MODEL_DIRECTORY}'")

# 按模式分发执行。
if args.mode == "convert":
    do_convert(args)

elif args.mode == "leader":
    do_leader(args)

elif args.mode == "worker":
    do_worker(args)

else:
    write_error(f"usage: server.py <mode> [<options>].")
    write_error(f'       Invalid mode ("{args.mode}") provided.')
    write_error(f'       Supported values are "init" or "exec".')
    die(ERROR_CODE_USAGE)
