"""Asynchronous action queue and private JAX inference worker.

The control thread owns physics; the child process owns model inference."""

import argparse
import base64
import copy
import json
import math
import os
import queue
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Request:
    generation: int
    serial: int
    observation_step: int


class AsyncJointQueue:
    horizon = 30
    replan_remaining = 24

    def __init__(self, joint_count, lower, upper, initial_target):
        if joint_count not in (29, 58):
            raise ValueError("Expected29 or58 named physical joint channels")
        self.n = joint_count
        self.lower = np.array(lower, dtype=np.float64, copy=True)
        self.upper = np.array(upper, dtype=np.float64, copy=True)
        if (
            self.lower.shape != (self.n,)
            or self.upper.shape != (self.n,)
            or not np.all(np.isfinite(self.lower))
            or not np.all(np.isfinite(self.upper))
            or np.any(self.lower > self.upper)
        ):
            raise ValueError("Invalid joint limits")
        self.generation = 0
        self.serial = 0
        self.step = 0
        self.pending = None
        self.queue = {}
        self.held = self._target(initial_target)
        self.starvation_steps = 0
        self.clipped_values = 0

    def _target(self, target):
        result = np.asarray(target, dtype=np.float64)
        if result.shape != (self.n,) or not np.isfinite(result).all():
            raise ValueError("Invalid physical target")
        return np.clip(result, self.lower, self.upper).copy()

    @property
    def remaining(self):
        # Count contiguous executable control steps, not raw message lengths.
        k = self.step
        while k in self.queue:
            k += 1
        return k - self.step

    def request(self):
        if self.pending is not None or self.remaining >= self.replan_remaining:
            return None
        self.serial += 1
        self.pending = Request(self.generation, self.serial, self.step)
        return self.pending

    def fail(self, request):
        if request == self.pending:
            self.pending = None

    def accept(self, request, physical_actions):
        if request != self.pending or request.generation != self.generation:
            return False  # stale/duplicate worker response
        actions = np.asarray(physical_actions, dtype=np.float64)
        if actions.shape != (self.horizon, self.n) or not np.isfinite(actions).all():
            self.pending = None
            raise ValueError("Expected finite30 x joint_count physical actions")
        # Timestamp0 of the prediction is the observation step, NOT arrival time.
        first = max(0, self.step - request.observation_step)
        slots = list(range(first, self.horizon))
        overlap = [i for i in slots if request.observation_step + i in self.queue]
        weights = {i: (k + 1) / (len(overlap) + 1) for k, i in enumerate(overlap)}
        for i in slots:
            target_step = request.observation_step + i
            new = actions[i]
            if i in weights:
                w = weights[i]
                new = (1 - w) * self.queue[target_step] + w * new
            self.clipped_values += int(np.count_nonzero((new < self.lower) | (new > self.upper)))
            self.queue[target_step] = self._target(new)
        self.pending = None
        return bool(slots)

    def take(self):
        if self.step in self.queue:
            self.held = self.queue.pop(self.step)
        else:
            self.starvation_steps += 1
        self.step += 1
        return self.held.copy()

    def reset(self, initial_target, *, control_step=0):
        if type(control_step) is not int or control_step < 0:
            raise ValueError("Invalid control step")
        target = self._target(initial_target)
        self.generation += 1
        self.step = control_step
        self.queue.clear()
        self.pending = None
        self.held = target


class AsyncController:
    def __init__(self, joint_count, lower, upper, initial_target, infer, *, executor=None):
        self.queue = AsyncJointQueue(joint_count, lower, upper, initial_target)
        self.infer = infer
        self.executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pi05-infer"
        )
        self.owns_executor = executor is None
        self.inflight = None

    def _receive(self):
        if self.inflight is None or not self.inflight[1].done():
            return
        request, future = self.inflight
        self.inflight = None
        if request.generation != self.queue.generation:
            # Completed prior-stage/episode inference is discarded even if error.
            return
        try:
            actions = future.result()
        except Exception:
            self.queue.fail(request)
            raise  # infrastructure failure, not policy failure or native success
        self.queue.accept(request, actions)

    def tick(self, observation):
        self._receive()
        # A non-cancellable old request may be retiring after reset. Never add
        # another concurrent model request or reuse its stale response.
        if self.inflight is None:
            request = self.queue.request()
            if request is not None:
                try:
                    future = self.executor.submit(self.infer, copy.deepcopy(observation), request)
                except Exception:
                    self.queue.fail(request)
                    raise
                self.inflight = (request, future)
                self._receive()
        return self.queue.take()

    def reset(self, initial_target, *, control_step=0, infer=None):
        self.queue.reset(initial_target, control_step=control_step)
        if infer is not None:
            self.infer = infer
        if self.inflight is not None and self.inflight[1].cancel():
            self.inflight = None

    def close(self):
        if self.inflight is not None:
            self.inflight[1].cancel()
        # Drain only after the episode has ended. Otherwise a subsequent episode
        # can overlap the same model with this controller's non-cancellable call.
        if self.owns_executor:
            self.executor.shutdown(wait=True, cancel_futures=True)


MAX_FRAME = 4 * 1024 * 1024


DTYPES = {"uint8", "float32", "float64", "int32", "int64", "uint32", "uint64", "bool"}


def encode(value):
    if isinstance(value, np.ndarray):
        if str(value.dtype) not in DTYPES:
            raise ValueError("Unsupported IPC array dtype")
        value = np.ascontiguousarray(value)
        if value.nbytes > MAX_FRAME:
            raise ValueError("IPC array exceeds limit")
        return {
            "__array__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": base64.b64encode(value.tobytes()).decode("ascii"),
        }
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode(value):
    if isinstance(value, dict):
        if "__array__" in value:
            if (
                set(value) != {"__array__", "dtype", "shape", "data"}
                or value["__array__"] is not True
            ):
                raise ValueError("Malformed array envelope")
            shape = value["shape"]
            dtype = value["dtype"]
            if (
                dtype not in DTYPES
                or not isinstance(shape, list)
                or len(shape) > 4
                or any(type(n) is not int or n < 0 for n in shape)
            ):
                raise ValueError("Invalid IPC array shape/dtype")
            size = math.prod(shape) * np.dtype(dtype).itemsize
            if size > MAX_FRAME:
                raise ValueError("Oversized IPC array")
            raw = base64.b64decode(value["data"], validate=True)
            if len(raw) != size:
                raise ValueError("IPC array byte count mismatch")
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def read_exact(stream, n):
    parts = []
    while n:
        part = stream.read(n)
        if not part:
            raise EOFError("Inference worker pipe closed")
        parts.append(part)
        n -= len(part)
    return b"".join(parts)


def read_frame(stream):
    size = struct.unpack("!I", read_exact(stream, 4))[0]
    if not 0 < size <= MAX_FRAME:
        raise ValueError("Invalid IPC frame size")

    def invalid_constant(x):
        raise ValueError("Nonfinite JSON number:" + x)

    return decode(json.loads(read_exact(stream, size), parse_constant=invalid_constant))


def write_frame(stream, value):
    raw = json.dumps(encode(value), separators=(",", ":"), allow_nan=False).encode()
    if not 0 < len(raw) <= MAX_FRAME:
        raise ValueError("Invalid IPC frame size")
    stream.write(struct.pack("!I", len(raw)) + raw)
    stream.flush()


class PipeInference:
    def __init__(
        self, command, *, stderr, environment=None, startup_timeout=600, request_timeout=180
    ):
        if (
            not isinstance(command, list)
            or not command
            or min(startup_timeout, request_timeout) <= 0
        ):
            raise ValueError("Explicit command argv and bounded timeouts required")
        self.lock = threading.Lock()
        self.responses = queue.Queue()
        self.counter = 0
        self.request_timeout = request_timeout
        self.closed = False
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=environment,
            shell=False,
        )

        def receive():
            try:
                while True:
                    self.responses.put(read_frame(self.process.stdout))
            except Exception as exc:
                self.responses.put(exc)

        self.reader = threading.Thread(target=receive, daemon=True)
        self.reader.start()
        try:
            ready = self._receive(startup_timeout)
            if ready.get("type") != "ready":
                raise RuntimeError(
                    "Worker startup failed: " + str(ready.get("error", "missing certificate"))
                )
            self.certificate = ready["certificate"]
        except Exception:
            self.close()
            raise

    def _receive(self, timeout):
        try:
            result = self.responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("Inference child response timeout") from exc
        if isinstance(result, Exception):
            raise result
        if not isinstance(result, dict):
            raise ValueError("Expected IPC message object")
        return result

    def __call__(self, observation, prompt, seed, request):
        # Called by AsyncController's background thread, not the physics thread.
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Concurrent model calls forbidden")
        try:
            if self.closed:
                raise RuntimeError("Inference child is closed")
            self.counter += 1
            identity = [request.generation, request.serial, request.observation_step]
            message = dict(
                type="infer",
                id=self.counter,
                request=identity,
                seed=seed,
                prompt=prompt,
                observation=observation,
            )
            # A bounded writer also prevents a stuck child from blocking forever
            # before response timeout starts (image frames exceed pipe capacity).
            sent = queue.Queue(maxsize=1)

            def send():
                try:
                    write_frame(self.process.stdin, message)
                    sent.put(None)
                except Exception as exc:
                    sent.put(exc)

            threading.Thread(target=send, daemon=True).start()
            try:
                send_error = sent.get(timeout=self.request_timeout)
            except queue.Empty as exc:
                raise TimeoutError("Inference child write timeout") from exc
            if send_error is not None:
                raise send_error
            reply = self._receive(self.request_timeout)
            if reply.get("id") != self.counter or reply.get("request") != identity:
                raise ValueError("Inference response identity mismatch")
            if reply.get("type") == "error":
                raise RuntimeError(reply.get("error", "Worker error"))
            if reply.get("type") != "actions":
                raise ValueError("Expected action response")
            return reply["actions"]
        except Exception:
            self.close()
            raise
        finally:
            self.lock.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Stop ONLY the Popen-owned inference child; never process groups/Slurm jobs.
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self.reader.join(timeout=1)


def worker_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    wire = sys.stdout.buffer
    sys.stdout = sys.stderr
    try:
        from utils.pi05_utils.checkpoint import read_run
        from utils.pi05_utils.data import camera_mapping

        identity, prepared, _ = read_run(args.run)
        import jax

        from algo.pi05 import NativeJointInference, load_final_model
        from utils.pi05_utils.data import evaluation_prompt

        devices = jax.devices("gpu")
        if len(devices) != 1:
            raise ValueError("Exactly one allocated visible model GPU is required")
        model, certificate = load_final_model(args.run, identity, prepared, device=devices[0])
        infer = NativeJointInference(
            model,
            prepared,
            live_state_names=prepared["state_names"],
            camera_keys=camera_mapping(prepared),
        )
        write_frame(wire, dict(type="ready", certificate=certificate))
        while True:
            try:
                message = read_frame(sys.stdin.buffer)
            except EOFError:
                return
            if message.get("type") != "infer":
                raise ValueError("Unknown inference command")
            request = message["request"]
            if len(request) != 3 or any(type(x) is not int or x < 0 for x in request):
                raise ValueError("Invalid request clock")
            if type(message["seed"]) is not int or message["seed"] not in range(10001, 10051):
                raise ValueError("Official Eval50 seed required")
            if message["prompt"] != evaluation_prompt(prepared["env_id"], message["seed"]):
                raise ValueError("Inference prompt differs from the shared episode assignment")
            try:
                actions = infer(
                    message["observation"], message["prompt"], message["seed"], Request(*request)
                )
                write_frame(
                    wire, dict(type="actions", id=message["id"], request=request, actions=actions)
                )
            except Exception as exc:
                write_frame(
                    wire,
                    dict(
                        type="error",
                        id=message["id"],
                        request=request,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
                return
    except Exception as exc:
        write_frame(wire, dict(type="startup_error", error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    worker_main()
