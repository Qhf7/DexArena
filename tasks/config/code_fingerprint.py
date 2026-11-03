"""Per-environment code fingerprint: which build recorded THIS environment.

WHY NOT THE REPO SHA
--------------------
The export gate has to answer "were these episodes recorded by the same code?".
It first answered it with the repository commit sha, which is wrong in a way
that bites during a multi-week collection run: editing a task nobody has
collected yet changes the sha, so every episode recorded afterwards -- of every
OTHER environment, whose code did not move -- gets a new version and the gate
refuses to add it to its own dataset. One family's tweak forks forty datasets.

So the version is computed per environment instead: hash the source of the
files that actually determine this environment's recorded behaviour.

HOW THE FILE SET IS DERIVED
---------------------------
Not from `sys.modules`. `tasks/__init__.py` imports every task module so that
registration happens automatically, so the runtime import graph claims every
environment depends on every other one. That is a side effect, not a dependency.

The closure is read out of the SOURCE instead (AST), starting from the modules
of the classes in the environment's MRO, following only repo-local imports, and
treating the `task` package itself as a leaf.

Two prunes on top, both measured:

  * `tasks.robots.*` is dropped from the task-module closure and replaced by the
    closure of the agent the environment actually builds (`robot_uids`). Every
    family registers its Sharpa, Wuji and XHand variants in one module, so the
    raw closure says a Sharpa env depends on the XHand agent. It does not.
  * shared plant -- `sim_profiles`, `panda_base`, `hand_controller_gains`,
    `angled_mount` -- stays in, and is therefore in all 40 closures. That is the
    correct answer, not over-inclusion: changing the sim profile or the hand
    gains changes what every environment records. It also makes the gate say out
    loud which files must not move during a collection run.

Over-inclusion is safe (the gate is merely stricter than needed); under-
inclusion is not (two behaviours land in one column). When in doubt this errs
toward including.

RETROACTIVE
-----------
`fingerprint_at_sha` reads the same file set out of git rather than the
worktree, so episodes recorded before this existed -- which carry only a repo
sha -- can still be compared against episodes recorded after it.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REPO_PREFIXES = ("tasks", "data_gen", "utils")


def _module_path(name: str) -> pathlib.Path | None:
    direct = REPO_ROOT / (name.replace(".", "/") + ".py")
    if direct.exists():
        return direct
    pkg = REPO_ROOT / name.replace(".", "/") / "__init__.py"
    return pkg if pkg.exists() else None


def _walk(modname: str, seen: set[str]) -> None:
    if modname in seen:
        return
    seen.add(modname)
    path = _module_path(modname)
    if path is None:
        return
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            # `from tasks.juice_preparation.juice_preparation_long import X` may name either a module or a symbol;
            # try the dotted form first and fall back to the module.
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for name in names:
            if name == "tasks":         # the auto-import aggregator: a leaf
                continue
            if not name.startswith(_REPO_PREFIXES):
                continue
            if _module_path(name) is None:
                name = name.rsplit(".", 1)[0]
                if name == "tasks" or _module_path(name) is None:
                    continue
            _walk(name, seen)


def _agent_module(robot_uids) -> str | None:
    """The module defining the agent this environment builds, if it is ours."""
    from mani_skill.agents.registration import REGISTERED_AGENTS

    uid = robot_uids
    while isinstance(uid, (tuple, list)) and uid:
        uid = uid[0]
    entry = REGISTERED_AGENTS.get(uid)
    if entry is None:
        return None
    module = getattr(entry.agent_cls, "__module__", "")
    return module if module.startswith(_REPO_PREFIXES) else None


@functools.lru_cache(maxsize=None)
def env_modules(env_id: str) -> tuple[str, ...]:
    """Repo modules whose source determines what `env_id` records."""
    from mani_skill.utils.registration import REGISTERED_ENVS

    spec = REGISTERED_ENVS.get(env_id)
    if spec is None:
        raise KeyError(f"{env_id} is not registered")

    seen: set[str] = set()
    for klass in type.mro(spec.cls):
        module = getattr(klass, "__module__", "")
        if module.startswith(_REPO_PREFIXES):
            _walk(module, seen)

    # Drop every agent reached only because a family registers all three hands
    # in one module, then add back the one this environment actually builds.
    seen = {m for m in seen if not m.startswith("tasks.robots.panda_")}
    uids = (spec.default_kwargs or {}).get("robot_uids")
    agent_module = _agent_module(uids) if uids else None
    if agent_module:
        _walk(agent_module, seen)

    return tuple(sorted(m for m in seen if _module_path(m) is not None))


_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _normalized(blob: bytes) -> bytes:
    """Source reduced to what can change behaviour.

    Hashing raw bytes makes a version bump out of a typo fix in a comment. That
    is not merely noisy: the shared-plant files sit in all 40 closures, so
    adding a warning banner to one would fork every dataset being collected --
    the exact accident the freeze list exists to prevent.

    So the hash is taken over the parsed tree with comments (never in the AST)
    and docstrings removed. Any real edit still moves it; prose no longer does.
    Falls back to the raw bytes if the file does not parse.
    """
    try:
        tree = ast.parse(blob)
    except SyntaxError:
        return blob
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree).encode()


def _hash(pairs) -> str:
    digest = hashlib.sha256()
    for name, blob in pairs:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(_normalized(blob)).digest())
    return digest.hexdigest()


def fingerprint_worktree(env_id: str) -> str:
    """Fingerprint from the files on disk -- what a recording run is running."""
    return _hash((m, _module_path(m).read_bytes()) for m in env_modules(env_id))


def fingerprint_at_sha(env_id: str, sha: str) -> str | None:
    """The same fingerprint as of a past commit. None if git cannot answer.

    Lets episodes recorded before fingerprints existed be compared with ones
    recorded after, instead of forcing a new dataset directory for every
    unrelated edit made during a collection run.
    """
    pairs = []
    for module in env_modules(env_id):
        rel = _module_path(module).relative_to(REPO_ROOT)
        try:
            blob = subprocess.run(
                ["git", "show", f"{sha}:{rel.as_posix()}"],
                cwd=REPO_ROOT, capture_output=True, check=True,
            ).stdout
        except (subprocess.CalledProcessError, OSError):
            return None
        pairs.append((module, blob))
    return _hash(pairs)
