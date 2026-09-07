"""
Thread-safety of :func:`krrood.utils.memoize` and :func:`krrood.utils.copy_memoize`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from krrood.utils import copy_memoize, memoize

# %% mimics for the hazard: a value whose copy is unsafe to run concurrently


@dataclass
class _ReentrancyDetectingValue:
    """
    A value whose copy detects if another thread is copying the same value at the same
    time, mimicking a CasADi-backed object whose ``deepcopy`` is unsafe against
    concurrent execution.
    """

    payload: int
    """
    Arbitrary data carried by this value, used only to check that copies stay
    value-equal to the original.
    """

    _copy_in_progress: list[bool] = field(default_factory=lambda: [False])
    """
    One-element flag shared by every copy made directly from this instance, marking
    whether some thread is currently inside :meth:`__deepcopy__` on it.
    """

    def __deepcopy__(self, memo: dict) -> _ReentrancyDetectingValue:
        if self._copy_in_progress[0]:
            raise RuntimeError("concurrent deepcopy detected on the same value")
        self._copy_in_progress[0] = True
        try:
            time.sleep(0.01)
            return _ReentrancyDetectingValue(payload=self.payload)
        finally:
            self._copy_in_progress[0] = False


@dataclass(eq=False)
class _CopyMemoizingOwner:
    """
    Minimal host object exercising :func:`~krrood.utils.copy_memoize` under concurrency.
    """

    build_count: int = 0
    """
    Number of times :meth:`build_value` actually ran its body, rather than returning a
    cached copy.
    """

    @copy_memoize
    def build_value(self, payload: int) -> _ReentrancyDetectingValue:
        self.build_count += 1
        return _ReentrancyDetectingValue(payload=payload)


@dataclass(eq=False)
class _MemoizingOwner:
    """
    Minimal host object exercising :func:`~krrood.utils.memoize` under concurrency.
    """

    call_count: int = 0
    """
    Number of times :meth:`compute` actually ran its body, rather than returning a
    cached value.
    """

    @memoize
    def compute(self, key: int) -> int:
        self.call_count += 1
        return key * key


# %% tests


def test_concurrent_copy_memoize_hits_on_the_same_key_do_not_race():
    """
    Many threads calling a `@copy_memoize`-decorated method with the same arguments at
    once must never call ``deepcopy`` on the same cached value concurrently, must run
    the underlying computation exactly once, and must each get back their own,
    independent, value-equal copy.
    """
    owner = _CopyMemoizingOwner()
    thread_count = 16
    results: list[_ReentrancyDetectingValue] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def call() -> None:
        try:
            result = owner.build_value(42)
            with lock:
                results.append(result)
        except BaseException as error:
            with lock:
                errors.append(error)

    threads = [threading.Thread(target=call) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert owner.build_count == 1
    assert len(results) == thread_count
    assert all(result.payload == 42 for result in results)
    assert len({id(result) for result in results}) == thread_count


def test_concurrent_memoize_misses_on_distinct_keys_each_compute_once():
    """
    Many threads calling a `@memoize`-decorated method with distinct arguments at once
    must compute each distinct key's result exactly once, and the cache must end up
    holding exactly one entry per distinct key.
    """
    owner = _MemoizingOwner()
    distinct_keys = range(50)
    callers_per_key = 4
    errors: list[BaseException] = []
    lock = threading.Lock()

    def call(key: int) -> None:
        try:
            owner.compute(key)
        except BaseException as error:
            with lock:
                errors.append(error)

    threads = [
        threading.Thread(target=call, args=(key,))
        for key in distinct_keys
        for _ in range(callers_per_key)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert owner.call_count == len(distinct_keys)
    assert len(owner.__memo__) == len(distinct_keys)
    assert all(owner.compute(key) == key * key for key in distinct_keys)
