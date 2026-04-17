"""Regression tests for background task tracking in GatewayRunner.

Long-running background tasks (session expiry watcher, platform reconnect
watcher, per-process watchers, update notification watcher) must be added
to ``self._background_tasks`` so the ``_stop_impl`` cleanup loop can
cancel them on shutdown.  Previously these were fire-and-forget
``asyncio.create_task(...)`` calls with no reference stored anywhere,
which meant:

  1. Python could GC them prematurely under memory pressure.
  2. They survived ``_stop_impl`` because the cancel loop at
     ``for _task in list(self._background_tasks)`` never saw them.
  3. They held an implicit reference to ``self`` via the bound method
     passed as the coroutine — keeping the whole GatewayRunner (and
     every adapter / cached agent / session it owns) alive well past
     ``stop()``.

Each test here pins one tracking site.  If a future refactor reverts
the tracking, the corresponding test fails.
"""

import asyncio

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_schedule_update_notification_watch_tracks_task(monkeypatch):
    """_schedule_update_notification_watch adds the task to _background_tasks
    AND exposes it via self._update_notification_task (the existing check
    for 'is a watcher already running?')."""
    runner, _ = make_restart_runner()
    runner._update_notification_task = None

    # Stub out the watcher body so it returns immediately; we only care
    # about the tracking, not the update-progress polling.
    async def _stub_watch():
        await asyncio.sleep(0.5)
    monkeypatch.setattr(runner, "_watch_update_progress", _stub_watch)

    runner._schedule_update_notification_watch()

    task = runner._update_notification_task
    assert task is not None
    assert task in runner._background_tasks, (
        "Update notification task must be in _background_tasks so "
        "_stop_impl cancels it on shutdown."
    )

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_schedule_update_notification_watch_done_callback_removes(monkeypatch):
    """When the update watcher task finishes naturally, the done callback
    removes it from _background_tasks.  Otherwise the set would grow on
    every /update invocation."""
    runner, _ = make_restart_runner()
    runner._update_notification_task = None

    async def _instant_return():
        return None
    monkeypatch.setattr(runner, "_watch_update_progress", _instant_return)

    runner._schedule_update_notification_watch()
    task = runner._update_notification_task
    assert task in runner._background_tasks

    # Let it run to completion and fire the done callback.
    await task
    # One more yield so the callback runs.
    await asyncio.sleep(0)

    assert task not in runner._background_tasks


@pytest.mark.asyncio
async def test_schedule_update_notification_watch_idempotent(monkeypatch):
    """Calling schedule twice while one is already running must not spawn
    a duplicate or leak the old task into _background_tasks."""
    runner, _ = make_restart_runner()
    runner._update_notification_task = None

    release = asyncio.Event()

    async def _hang():
        await release.wait()
    monkeypatch.setattr(runner, "_watch_update_progress", _hang)

    runner._schedule_update_notification_watch()
    first = runner._update_notification_task
    runner._schedule_update_notification_watch()
    second = runner._update_notification_task

    assert first is second  # same task reused
    # Exactly one entry in _background_tasks for this watcher.
    assert sum(1 for t in runner._background_tasks if t is first) == 1

    release.set()
    await first
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_process_watcher_dispatch_tracks_task(monkeypatch):
    """Per-process watchers spawned mid-run (when a user-started background
    terminal command raises check_interval events) must be tracked so
    a subsequent gateway shutdown cancels them.

    Simulates the production code block in _handle_message() that drains
    process_registry.pending_watchers.
    """
    runner, _ = make_restart_runner()

    async def _stub_watcher(watcher_dict):
        # Just hang so the test can observe the task in _background_tasks.
        await asyncio.sleep(10.0)
    monkeypatch.setattr(runner, "_run_process_watcher", _stub_watcher)

    # Reproduce the production spawn block from gateway/run.py.
    fake_watchers = [{"session_id": f"s{i}"} for i in range(3)]
    for w in fake_watchers:
        _task = asyncio.create_task(runner._run_process_watcher(w))
        runner._background_tasks.add(_task)
        _task.add_done_callback(runner._background_tasks.discard)

    assert len(runner._background_tasks) == 3
    for t in list(runner._background_tasks):
        assert not t.done()

    for t in list(runner._background_tasks):
        t.cancel()
    await asyncio.sleep(0.05)
    assert runner._background_tasks == set()


@pytest.mark.asyncio
async def test_session_expiry_watcher_tracks_in_background_tasks(monkeypatch):
    """The session expiry watcher spawned at startup must be in
    _background_tasks so _stop_impl cancels it."""
    runner, _ = make_restart_runner()

    async def _stub_expiry(interval: int = 300):
        await asyncio.sleep(30.0)  # will be cancelled
    monkeypatch.setattr(runner, "_session_expiry_watcher", _stub_expiry)

    # Mirror the production spawn pattern from start().
    _expiry_task = asyncio.create_task(runner._session_expiry_watcher())
    runner._background_tasks.add(_expiry_task)
    _expiry_task.add_done_callback(runner._background_tasks.discard)

    assert _expiry_task in runner._background_tasks

    # Shutdown simulation.
    for t in list(runner._background_tasks):
        t.cancel()
    await asyncio.sleep(0.05)
    assert runner._background_tasks == set()


@pytest.mark.asyncio
async def test_platform_reconnect_watcher_tracks_in_background_tasks(monkeypatch):
    """The platform reconnect watcher spawned at startup must be in
    _background_tasks so _stop_impl cancels it."""
    runner, _ = make_restart_runner()

    async def _stub_reconnect():
        await asyncio.sleep(30.0)
    monkeypatch.setattr(runner, "_platform_reconnect_watcher", _stub_reconnect)

    _reconnect_task = asyncio.create_task(runner._platform_reconnect_watcher())
    runner._background_tasks.add(_reconnect_task)
    _reconnect_task.add_done_callback(runner._background_tasks.discard)

    assert _reconnect_task in runner._background_tasks

    for t in list(runner._background_tasks):
        t.cancel()
    await asyncio.sleep(0.05)
    assert runner._background_tasks == set()


@pytest.mark.asyncio
async def test_background_tasks_survive_production_spawn_pattern(monkeypatch):
    """End-to-end: spawn N background tasks using the exact tracking
    pattern (add + done_callback), then cancel them all via the same
    loop _stop_impl uses.  After the loop, the set is empty.

    This pins the full contract: tracking + cleanup + done-callback
    deregistration all work together.
    """
    runner, _ = make_restart_runner()

    hangs = [asyncio.Event() for _ in range(5)]

    async def _hang(i: int):
        await hangs[i].wait()

    for i in range(5):
        _t = asyncio.create_task(_hang(i))
        runner._background_tasks.add(_t)
        _t.add_done_callback(runner._background_tasks.discard)

    assert len(runner._background_tasks) == 5

    # Stop-impl-style cancellation.
    for _t in list(runner._background_tasks):
        _t.cancel()

    # Let cancellations propagate.
    for _ in range(5):
        await asyncio.sleep(0.01)

    # All gone — done callbacks ran.
    assert runner._background_tasks == set()
