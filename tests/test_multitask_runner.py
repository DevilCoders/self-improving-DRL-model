import time

from drl_system.runners.multitask_runner import MultiTaskScheduler


def test_scheduler_runs_tasks_and_gathers_results():
    scheduler = MultiTaskScheduler(max_workers=2)
    scheduler.schedule(lambda x: x + 1, 1, description="increment")
    scheduler.schedule(time.sleep, 0.01, description="sleep")
    results = scheduler.gather(timeout=0.5)
    scheduler.shutdown()
    descriptions = {result.description for result in results}
    assert {"increment", "sleep"}.issubset(descriptions)
    increment_result = next(result for result in results if result.description == "increment")
    assert increment_result.result == 2
