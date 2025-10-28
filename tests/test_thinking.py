from drl_system.config import ThinkingConfig
from drl_system.self_improvement.thinking import DeliberationEngine


def test_deliberation_engine_limits_steps_and_summarizes():
    config = ThinkingConfig(max_steps=4)
    engine = DeliberationEngine(config)
    trace = engine.start_trace("unit-test", mode="deep")
    for idx in range(6):
        engine.add_step(trace.trace_id, f"thought {idx}", step=idx)
    stored = engine.get_trace(trace.trace_id)
    assert stored is not None
    # One step is automatically inserted at creation time.
    assert len(stored.steps) <= config.max_steps
    summary = engine.summarize(trace.trace_id)
    assert "thought" in summary


def test_deliberation_engine_records_metrics():
    engine = DeliberationEngine(ThinkingConfig())
    summary = engine.record_metrics({"loss": 0.1, "entropy": 0.5}, description="update")
    assert summary is not None
    assert "loss=" in summary
