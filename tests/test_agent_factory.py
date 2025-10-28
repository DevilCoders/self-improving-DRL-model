import torch

from drl_system import SystemConfig, create_agent


def test_agent_factory_creates_a3c():
    config = SystemConfig()
    config.agent.type = "a3c"
    agent = create_agent(config.agent, config, obs_dim=4, action_dim=4, device=torch.device("cpu"))
    assert agent.__class__.__name__ == "A3CAgent"


def test_agent_factory_creates_sac():
    config = SystemConfig()
    config.agent.type = "sac"
    agent = create_agent(config.agent, config, obs_dim=4, action_dim=4, device=torch.device("cpu"))
    assert agent.__class__.__name__ == "SACAgent"


def test_agent_factory_creates_dqn():
    config = SystemConfig()
    config.agent.type = "dqn"
    agent = create_agent(config.agent, config, obs_dim=4, action_dim=4, device=torch.device("cpu"))
    assert agent.__class__.__name__ == "DQNAgent"


def test_agent_factory_creates_ddpg():
    config = SystemConfig()
    config.agent.type = "ddpg"
    agent = create_agent(config.agent, config, obs_dim=4, action_dim=4, device=torch.device("cpu"))
    assert agent.__class__.__name__ == "DDPGAgent"


def test_agent_factory_creates_td3():
    config = SystemConfig()
    config.agent.type = "td3"
    agent = create_agent(config.agent, config, obs_dim=4, action_dim=4, device=torch.device("cpu"))
    assert agent.__class__.__name__ == "TD3Agent"


def test_agent_factory_creates_reinforce():
    config = SystemConfig()
    config.agent.type = "reinforce"
    agent = create_agent(config.agent, config, obs_dim=4, action_dim=4, device=torch.device("cpu"))
    assert agent.__class__.__name__ == "ReinforceAgent"


def test_agent_factory_creates_quantile_dqn():
    config = SystemConfig()
    config.agent.type = "quantile_dqn"
    agent = create_agent(config.agent, config, obs_dim=4, action_dim=4, device=torch.device("cpu"))
    assert agent.__class__.__name__ == "QuantileDQNAgent"


def test_agent_prepare_batch_shapes():
    config = SystemConfig()
    agent = create_agent(config.agent, config, obs_dim=3, action_dim=3, device=torch.device("cpu"))
    from drl_system.memory.replay_buffer import Transition

    transitions = [
        Transition(
            state=torch.zeros(3).numpy(),
            action=torch.zeros(1).numpy(),
            reward=0.0,
            next_state=torch.zeros(3).numpy(),
            done=False,
            info={"log_prob": 0.0},
        )
        for _ in range(4)
    ]
    advantages = torch.zeros(4)
    returns = torch.zeros(4)
    batch = agent.prepare_batch(transitions, advantages, returns)
    assert batch.observations.shape == (4, 3)
    assert batch.actions.shape[0] == 4


def test_policy_network_emits_quantile_diagnostics():
    config = SystemConfig()
    config.agent.quantile_atoms = 16
    agent = create_agent(config.agent, config, obs_dim=5, action_dim=2, device=torch.device("cpu"))
    observations = torch.zeros(2, 5)
    _, _, _, _, diagnostics = agent.model(observations)
    assert diagnostics["quantiles"].shape == (2, 2, 16)
    assert diagnostics["risk"].shape == (2, 2)
