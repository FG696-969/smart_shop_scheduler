from visualization import action_distribution_figure, dqn_learning_figure, q_value_figure


def test_dqn_learning_figure_exposes_reward_loss_and_epsilon():
    figure = dqn_learning_figure([0.2, 0.4], [0.8, 0.3], [1.0, 0.5])
    assert [trace.name for trace in figure.data] == ["Reward", "Loss", "Epsilon"]


def test_action_and_q_figures_handle_real_traces():
    actions = action_distribution_figure(["Balanced GA", "Balanced GA", "Insertion mutation"])
    assert sum(actions.data[0].y) == 3

    q_values = q_value_figure([[1.0, 0.5], [1.2, 0.7]])
    assert len(q_values.data) == 2

