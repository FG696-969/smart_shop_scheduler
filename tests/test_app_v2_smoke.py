from streamlit.testing.v1 import AppTest


def test_app_exposes_dqn_algorithm_and_main_views():
    app = AppTest.from_file("app.py").run(timeout=30)
    assert not app.exception
    algorithm_options = [option for box in app.selectbox for option in box.options]
    assert "DQN-AOL-GA" in algorithm_options
    tab_labels = [tab.label for tab in app.tabs]
    assert "Scheduling" in tab_labels
    assert "Intelligence" in tab_labels
    assert "Disturbance Analysis" in tab_labels
