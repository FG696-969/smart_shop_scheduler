def test_v2_packages_import():
    import rl
    import services

    assert rl is not None
    assert services is not None
