def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: exercises the real tabpfn package; run in the integration CI job",
    )
