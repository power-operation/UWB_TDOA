def pytest_addoption(parser):
    parser.addoption(
        "--target", action="store", default=None, help="Specify target module to test"
    )

def pytest_configure(config):
    target = config.getoption("--target")
    if target:
        print(f"\n🧪 [pytest] Running tests for target: {target}")
