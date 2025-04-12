import pytest
import pytest_socket


@pytest.fixture(scope="package", autouse=True)
def _disable_socket_access():
    """
    Ensure all unit tests cannot access any of the databases.
    """
    pytest_socket.disable_socket()
