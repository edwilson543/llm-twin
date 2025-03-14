from llm_twin import main


def test_returns_thirty_one():
    result = main.main()

    assert result == 31
