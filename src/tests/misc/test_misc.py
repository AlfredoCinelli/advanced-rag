# Import packages and modules

import pytest
from src.utils.misc import format_docs
from src.tests.misc.data import scenario


# Define test functions
@pytest.mark.parametrize("scenario", scenario("format_docs"))
def test_format_docs(
    scenario: dict[str, any],
) -> None:
    """Test format_docs function."""
    output = format_docs(scenario.get("documents"))
    assert output == scenario.get("expected_output")
