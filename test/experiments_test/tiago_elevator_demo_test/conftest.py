import pytest

from .scripted_social_navigation import ScriptedSocialNavigation

# %% the social navigation module


@pytest.fixture
def social_navigation(rclpy_node, request):
    module = ScriptedSocialNavigation(node=rclpy_node, answer=request.param)
    yield module
    module.close()
