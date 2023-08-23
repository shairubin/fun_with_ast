import unittest

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.base_matcher import SourceMatcher
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError, EmptyStackException

from fun_with_ast.manipulate_node import create_node
from tests.source_match_tests.base_test_utils import BaseTestUtils

@pytest.fixture(autouse=True)
def run_around_tests(): # TODO not very smart global variable
    SourceMatcher.parentheses_stack.reset()
    yield

class DictMatcherTest(BaseTestUtils):


    def testBasicDictMatch(self):
        node = create_node.Num('1')
        string = """deepnet_gain = {
        "encoder": {
            "alpha": lambda config: 0.81
            * (config.encoder_layers**4 * config.decoder_layers) ** 0.0625,
            "beta": lambda config: 0.87
            * (config.encoder_layers**4 * config.decoder_layers) ** -0.0625,
        },
        "decoder": {
            "alpha": lambda config: (3 * config.decoder_layers) ** 0.25,
            "beta": lambda config: (12 * config.decoder_layers) ** -0.25,
        },
}"""
        node = GetNodeFromInput(string)

        self._verify_match(node, string)

