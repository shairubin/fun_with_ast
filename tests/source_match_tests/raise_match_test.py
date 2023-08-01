import unittest

import pytest
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class RaiseMatcherTest(BaseTestUtils):
    @pytest.mark.skip(reason="Issue 42")
    def testSimpleRaise(self):
        string = 'raise e'
        node = GetNodeFromInput(string)
        self._assert_match(node, string)

