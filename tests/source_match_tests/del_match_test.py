import unittest

import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class DelMatcherTest(BaseTestUtils):

    def testSimpleDel(self):
        string = "del a"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
