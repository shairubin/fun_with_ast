import unittest

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class AnnAssignMatcherTest(BaseTestUtils):

    def testAnnAssignFromSource(self):
        string = 'a: int =1'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def _assert_matched_source(self, node, string):
        self._verify_match(node, string)
