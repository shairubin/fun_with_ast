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
    def testSimpleDel2(self):
        string = "del \t a"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testSimpleDel3(self):
        string = "del \t a \t "
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testSimpleDel4(self):
        string = "del \t a \t #comment "
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testListDel(self):
        string = "del a, b"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
