import unittest

import pytest
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class RetrunMatcherTest(BaseTestUtils):
    def testSimpleReturn(self):
        node = create_node.Return(1)
        string = 'return 1'
        self._assert_match(node, string)

    def testSimpleReturnFromString(self):
        string = 'return 1'
        node = GetNodeFromInput(string)
        self._assert_match(node, string)

    def testSimpleReturnFromString2(self):
        string = 'return  isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)'
        node = GetNodeFromInput(string)
        self._assert_match(node, string)


    def testReturnStr(self):
        node = create_node.Return("1", "'")
        string = "return '1'"
        self._assert_match(node, string)

    def testReturnStrDoubleQuote(self):
        node = create_node.Return('1', "\"")
        string = "return \"1\""
        self._assert_match(node, string)

    def testReturnName(self):
        node = create_node.Return(create_node.Name('a'))
        string = "return a"
        self._assert_match(node, string)

    def testReturnTuple(self):
        node = create_node.Return(create_node.Tuple(['a', 'b']))
        string = "return (a,b)"
        self._assert_match(node, string)

    def testReturnTupleNoParans(self):
        node = create_node.Return(create_node.Tuple(['a', 'b']))
        string = "return a,b"
        self._assert_match(node, string)

    def _assert_match(self, node, string):
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)
