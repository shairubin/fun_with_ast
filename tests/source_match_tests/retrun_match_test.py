import unittest

import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.dynamic_matcher import GetDynamicMatcher


class PassMatcherTest(unittest.TestCase):
    def testSimpleReturn(self):
        node = create_node.Return(1)
        string = 'return 1'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)

    def testReturnStr(self):
        node = create_node.Return('1')
        string = "return '1'"
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)

    def testReturnName(self):
        node = create_node.Return(create_node.Name('a'))
        string = "return a"
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)

    def testReturnTuple(self):
        node = create_node.Return(create_node.Tuple(['a', 'b']))
        string = "return (a,b)"
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)

    @pytest.mark.skip(reason="Not Implemented Yet")
    def testReturnTupleNoParans(self):
        node = create_node.Return(create_node.Tuple(['a', 'b']))
        string = "return a,b"
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)
