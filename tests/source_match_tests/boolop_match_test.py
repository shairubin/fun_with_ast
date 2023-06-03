import unittest

from fun_with_ast.manipulate_node import create_node as create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class BoolOpMatcherTest(unittest.TestCase):

    def testAndBoolOp(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.And(),
            create_node.Name('b'))
        string = 'a and b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testOrBoolOp(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.Or(),
            create_node.Name('b'))
        string = 'a or b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testAndOrBoolOp(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.And(),
            create_node.Name('b'),
            create_node.Or(),
            create_node.Name('c'))
        string = 'a and b or c'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testOrAndBoolOp(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.Or(),
            create_node.Name('b'),
            create_node.And(),
            create_node.Name('c'))
        string = 'a or b and c'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
