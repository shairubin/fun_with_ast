import unittest

from fun_with_ast.manipulate_node import create_node as create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class CallMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Call('a')
        string = 'a()'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchStarargs(self):
        node = create_node.Call('a', starargs='args')
        string = 'a(*args)'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchWithStarargsBeforeKeyword(self):
        node = create_node.Call('a', keywords=[create_node.keyword('b', 'c')], starargs='args')
        string = 'a(*args, b=c)'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testCallWithAttribute(self):
        node = create_node.Call('a.b')
        string = 'a.b()'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testCallWithAttributeAndParam(self):
        node = create_node.Call('a.b', args=[create_node.Str('fun-with-ast')])
        string = 'a.b(\'fun-with-ast\')'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
