import unittest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class SubscriptMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Subscript('a', 1)
        string = 'a[1]'
        matcher = GetDynamicMatcher(node)
        matcher._match(string)
        self.assertEqual('a[1]', matcher.GetSource())

    def testAllPartsMatch(self):
        node = create_node.Subscript('a', 1, 2, 3)
        string = 'a[1:2:3]'
        matcher = GetDynamicMatcher(node)
        matcher._match(string)
        self.assertEqual('a[1:2:3]', matcher.GetSource())

    def testSeparatedWithStrings(self):
        node = create_node.Subscript('a', 1, 2, 3)
        string = 'a [ 1 : 2 : 3 ]'
        matcher = GetDynamicMatcher(node)
        matcher._match(string)
        self.assertEqual('a [ 1 : 2 : 3 ]', matcher.GetSource())
