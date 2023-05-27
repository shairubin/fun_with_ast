import unittest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.dynamic_matcher import GetDynamicMatcher


class PassMatcherTest(unittest.TestCase):
    def testSimplePass(self):
        node = create_node.Pass()
        string = 'pass'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)
    def testPassWithWS(self):
        node = create_node.Pass()
        string = '   \t pass  \t  '
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)

    def testPassWithWSAndComment(self):
        node = create_node.Pass()
        string = '   \t pass  \t #comment \t '
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)
