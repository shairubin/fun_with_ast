import unittest

import create_node
import source_match
from fun_with_ast.dynamic_matcher import GetDynamicMatcher


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
