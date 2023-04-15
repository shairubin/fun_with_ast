import unittest

import create_node
import source_match


class PassMatcherTest(unittest.TestCase):
    def testSimplePass(self):
        node = create_node.Pass()
        string = 'pass'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)
    def testPassWithWS(self):
        node = create_node.Pass()
        string = '   \t pass  \t  '
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)

    def testPassWithWSAndComment(self):
        node = create_node.Pass()
        string = '   \t pass  \t #comment \t '
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        matcher_source = matcher.GetSource()
        self.assertEqual(string, matcher_source)
