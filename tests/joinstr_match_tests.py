import unittest

import create_node
import source_match


class JoinStrMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.JoinedStr([create_node.Str('fun-with-ast')])
        string = 'fun-with-ast'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
#        self.assertEqual(string, matcher.GetSource())
