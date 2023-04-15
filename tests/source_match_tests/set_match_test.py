import unittest

import create_node
import source_match


class SetMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Set('c', 'a', 'b')
        string = '{c, a, b}'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
