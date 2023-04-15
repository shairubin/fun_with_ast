import unittest

import create_node
import source_match


class SubscriptMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Subscript('a', 1)
        string = 'a[1]'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual('a[1]', matcher.GetSource())

    def testAllPartsMatch(self):
        node = create_node.Subscript('a', 1, 2, 3)
        string = 'a[1:2:3]'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual('a[1:2:3]', matcher.GetSource())

    def testSeparatedWithStrings(self):
        node = create_node.Subscript('a', 1, 2, 3)
        string = 'a [ 1 : 2 : 3 ]'
        matcher = source_match.GetMatcher(node)
        matcher.Match(string)
        self.assertEqual('a [ 1 : 2 : 3 ]', matcher.GetSource())
