import unittest

import create_node
import source_match
from dynamic_matcher import GetDynamicMatcher


class LambdaMatcherTest(unittest.TestCase):

    def testBasicMatch(self):
        node = create_node.Lambda(create_node.Pass(), args=['a'])
        string = 'lambda a:\tpass\n'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchWithArgs(self):
        node = create_node.Lambda(
            create_node.Name('a'),
            args=['b'])
        string = 'lambda b: a'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMatchWithArgsOnNewLine(self):
        node = create_node.Lambda(
            create_node.Name('a'),
            args=['b'])
        string = '(lambda\nb: a)'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())