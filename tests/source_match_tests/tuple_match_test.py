import unittest

import create_node
import source_match
from fun_with_ast.dynamic_matcher import GetDynamicMatcher


class TupleTest(unittest.TestCase):

    def testBasicTuple(self):
        node = create_node.Tuple(['a', 'b'])
        string = '(a,b)'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicSingleTuple(self):
        node = create_node.Tuple(['a'])
        string = '(\t   a, \t)'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testTupleWithCommentAndWS(self):
        node = create_node.Tuple(['a'])
        string = ' (\t   a, \t) \t #comment'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testTupleWithCommentAndWS2(self):
        node = create_node.Tuple(['a', 'b'])
        string = ' (\t   a, b \t)#comment'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testTupleWithCommentAndWSAndConst(self):
        node = create_node.Tuple(['a', 1])
        string = ' (\t   a\t, 1 \t) \t #comment'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
