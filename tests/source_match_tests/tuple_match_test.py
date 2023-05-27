import unittest

import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class TupleTest(unittest.TestCase):

    def testBasicTuple(self):
        node = create_node.Tuple(['a', 'b'])
        string = '(a,b)'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    def testBasicTupleNoParans(self):
        node = create_node.Tuple(['a', 'b'])
        string = 'a,b'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())
    def testBasicTupleNoParansComment(self):
        node = create_node.Tuple(['a', 'b'])
        string = '\t a,\t\tb \t #comment'
        matcher = GetDynamicMatcher(node)
        matcher.Match(string)
        self.assertEqual(string, matcher.GetSource())

    @pytest.mark.skip(reason="Not Implemented Yet - illegal tuple ")
    def testBasicTupleNoParansComment(self):
        node = create_node.Tuple(['a', 'b'])
        string = '(\t a,\t\tb \t #comment'
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
