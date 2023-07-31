import unittest

import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class TupleTest(BaseTestUtils):

    def testBasicTuple(self):
        node = create_node.Tuple(['a', 'b'])
        string = '(a,b)'
        self._assert_match(node, string)

    def testBasicTupleNone(self):
        node = create_node.Tuple(['a', 'None'])
        string = '(a,None)'
        self._assert_match(node, string)


    def testBasicTupleNoParans(self):
        node = create_node.Tuple(['a', 'b'])
        string = 'a,b'
        self._assert_match(node, string)

    def testBasicTupleNoParansComment(self):
        node = create_node.Tuple(['a', 'b'])
        string = '\t a,\t\tb \t #comment'
        self._assert_match(node, string)

    #@pytest.mark.skip(reason="Not Implemented Yet - illegal tuple ")
    def testBasicTupleNoIllegal(self):
        node = create_node.Tuple(['a', 'b'])
        string = '(\t a,\t\tb \t #comment'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)
        #self.assertEqual(string, matcher.GetSource())

    # @pytest.mark.skip(reason="Not Implemented Yet - comma without another name")
    def testBasicSingleTuple(self):
        node = create_node.Tuple(['a'])
        string = '(\t   a, \t)'
        self._assert_match(node, string)

    def testBasicSingleTuple(self):
        node = create_node.Tuple(['a'])
        string = '(\t   a \t)'
        self._assert_match(node, string)

    def testTupleWithCommentAndWS2(self):
        node = create_node.Tuple(['a', 'b'])
        string = ' (\t   a, b \t)#comment'
        self._assert_match(node, string)

    def testTupleWithCommentAndWSAndConst(self):
        node = create_node.Tuple(['a', 1])
        string = ' (\t   a\t, 1 \t) \t #comment'
        self._assert_match(node, string)

    def _assert_match(self, node, string):
        self._verify_match(node, string)
