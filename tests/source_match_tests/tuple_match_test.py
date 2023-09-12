import unittest

import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
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

    def testBasicTupleNoIllegal(self):
        node = create_node.Tuple(['a', 'b'])
        string = '(\t a,\t\tb \t #comment'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)

    def testBasicSingleTuple(self):
        node = create_node.Tuple(['a'])
        string = '(\t   a, \t)'
        self._assert_match(node, string)
    def testBasicSingleTuple3(self):
        node = create_node.Tuple(['a'])
        string = '(a,)'
        self._assert_match(node, string)

    def testBasicSingleTuple4(self):
        node = create_node.Tuple(['a'])
        string = 'a,'
        self._assert_match(node, string)

    def testBasicSingleTuple2(self):
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
    def testCreateNodeFromInput(self):
        string = '(\t   a\t, 1 \t) \t #comment'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput1(self):
        string = '(a, 1,) \t #comment'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput11(self):
        string = '(a, 1)  #comment'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInput2(self):
        string = '(a,)'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput3(self):
        string = '(a,  \t b,   )'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput4(self):
        string = 'a,  \t b,   '
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput5(self):
        string = '-1,'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInput6(self):
        string = '(-1,)'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL(self):
        string = '((1,2), (3,4))'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL2(self):
        string = '((1,2), (3,4),)'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)
    def testCreateNodeFromInputTupleWithEOL21(self):
        string = '((1,2), (3,4),)\n'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL22(self):
        string = '((1,2), (3,4),)\n     '
        node =GetNodeFromInput(string, get_module=True)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL23(self):
        string = '(1,2)\n     '
        node =GetNodeFromInput(string, get_module=True)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL24(self): # empty line at end is supported only in modules.
        string = '(1,2)\n     '
        node =GetNodeFromInput(string, get_module=False)
        with pytest.raises(AssertionError):
            self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL3(self):
        string = '((1,2),\n (3,4))'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def testCreateNodeFromInputTupleWithEOL31(self):
        string = '(1,\n 2)'
        node =GetNodeFromInput(string)
        self._assert_match(node, string)

    def _assert_match(self, node, string):
        self._verify_match(node, string)
