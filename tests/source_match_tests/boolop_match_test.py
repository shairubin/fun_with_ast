import pytest

from fun_with_ast.manipulate_node import create_node as create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError, EmptyStackException
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class BoolOpMatcherTest(BaseTestUtils):

    def testAndBoolOp(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.And(),
            create_node.Name('b'))
        string = 'a and b'
        self._assert_match(node, string)

    def testAndBoolOpNoMatch(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.And(),
            create_node.Name('b'))
        string = '(a and b'
        self._assert_no_match(node, string)


    def testAndBoolOp2(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.And(),
            create_node.Name('b'))
        string = '(a and b)'
        self._assert_match(node, string)

    def testAndBoolOpNone(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.And(),
            create_node.CreateNone('None'))
        string = '(a and None)'
        self._assert_match(node, string)

    def testAndBoolOp3(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.And(),
            create_node.Name('b'))
        string = '(a and (b))'
        self._assert_match(node, string)
    def testAndBoolOp4(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.And(),
            create_node.Name('b'))
        string = '(a) and (b)'
        self._assert_match(node, string)

    def testOrBoolOp(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.Or(),
            create_node.Name('b'))
        string = '(    (  (a or b)))'
        self._assert_match(node, string)

    def testOrBoolOpException(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.Or(),
            create_node.Name('b'))
        string = '(    (  (a or b))))'
        self._assert_no_match(node, string)

    def testOrBoolOpException2(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.Or(),
            create_node.Name('b'))
        string = '(    (  (a) or b)))'
        self._assert_no_match(node, string)

    def testAndOrBoolOp(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.And(),
            create_node.Name('b'),
            create_node.Or(),
            create_node.Name('c'))
        string = 'a and b or c'
        self._assert_match(node, string)

    def testOrAndBoolOp2(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            create_node.Or(),
            create_node.Name('b'),
            create_node.And(),
            create_node.Name('c'))
        string = '(a or b) and c'
        self._assert_match(node, string)
    def testOrAndBoolOp3(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            'and',
            create_node.BoolOp(create_node.Name('b'), 'or', create_node.Name('c')))
        string = 'a and (b or c)'
        self._assert_match(node, string)

    def testOrAndBoolOp4(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            'and',
            create_node.BoolOp(create_node.Name('b'), 'or', create_node.Name('c')))
        string = '(a and (b or c))'
        self._assert_match(node, string)

    def testOrAndBoolOp5(self):
        node = create_node.BoolOp(
            create_node.Name('a'),
            'and',
            create_node.BoolOp(create_node.Name('b'), 'or', create_node.Name('c')))
        string = '(a and ((b) or c))'
        self._assert_match(node, string)

    def testOrFromSource(self):
        string = "a or b"
        node = GetNodeFromInput(string)
        self._assert_match(node, string)
    #
    def testOrFromSource2(self):
        string = "a(c) or b(d)\n"
        node = GetNodeFromInput(string)
        self._assert_match(node, string)

    def testOrFromSource3(self):
        string = "isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)\n"
        node = GetNodeFromInput(string)
        self._assert_match(node, string)

#    isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)
    def _assert_no_match(self, node, string):
        matcher = GetDynamicMatcher(node)
        with pytest.raises((BadlySpecifiedTemplateError, EmptyStackException)):
            matcher.do_match(string)

    def _assert_match(self, node, string):
        self._verify_match(node, string)
