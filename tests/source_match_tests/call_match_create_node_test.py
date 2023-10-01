import unittest

import pytest

from fun_with_ast.manipulate_node import create_node as create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from tests.source_match_tests.base_test_utils import BaseTestUtils


class CallMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Call('a')
        string = 'a()'
        self._verify_match(node, string)
    def testBasicMatchWarp(self):
        node = create_node.Call('a')
        string = '(a())'
        self._verify_match(node, string)
    def testBasicMatchWS(self):
        node = create_node.Call('a')
        string = ' a()'
        self._verify_match(node, string)

    def testBasicMatchWS2(self):
        node = create_node.Call('a.b')
        string = ' a.b()'
        self._verify_match(node, string)
    def testMatchStarargs(self):
        node = create_node.Call('a', args=[create_node.Starred('args')])
        string = 'a(*args)'
        self._verify_match(node, string)
    def testMatchStarargs2(self):
        node = create_node.Call('a', args=[create_node.Name('b'), create_node.Starred('args')])
        string = 'a(b, *args)'
        self._verify_match(node, string)

    def testNoMatchStarargs(self):
        node = create_node.Call('a', args=[create_node.Name('b'), create_node.Starred('arrrggs')])
        string = 'a(b, *args)'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)
    def testNoMatchStarargs2(self):
        node = create_node.Call('a', args=[create_node.Name('c'), create_node.Starred('args')])
        string = 'a(b, *args)'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testMatchWithStarargsBeforeKeyword(self):
        node = create_node.Call('a', args=[create_node.Name('d')], keywords=[create_node.keyword('b', 'c')])
        string = 'a(d \t , \t b= c)'
        self._verify_match(node, string)
    def testMatchWithStarargsBeforeKeyword2(self):
        node = create_node.Call('a', args=[create_node.Stared('fun-with-ast')],
                                keywords=[create_node.keyword('b', 'c'), create_node.keyword('e', 'f')])
        string = 'a(*fun-with-ast, b=c, e = f)'
        self._verify_match(node, string)

    def testMatchWithStarargsBeforeKeyword3(self):
        node = create_node.Call('a', args=[create_node.Name('d'), create_node.Stared('starred')],
                                keywords=[create_node.keyword('b', 'c'), create_node.keyword('e', 'f')])
        string = 'a(d,   *starred, b=c, e = f )'
        self._verify_match(node, string)


    def testMatchKeywordOnly(self):
        node = create_node.Call('a', keywords=[create_node.keyword('b', 'c')])
        string = 'a(b=c)'
        self._verify_match(node, string)

    def testCallWithAttribute(self):
        node = create_node.Call('a.b')
        string = 'a.b()'
        self._verify_match(node, string)

    def testCallWithAttributeAndParam(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "'")])
        string = 'a.b(\'fun-with-ast\')'
        self._verify_match(node, string)

    def testCallWithAttributeAndNone(self):
        node = create_node.Call('a.b', args=[create_node.CreateNone('None')])
        string = 'a.b(None)'
        self._verify_match(node, string)

    def testCallWithAttributeAndNoneNoMatch(self):
        node = create_node.Call('a.b', args=[create_node.CreateNone('None')])
        string = 'a.b(none)'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testCallWithAttributeAndParamAndQuate(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "\"")])
        string = "a.b(\"fun-with-ast\")"
        self._verify_match(node, string)

    def testNoMatchCallWithAttributeAndParamAndQuate(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "'")])
        string = "a.b(\"fun-with-ast\")"
        self._verify_match(node, string)
    def testNoMatchCallWithAttributeAndParamAndQuate2(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "\"")]) #TODO: do not need second param
        string = "a.b('fun-with-ast')"
        self._verify_match(node, string)

    def testCallWithAttributeAndParam2(self):
        node = create_node.Call('a.b', args=[create_node.Num('1')])
        string = 'a.b(1)'
        self._verify_match(node, string)
    def testCallWithAttributeAndParam4(self):
        node = create_node.Call('a.b', args=[create_node.Num('1'), create_node.Num('2')])
        string = 'a.b(1,2)'
        self._verify_match(node, string)

    def testCallWithAttributeAndParam5(self):
        node = create_node.Call('a', args=[create_node.Num('1'), create_node.Num('2')])
        string = 'a( 1,2)'
        self._verify_match(node, string)

    def testCallWithAttributeAndParam6(self):
        node = create_node.Call('a', args=[create_node.Num('1'), create_node.Num('2')])
        string = 'a(1,2)'
        self._verify_match(node, string)

    def testCallWithAttributeAndParam7(self):
        node = create_node.Call('a', args=[create_node.Num('1')])
        string = 'a(1)'
        self._verify_match(node, string)
    def testCallWithAttributeAndParam3(self):
        node = create_node.Call('a.b', args=[create_node.Num('1')])
        string = '(a.b(1))'
        self._verify_match(node, string)

    def testCallWithMultiLines(self):
        string = "fileparse.parse_csv(lines,\n \
                                     select=['name', 'shares', 'price'],\n \
                                     types=[str, int, float],\n \
                                    **opts)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithAttributeAndParamWS(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "'")])
        string = 'a.b(\'fun-with-ast\')'
        self._verify_match(node, string)
