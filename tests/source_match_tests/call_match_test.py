import unittest

import pytest

from fun_with_ast.manipulate_node import create_node as create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
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
        node = create_node.Call('a', starargs='args')
        string = 'a(*args)'
        self._verify_match(node, string)

    def testMatchWithStarargsBeforeKeyword(self):
        node = create_node.Call('a', keywords=[create_node.keyword('b', 'c')], starargs='started')
        string = 'a(*started, b=c)'
        self._verify_match(node, string)
    def testMatchWithStarargsBeforeKeyword2(self):
        node = create_node.Call('a', keywords=[create_node.keyword('b', 'c'), create_node.keyword('e', 'f')], starargs='started')
        string = 'a(*started, b=c, e = f)'
        self._verify_match(node, string)

    def testMatchWithStarargsBeforeKeyword3(self):
        node = create_node.Call('a', keywords=[create_node.keyword('b', 'c'), create_node.keyword('e', 'f')], starargs='started')
        string = 'a(   *started, b=c, e = f )'
        self._verify_match(node, string)

    def testCallWithAttribute(self):
        node = create_node.Call('a.b')
        string = 'a.b()'
        self._verify_match(node, string)

    def testCallWithAttributeAndParam(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "'")])
        string = 'a.b(\'fun-with-ast\')'
        self._verify_match(node, string)

    def testCallWithAttributeAndParamAndQuate(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "\"")])
        string = "a.b(\"fun-with-ast\")"
        self._verify_match(node, string)

    def testNoMatchCallWithAttributeAndParamAndQuate(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "'")])
        string = "a.b(\"fun-with-ast\")"
        from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)
    def testNoMatchCallWithAttributeAndParamAndQuate2(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "\"")])
        string = "a.b('fun-with-ast')"
        from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testCallWithAttributeAndParam2(self):
        node = create_node.Call('a.b', args=[create_node.Num('1')])
        string = 'a.b(1)'
        self._verify_match(node, string)
    def testCallWithAttributeAndParam4(self):
        node = create_node.Call('a.b', args=[create_node.Num('1'), create_node.Num('2')])
        string = 'a.b(1,2)'
        self._verify_match(node, string)
    #@pytest.mark.skip('issue #5 should solve it')
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

    @pytest.mark.skip('reproduce issue portfolio.py issue')
    def testCallWithMultiLines(self):
        string = "fileparse.parse_csv(lines,\n \
                                     select=['name', 'shares', 'price'],\n \
                                     types=[str, int, float],\n \
                                    **opts)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallWithMultiLinesSimple(self):
        string = "fileparse.parse_csv(lines,\n \
                                      a)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithMultiLinesSimple2(self):
        string = "fileparse.parse_csv(lines)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    @pytest.mark.skip('reproduce issue portfolio.py issue')
    def testCallWithMultiLinesSimple3(self):
        string = "a.b(c,\n \
                       d=[e])\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    @pytest.mark.skip('reproduce issue portfolio.py issue')
    def testCallWithMultiLinesSimple4(self):
        string = "a.b(d=[])\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)



    def testCallWithAttributeAndParamWS(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "'")])
        string = 'a.b(\'fun-with-ast\')'
        self._verify_match(node, string)
