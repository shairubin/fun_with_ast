import unittest

from fun_with_ast.manipulate_node import create_node as create_node
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
        node = create_node.Call('a', keywords=[create_node.keyword('b', 'c')], starargs='args')
        string = 'a(*args, b=c)'
        self._verify_match(node, string)

    def testCallWithAttribute(self):
        node = create_node.Call('a.b')
        string = 'a.b()'
        self._verify_match(node, string)

    def testCallWithAttributeAndParam(self):
        node = create_node.Call('a.b', args=[create_node.Str('fun-with-ast')])
        string = 'a.b(\'fun-with-ast\')'
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
    def testCallWithAttributeAndParam3(self):
        node = create_node.Call('a.b', args=[create_node.Num('1')])
        string = '(a.b(1))'
        self._verify_match(node, string)

    def testCallWithAttributeAndParamWS(self):
        node = create_node.Call('a.b', args=[create_node.Str('fun-with-ast')])
        string = 'a.b(\'fun-with-ast\')'
        self._verify_match(node, string)
