import unittest

import pytest
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

from fun_with_ast.manipulate_node import create_node as create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class CompareMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Compare(
            create_node.Call('a.b', args=[create_node.Num('2')]),
            '>=',
            create_node.Num('1'))
        string = 'a.b(2) >= 1'
        self._verify_match(node, string)
    def testBasicMatchIs(self):
        node = create_node.Compare(
            create_node.Call('a.b', args=[create_node.Num('2')]),
            'is',
            create_node.Num('1'))
        string = 'a.b(2) is 1'
        self._verify_match(node, string)

    def testBasicMatchIsNone(self):
        node = create_node.Compare(
            create_node.Call('a.b', args=[create_node.Num('2')]),
            'is',
            create_node.CreateNone('None'))
        string = 'a.b(2) is None'
        self._verify_match(node, string)

    def testBasicMatchIsNot(self):
        node = create_node.Compare(
            create_node.Call('a.b', args=[create_node.Num('2')]),
            'is not',
            create_node.Num('1'))
        string = 'a.b(2) is       not 1'
        self._verify_match(node, string)
    def testBasicNoMatchIs(self):
        node = create_node.Compare(
            create_node.Call('a.b', args=[create_node.Num('2')]),
            'is',
            create_node.Num('1'))
        string = 'a.b(2) is 2'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testBasicMatch4(self):
        node = create_node.Compare(
            create_node.Call('a.b'),
            '>=',
            create_node.Num('1'))
        string = '(a.b() >= 1)'
        self._verify_match(node, string)
    def testBasicMatch5(self):
        node = create_node.Compare(
            create_node.Call('a.b'),
            '>=',
            create_node.Num('1'))
        string = 'a.b() >= 1'
        self._verify_match(node, string)
    def testBasicMatch3(self):
        node = create_node.Compare(
            create_node.Call('a.b', args=[create_node.Num('2')]),
            '>=',
            create_node.Num('1'))
        string = '(a.b(2) >= 1)'
        self._verify_match(node, string)


    def testBasicMatch2(self):
        node = create_node.Compare(
            create_node.Name('a'),
            '>=',
            create_node.Num('1'))
        string = '(a >= 1)'
        self._verify_match(node, string)

