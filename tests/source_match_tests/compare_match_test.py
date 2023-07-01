import unittest

import pytest

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
    @pytesadd tests t.mark.skip('not implemented, see anso: testIfFromSource7')
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

