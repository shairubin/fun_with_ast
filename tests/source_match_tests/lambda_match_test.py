import unittest

import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class LambdaMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Lambda(create_node.Pass(), args=['a'])
        string = 'lambda a:\tpass\n'
        self._verify_match(node, string)

    def testMatchWithArgs(self):
        node = create_node.Lambda(
            create_node.Name('a'),
            args=['b'])
        string = 'lambda b: a'
        self._verify_match(node, string)

    def testMatchWithArgsOnNewLine(self):
        node = create_node.Lambda(
            create_node.Name('a'),
            args=['b'])
        string = '(lambda\nb: a)'
        self._verify_match(node, string)

    def testMatchComplexLambda(self):
        string = """lambda g, ans, x: lax.select(
            x == -beta,
            lax.full_like(g, 0),
            lax.select(x == beta, lax.full_like(g, 1), g))"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testMatchComplexLambda(self):
        string = """lambda g, ans, x: lax.select(
            x == -beta,
            lax.full_like(g, 0),
            lax.select(x == beta, lax.full_like(g, 1), g),)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testMatchComplexLambda21(self):
        string = """lambda config: math.sqrt(
        1.0
        / 3.0,
    ),"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testMatchComplexLambda22(self):
        string = """lambda config: math.sqrt(
        1.0
        / 3.0
    ),"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testMatchComplexLambda23(self):
        string = """lambda config: math.sqrt(
        1.0
        / 3.0
    )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testMatchComplexLambda2(self):
        string = """lambda x:a"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
