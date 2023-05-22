import unittest

import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.dynamic_matcher import GetDynamicMatcher


class UnaryOpMatcherTest(unittest.TestCase):

    def testUAddUnaryOp(self):
        node = create_node.UnaryOp(
            create_node.UAdd(),
            create_node.Name('a'))
        string = '+a'
        self._validate_match(node, string)

    def _validate_match(self, node, string):
        matcher = GetDynamicMatcher(node)
        source = matcher.Match(string)
        self.assertEqual(string, source)

    def testUSubUnaryOp(self):
        node = create_node.UnaryOp(
            create_node.USub(),
            create_node.Name('a'))
        string = '-a'
        self._validate_match(node, string)

    def testUSubUnaryOWithParans(self):
        node = create_node.UnaryOp(
            create_node.USub(),
            create_node.Name('a'))
        string = '-(a)'
        self._validate_match(node, string)

    @pytest.mark.xfail(reason='not implemented yet')
    def testUSubUnaryOWithExternalParans(self):
        node = create_node.UnaryOp(
            create_node.USub(),
            create_node.Name('a'))
        string = '(-(a))'
        self._validate_match(node, string)

    def testNotUnaryOp(self):
        node = create_node.UnaryOp(
            create_node.Not(),
            create_node.Name('a'))
        string = 'not a'
        self._validate_match(node, string)

    def testInvertUnaryOp(self):
        node = create_node.UnaryOp(
            create_node.Invert(),
            create_node.Name('a'))
        string = '~a'
        self._validate_match(node, string)

    def testInvertUnaryOpWithParans(self):
        node = create_node.UnaryOp(
            create_node.Invert(),
            create_node.Name('a'))
        string = '~(a) #comment'
        self._validate_match(node, string)

    def testInvertUnaryOpWithWS(self):
        node = create_node.UnaryOp(
            create_node.Invert(),
            create_node.Name('a'))
        string = '~a    \t '
        self._validate_match(node, string)

    def testInvertUnaryOpWithWSAndComment(self):
        node = create_node.UnaryOp(
            create_node.Invert(),
            create_node.Name('a'))
        string = '~(a)    \t #comment'
        self._validate_match(node, string)

    def _validate_match(self, node, string):
        matcher = GetDynamicMatcher(node)
        source = matcher.Match(string)
        self.assertEqual(string, source)
