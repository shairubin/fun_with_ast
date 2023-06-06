import unittest

import pytest

from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError

from fun_with_ast.manipulate_node import create_node
from tests.source_match_tests.base_test_utils import BaseTestUtils


class BinOpMatcherTest(BaseTestUtils):

    def testAddBinOp(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Add(),
            create_node.Name('b'))
        string = 'a + b'
        self._verify_match(node, string)

    def testAddBinOpNegativeTest(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Add(),
            create_node.Name('b'))
        string = 'b + a'
        self._validate_no_match(node, string)
#        matcher = GetDynamicMatcher(node)
#        with pytest.raises(BadlySpecifiedTemplateError):
#            matcher._match(string)

    def testSubBinOp(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Sub(),
            create_node.Num('1'))
        string = '\ta - 1  \t'
        self._verify_match(node, string)


    def testSubBinOpNegativeTest(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Sub(),
            create_node.Num('2'))
        string = '\t  a - 1'
        self._validate_no_match(node, string)


    def testMultBinOp(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Mult(),
            create_node.Name('b'))
        string = 'a * b'
        self._verify_match(node, string)

    def testMultBinOpWithWS(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Mult(),
            create_node.Name('b'))
        string = '\t a  *  \t b'
        self._verify_match(node, string)

    def testMultBinOpWithWSAndParans(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Mult(),
            create_node.Name('b'))
        string = '(\t a  *  \t b    )'
        self._verify_match(node, string)

    def testNoMatchMultBinOpWithWSAndParans(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Mult(),
            create_node.Name('b'))
        string = '(\t a  *  \t b    '
        self._validate_no_match(node, string)

    def testDivBinOp(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Div(),
            create_node.Name('b'))
        string = ' a    /        b '
        self._verify_match(node, string)

    def testFloorDivBinOp(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.FloorDiv(),
            create_node.Name('b'))
        string = '  \t a // \t b  \t'
        self._verify_match(node, string)

    def testFloorDivBinOpWithComment(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.FloorDiv(),
            create_node.Num('1'))
        string = '  \t a // \t 1  \t #comment'
        self._verify_match(node, string)

    def testModBinOp(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Mod(),
            create_node.Name('b'))
        string = 'a % b    '
        self._verify_match(node, string)

    def testModBinOpWithComment(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Mod(),
            create_node.Name('b'))
        string = 'a % b    #comment'
        self._verify_match(node, string)

    def testModBinOpWithCommentNoMatch(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Mod(),
            create_node.Name('b'))
        string = 'a % c    #comment'
        self._validate_no_match(node, string)

    def testPowBinOp(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.Pow(),
            create_node.Name('b'))
        string = 'a ** b'
        self._verify_match(node, string)

    def testLShiftBinOp(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.LShift(),
            create_node.Name('b'))
        string = 'a << b'
        self._verify_match(node, string)

    def testRShiftBinOp(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.RShift(),
            create_node.Name('b'))
        string = 'a >> b'
        self._verify_match(node, string)

    def testBitOrBinOp(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.BitOr(),
            create_node.Name('b'))
        string = 'a | b'
        self._verify_match(node, string)

    def testBitXorBinOp(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.BitXor(),
            create_node.Name('b'))
        string = 'a ^ b'
        self._verify_match(node, string)

    def testAndBinOp(self):
        node = create_node.BinOp(
            create_node.Name('a'),
            create_node.BitAnd(),
            create_node.Name('b'))
        string = 'a & b'
        self._verify_match(node, string)

    def _validate_no_match(self, node, string):
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError) as e:
            matcher._match(string)
