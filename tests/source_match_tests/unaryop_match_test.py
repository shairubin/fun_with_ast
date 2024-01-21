import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class UnaryOpMatcherTest(BaseTestUtils):

    def testUAddUnaryOp(self):
        node = create_node.UnaryOp(
            create_node.UAdd(),
            create_node.Name('a'))
        string = '+a'
        self._validate_match(node, string)
    def testUAddUnaryOp2(self):
        node = create_node.UnaryOp(
            create_node.UAdd(),
            create_node.Num('1'))
        string = '+1'
        self._validate_match(node, string)

    def testUAddUnaryOp3(self):
        node = create_node.UnaryOp(
            create_node.USub(),
            create_node.Num('1'))
        string = '-1'
        self._validate_match(node, string)

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

    def testNotUnaryOp2(self):
        node = create_node.UnaryOp(
            create_node.Not(),
            create_node.Name('a'))
        string = '(not a)'
        self._validate_match(node, string)

    def testNotUnaryOpIllegal(self):
        node = create_node.UnaryOp(
            create_node.Not(),
            create_node.Name('a'))
        string = 'not a:\n'
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)
        #self._validate_no_match(node, string)

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
        #matcher = GetDynamicMatcher(node)
        #source = matcher.do_match(string)
        #self.assertEqual(string, source)
        self._verify_match(node, string)
