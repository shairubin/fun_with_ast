import unittest

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.manipulate_node import create_node
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ConstantStrMatcherTest(BaseTestUtils):





    def testBasicMatchMultiPart(self):
        node = create_node.Constant("'1''2'", "\"")
        string = "\"'1''2'\""
        self._verify_match(node, string)

    def testBasicNoMatchMultiPart(self):
        node = create_node.Constant("'1''2'", "\"")
        string = "\"'1''3'\""
        with pytest.raises(BadlySpecifiedTemplateError) as e:
            self._verify_match(node, string)



    def testBasicMatchMultiPart2(self):
        node = create_node.Constant('1''2', "'")
        string = '\'1\'\'2\''
        self._verify_match(node, string)

    def testNoMatchMultiPart(self):
        node = create_node.Constant("\"'1''2'\"", "\"")
        string = "\"'1''3'\""
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError) as e:
            matcher.do_match(string)


    def testBasicMatchConcatinatedString(self):
        node = create_node.Constant('1''2', "'")
        string = "'12'"
        self._verify_match(node, string)

    def testMultiPartFromInput(self):
        string = "'X''Y'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testMultiPartFromInput2(self):
        string = "'X'  'Y'" # note that the spaces are NOR part of the string
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testMultiPartFromInput3(self):
        string = "'X'  'Y' # comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testMultiPartFromInput4(self):
        string = "'X '  'Y ' # comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testMultiPartFromInput5(self):
        string = "'X '  'Y ' # comment\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testMultiPartEmpty(self):
        string = "''"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testMultiPartEmpty2(self):
        string = "''''" # not supported in python 3.10
        with pytest.raises(SyntaxError):
            node = GetNodeFromInput(string)

