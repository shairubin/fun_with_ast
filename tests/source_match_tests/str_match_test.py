import unittest

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.manipulate_node import create_node
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ConstantStrMatcherTest(BaseTestUtils):



    def testBasicMatchStr(self):
        node = create_node.Constant('1', "'")
        string = "'1'"
        self._verify_match(node, string)


    def testBasicMatchStrDoubelQ(self):
        node = create_node.Constant("1", "\"")
        string = "\"1\""
        self._verify_match(node, string)


    def testBasicMatchEmpty(self):
        node = create_node.Constant('', "'")
        string = "''"
        self._verify_match(node, string)

    def testBasicMatchEmptyDoubleQ(self):
        node = create_node.Constant('', "\"")
        string = "\"\""
        self._verify_match(node, string)





    def testBasicMatchConcatinatedString(self):
        node = create_node.Constant('1''2', "'")
        string = "'12'"
        self._verify_match(node, string)

    def testBasicMatchStrWithWS(self):
        node = create_node.Constant('  1  ', "'")
        string = "'  1  '"
        self._verify_match(node, string)

    def testBasicMatchStrWithNL(self):
        node = create_node.Constant('  1  ', "'")
        string = "'  1  '"
        self._verify_match(node, string)

    def testBasicNoMatchStr(self):
        node = create_node.Constant('1', "'")
        string = "'2'"
        matcher = GetDynamicMatcher(node)
        with pytest.raises(BadlySpecifiedTemplateError):
            matcher.do_match(string)

    def testStringWithNLString(self):
        string = "\"\\n\""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testStringWithNLString2(self):
        string = "'\\n'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithNLString3(self):
        string = "'abc\\n'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithNLString4(self):
        string = "'abc\ndef'"
        with pytest.raises(SyntaxError):
            node = GetNodeFromInput(string)
    def testStringWithNLString5(self):
        string = "'abc\\ndef'"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
