import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from tests.source_match_tests.base_test_utils import BaseTestUtils


class StartedatcherTest(BaseTestUtils):
    def testSimpeStarredd(self):
        node = create_node.Stared('a')
        string = '*a'
        self._verify_match(node, string)

    def testSimpeStarred2(self):
        node = create_node.Stared('a')
        string = '(*a)'
        self._verify_match(node, string)

    def testSimpeStarredComment(self):
        node = create_node.Stared('a')
        string = '(*a) \t # \t comment'
        self._verify_match(node, string)

    def testSimpeStarredComplex(self):
        node = create_node.Stared('fun-with-ast')
        string = '(*fun-with-ast) \t # \t comment'
        self._verify_match(node, string)
    def testNoMatchStarred(self):
        node = create_node.Stared('fun-with-ast')
        string = '(**fun-with-ast) \t # \t comment'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testNoMatchStarred2(self):
        node = create_node.Stared('fun-with-ast')
        string = '(* *fun-with-ast) \t # \t comment'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testNoMatchStarred3(self):
        node = create_node.Stared('fun-with-ast')
        string = '(* fun-with-ast) \t # \t comment'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

