import unittest

import pytest
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from tests.source_match_tests.base_test_utils import BaseTestUtils


class RaiseMatcherTest(BaseTestUtils):

    def testSimpleRaise(self):
        string = 'raise x'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSimpleRaise3(self):
        string = 'raise'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSimpleRaise2(self):
        string = 'raise x from y'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSimpleRaiseComment(self):
        string = 'raise \t x \t from \t y # coment'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSimpleRaiseNoMatch(self):
        string = 'raise x from '
        node = GetNodeFromInput(string+'y')
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testSimpleRaiseWithString(self):
        string = "raise \t x('string')"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSimpleRaiseWithString2(self):
        string = "raise \t x(f\"Unknown norm type {type}\")"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testSimpleRaiseWithString3(self):
        string = """raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSimpleRaiseWithString31(self):
        string = """raise ValueError(
                f"X"
                f"Y"
            )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSimpleRaiseWithString32(self):
        string = """raise ValueError(
                f"X"
            )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
