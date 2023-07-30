import unittest

import pytest

from fun_with_ast.manipulate_node import create_node as create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ClassMatcherTest(BaseTestUtils):

    def testClassSimple(self):
        string = "class FunWithAST:\n   pass"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testClassSimple2(self):
        string = "class FunWithAST:\n   def __init__(self):\n       pass"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testClassInheritance2(self):
        string = "class FunWithAST(ast):\n  def __init__(self):\n   pass\n  def forward(self, x):\n   return self.main(x)"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
