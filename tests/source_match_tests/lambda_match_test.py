import unittest

from fun_with_ast.manipulate_node import create_node
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
