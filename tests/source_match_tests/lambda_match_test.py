import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput, FailedToCreateNodeFromInput
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

    def testMatchNoArgs(self):
        string = """lambda : a"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testMatchNoArgs2(self):
        string = """lambda :a()"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testMatchMultiArgs(self):
        string = """lambda x,y, z  :   a(x,y)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testMatchArgsAnnotation(self):
        string = """lambda x:int :   a(x)"""
        with pytest.raises(FailedToCreateNodeFromInput):
            node = GetNodeFromInput(string)

    def testMatchVarargs(self):
        string = """lambda *_ :   a(x)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

