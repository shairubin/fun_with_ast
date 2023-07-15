import unittest

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class FunctionDefMatcherTest(BaseTestUtils):

    def testEmpty(self):
        node = create_node.FunctionDef('test_fun', body=[create_node.Pass()])
        string = 'def test_fun():\n\t\t\t\tpass\n'
        self._verify_match(node, string)

    def testDefWithReturn(self):
        node = create_node.FunctionDef('test_fun', body=[create_node.Return(0)])
        string = 'def test_fun():\n\t\t\t\treturn 0'
        self._verify_match(node, string)

    def testDefWithReturn2(self):
        node = create_node.FunctionDef('test_fun', body=[create_node.Return(0)])
        string = 'def test_fun():\n\t\t\t\treturn 0\n'
        self._verify_match(node, string)


    def testEmptyWithDocString(self):
        node = create_node.FunctionDef('test_fun', body=[create_node.Pass()])
        string = "def test_fun():\n\t\t\t'''docstring'''\n\t\t\tpass\n"
        self._verify_match(node, string)
    def testSingleArg(self):
        node = create_node.FunctionDef('test_fun', create_node.arguments(args=['a']), body=[create_node.Pass()])
        string = 'def test_fun(a):\n  pass\n'
        self._verify_match(node, string)

    def testMultipleArgs(self):
        node = create_node.FunctionDef('test_fun', create_node.arguments(args=['a', 'b']), body=[create_node.Pass()])
        string = 'def test_fun(a, b):\n  pass\n'
        self._verify_match(node, string)

    def testDefaultBool(self):
        node = create_node.FunctionDef(
            'MatchCommentEOL', create_node.arguments(args=['self', 'string', 'remove_comment'], defaults=[False]),
            body=[create_node.Pass()])
        string = """def MatchCommentEOL(self, string, remove_comment=False):
    pass
"""
        self._verify_match(node, string)

    def testDefaultName(self):
        #        node = create_node.FunctionDef('test_fun', keys=('a'), values=('b'))
        node = create_node.FunctionDef(
            'test_fun', create_node.arguments(args=['a'], defaults=['b']),
            body=[create_node.Pass()])

        string = "def test_fun(a=b):\npass\n"
        self._verify_match(node, string)

    def testDefaultConstant(self):
        #        node = create_node.FunctionDef('test_fun', keys=('a'), values=('b'))
        node = create_node.FunctionDef(
            'test_fun', create_node.arguments(args=['a'], defaults=[3]),
            body=[create_node.Pass()])

        string = "def test_fun(a=3):\npass\n"
        self._verify_match(node, string)

    def testDefaults(self):
        node = create_node.FunctionDef(
            'test_fun', create_node.arguments(args=['e', 'f', 'a', 'c'], defaults=['b', 'd']),
            body=[create_node.Pass()])

        string = 'def test_fun(e, f, a =b, c= d):\n  pass\n'
        self._verify_match(node, string)

    def testArgsDefaultsVarargs(self):
        node = create_node.FunctionDef(
            'test_fun', create_node.arguments(args=['e', 'f', 'a', 'c'], defaults=['b', 'd'], vararg='d'),
            body=[create_node.Pass()])

        string = 'def test_fun(e, f, a=b, c=d, *d):\n  pass\n'
        self._verify_match(node, string)

    def testArgsDefaultsVarargsKwargs(self):
        node = create_node.FunctionDef(
            'test_fun', create_node.arguments(args=['e', 'f', 'a', 'c'], defaults=['b', 'd'], vararg='d', kwarg='a'),
            body=[create_node.Pass()])
        string = 'def test_fun(e, f, a=b, c=d, *d, **a):\n  pass\n'
        self._verify_match(node, string)

    def testDecoratorList(self):
        node = create_node.FunctionDef(
            'test_fun',
            decorator_list=[create_node.Name('dec'),
                            create_node.Call('call_dec')],
            body=[create_node.Pass()])
        string = '@dec\n@call_dec()\ndef test_fun():\n  pass\n'
        self._verify_match(node, string)

    def testCommentInDecoratorList(self):
        node = create_node.FunctionDef(
            'test_fun',
            decorator_list=[create_node.Name('dec'),
                            create_node.Call('call_dec')],
            body=[create_node.Pass()])
        string = '@dec\n#hello world\n@call_dec()\ndef test_fun():\n  pass\n'
        self._verify_match(node, string)

    def testCommentAfterDecorator(self):
        node = create_node.FunctionDef(
            'test_fun',
            decorator_list=[create_node.Name('dec')],
            body=[create_node.Pass()])
        string = '@dec\n #comment\ndef test_fun():\n  pass\n'
        self._verify_match(node, string)

    def testBody(self):
        node = create_node.FunctionDef(
            'test_fun',
            body=(create_node.Expr(create_node.Name('a')),))
        string = 'def test_fun():\n  a\n'
        self._verify_match(node, string)

    def testBodyFromString(self):
        string = 'def test_fun():\n  a\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testBodyFromString2(self):
        string = 'def test_fun():\n  return 0'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBodyFromString3(self):
        string = 'def test_fun():\n  return a or b'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBodyFromString4(self):
        string = 'def test_fun():\n  return a or b\ndef test_fun2():\n  return d or c'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testBodyFromString5(self):
        string = 'def test_fun():\n  return a or b\n\n\n#\#       \ndef test_fun2():\n  return d or c'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBodyFromString7(self):
        string = 'def test_fun():\n  return isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)\ndef test_fun2():\n  return d or c'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    @pytest.mark.skip('this is issue #17')
    def testBodyFromString8(self):
        string = 'def test_fun():\n  return isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)\n#\ndef test_fun2():\n  return d or c'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
