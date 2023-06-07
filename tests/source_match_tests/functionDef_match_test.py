import unittest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class FunctionDefMatcherTest(unittest.TestCase):

    def testEmpty(self):
        node = create_node.FunctionDef('test_fun', body=[create_node.Pass()])
        string = 'def test_fun():\n\t\t\t\tpass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testEmptyWithDocString(self):
        node = create_node.FunctionDef('test_fun', body=[create_node.Pass()])
        string = "def test_fun():\n\t\t\t'''docstring'''\n\t\t\tpass\n"
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testSingleArg(self):
        node = create_node.FunctionDef('test_fun', create_node.arguments(args=['a']), body=[create_node.Pass()])
        string = 'def test_fun(a):\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testMultipleArgs(self):
        node = create_node.FunctionDef('test_fun', create_node.arguments(args=['a', 'b']), body=[create_node.Pass()])
        string = 'def test_fun(a, b):\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testDefaultBool(self):
        node = create_node.FunctionDef(
            'MatchCommentEOL', create_node.arguments(args=['self', 'string', 'remove_comment'], defaults=[False]),
            body=[create_node.Pass()])
        string = """def MatchCommentEOL(self, string, remove_comment=False):
    pass
"""
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        matched_source = matcher.GetSource()
        self.assertEqual(string, matched_source)

    def testDefaultName(self):
        #        node = create_node.FunctionDef('test_fun', keys=('a'), values=('b'))
        node = create_node.FunctionDef(
            'test_fun', create_node.arguments(args=['a'], defaults=['b']),
            body=[create_node.Pass()])

        string = "def test_fun(a=b):\npass\n"
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        matched_source = matcher.GetSource()
        self.assertEqual(string, matched_source)

    def testDefaultConstant(self):
        #        node = create_node.FunctionDef('test_fun', keys=('a'), values=('b'))
        node = create_node.FunctionDef(
            'test_fun', create_node.arguments(args=['a'], defaults=[3]),
            body=[create_node.Pass()])

        string = "def test_fun(a=3):\npass\n"
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        matched_source = matcher.GetSource()
        self.assertEqual(string, matched_source)

    def testDefaults(self):
        node = create_node.FunctionDef(
            'test_fun', create_node.arguments(args=['e', 'f', 'a', 'c'], defaults=['b', 'd']),
            body=[create_node.Pass()])

        string = 'def test_fun(e, f, a =b, c= d):\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testArgsDefaultsVarargs(self):
        # node = create_node.FunctionDef(
        #     'test_fun', arcg=('e', 'f'), keys=('a', 'c'), values=('b', 'd'),
        #     vararg_name='args')
        node = create_node.FunctionDef(
            'test_fun', create_node.arguments(args=['e', 'f', 'a', 'c'], defaults=['b', 'd'], vararg='d'),
            body=[create_node.Pass()])

        string = 'def test_fun(e, f, a=b, c=d, *d):\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testArgsDefaultsVarargsKwargs(self):
        # node = create_node.FunctionDef(
        #     'test_fun', args=('e', 'f'), keys=('a', 'c'), values=('b', 'd'),
        #     vararg_name='args', kwarg_name='kwargs')
        node = create_node.FunctionDef(
            'test_fun', create_node.arguments(args=['e', 'f', 'a', 'c'], defaults=['b', 'd'], vararg='d', kwarg='a'),
            body=[create_node.Pass()])

        string = 'def test_fun(e, f, a=b, c=d, *d, **a):\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testDecoratorList(self):
        node = create_node.FunctionDef(
            'test_fun',
            decorator_list=[create_node.Name('dec'),
                            create_node.Call('call_dec')],
            body=[create_node.Pass()])
        string = '@dec\n@call_dec()\ndef test_fun():\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testCommentInDecoratorList(self):
        node = create_node.FunctionDef(
            'test_fun',
            decorator_list=[create_node.Name('dec'),
                            create_node.Call('call_dec')],
            body=[create_node.Pass()])
        string = '@dec\n#hello world\n@call_dec()\ndef test_fun():\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())

    def testCommentAfterDecorator(self):
        node = create_node.FunctionDef(
            'test_fun',
            decorator_list=[create_node.Name('dec')],
            body=[create_node.Pass()])
        string = '@dec\n #comment\ndef test_fun():\n  pass\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        matcher_result = matcher.GetSource()
        self.assertEqual(string, matcher_result)

    def testBody(self):
        node = create_node.FunctionDef(
            'test_fun',
            body=(create_node.Expr(create_node.Name('a')),))
        string = 'def test_fun():\n  a\n'
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        self.assertEqual(string, matcher.GetSource())
