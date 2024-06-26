import pytest

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput, FailedToCreateNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class FunctionDefMatcherTest(BaseTestUtils):

    def testEmpty(self):
        node = create_node.FunctionDef('test_fun', body=[create_node.Pass()])
        string = 'def test_fun():\n\t\t\t\tpass\n'
        self._verify_match(node, string)
        async_string = 'async \t' + string
        node = GetNodeFromInput(async_string)
        self._verify_match(node, async_string)
    def testDefWithReturn(self):
        node = create_node.FunctionDef('test_fun', body=[create_node.Return(0)])
        string = 'def test_fun():\n\t\t\t\treturn 0'
        self._verify_match(node, string)

    def testDefWithReturn2(self):
        node = create_node.FunctionDef('test_fun', body=[create_node.Return(0)])
        string = 'def test_fun():\n\t\t\t\treturn 0\n'
        self._verify_match(node, string)

    def testCallWitJoinedStr6(self):
        string = """
        
@nn.compact
def __call__():
        layer(name=f"A_{i}",)(a,c,)
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallWitJoinedStr61(self):
        string ="""@nn.compact
def a():
   pass"""
        node = GetNodeFromInput(string)
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
        string = 'def test_fun():\n  return a or b\n\n\n##       \ndef test_fun2():\n  return d or c'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBodyFromString7(self):
        string = 'def test_fun():\n  return isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)\ndef test_fun2():\n  return d or c'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBodyFromString8(self):
        string = 'def test_fun():\n  return isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)\n#\ndef test_fun2():\n  return d or c'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBodyFromString9(self):
        string = 'def test_fun():\n  (b)\n#\ndef test_fun2():\n  pass'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testDefInit(self):
        string = 'def __init__(self):\n     pass'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testDefInit2(self):
        string = 'def __init__(self):\n     a.b()\n'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testInnerDef(self):
        string = """def convert_exception_to_response(get_response):

            if iscoroutinefunction(get_response):

                @wraps(get_response)
                async def inner(request):
                    try:
                        response = await get_response(request)
                    except Exception as exc:
                        response = await sync_to_async(
                            response_for_exception, thread_sensitive=False
                        )(request, exc)
                    return response

                return inner"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    def testArgsAndAnnotation(self):
        string = 'def test_fun(a: list = []):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testArgsAndAnnotation2(self):
        string = 'def test_fun(a: int):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testArgsAndAnnotation2_1(self):
        string = 'def test_fun(a,b,c):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testArgsAndAnnotation2_2(self):
        string = 'def test_fun(a,b,c,):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testArgsAndAnnotation2_3(self):
        string = 'def test_fun(a,):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testArgsAndAnnotation2_5(self):
        string = 'def test_fun(a: int=1):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testArgsAndAnnotation2_6(self):
        string = 'def test_fun(a: int=2):\n  pass\n'
        node = GetNodeFromInput(string)
        assert  node.args.args[0].annotation, 'must be not none'
        assert  node.args.args[0].arg == 'a'
        self._verify_match(node, string)

    def testArgsAndAnnotation3(self):
        string = "def test_fun(a: int, b  :      str='fun with ast'):\n  pass\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionLevelAnnotation(self):
        string = "def test_fun() -> int:\n  pass\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionLevelAnnotation2(self):
        string = "def test_fun()     \t->     my_class\t:\n  pass\n"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFunctionLevelAnnotation3(self):
        string = "def test_fun()     \t->     my_class\t: #comment\n  pass\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testFunctionLevelAnnotation4(self):
        string = """def _smelu(x: Any) -> Any:
    x = jnp.where(x <= -beta, 0.0, x)
    return jnp.where(x >= beta, x, jnp.square(x + beta) / (4 * beta))"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testFunctionLevelAnnotation5(self):
        string = """def argumenttype_type(
        t: Type, *, mutable: bool, binds: ArgName, symint: bool
        ) -> NamedCType:
        pass
        """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionLevelAnnotation5_3(self):
        string = """def argumenttype_type(
        t: Type, *, mutable: bool
        ) -> NamedCType:
        pass
        """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionLevelAnnotation5_3_1(self):
        string = """def a(\n\tt: Type, *, mutable: bool\n\t) -> NamedCType:\n\t\tpass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testFunctionLevelAnnotation5_3_2(self):
        string = """def a(\n\tt, *, mutable) -> NamedCType:\n\t\tpass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionLevelAnnotation5_3_3(self):
        string = """def a(\n\tt, *, mutable):\n\t\tpass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testFunctionLevelAnnotation5_3_4(self):
        string = """def a(t, *, mutable):\n\t\tpass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionLevelAnnotation5_2(self):
        string = """def argumenttype_type(
        t: Type
        ) -> NamedCType:
        pass
        """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionLevelAnnotation5_1(self):
        string = """def a(
        ) -> NamedCType:
        pass
        """
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionMultiLine(self):
        string = """def a(
        self,
        ):
           pass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionMultiLine2(self):
        string = """def a(
        ):
            pass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testFunctionMultiLine3(self):
        string = """def a(
        self,
        b,
        ):
            pass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testFunctionMultiLine31(self):
        string = """def a(
        self,

        ):
            pass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionMultiLine32(self):
        string = """def a(self,

        ):
            pass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionMultiLine33(self):
        string = """def a(
        self,

        b, 
        
        ):
            pass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionMultiLine4(self):
        string = """def a(
        self,
        b
        ):
             pass"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testFunctionDefWithSubscript(self):
        string =  """
def _generate_continue(self, sequences, model, tokenizer):
    generated_sequences[i * self.create_n + ii].replace(" ", "").replace("\\n", "")
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFunctionDefWithSubscript1_1(self):
        string = """
def _generate_continue(self, sequences, model, tokenizer):
    generated_sequences[i * self.create_n + ii].replace(" ", "").replace("\n", "")
"""
        with pytest.raises(FailedToCreateNodeFromInput):
            node = GetNodeFromInput(string, get_module=True)

    def testFunctionDefWithSubscript2(self):
        string =  """
def _generate_continue(self, sequences, model, tokenizer):
    generated_sequences.replace(" ", "").replace("\\n", "")
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testStarArgs(self):
        string = 'def test_fun(*,a):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStarArgs2(self):
        string = 'def test_fun(*b,a):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStarArgs2_2(self):
        string = 'def test_fun(*b,a:int, c:bool):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStarArgs3(self):
        string = 'def test_fun(*b,a, c):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStarArgs4(self):
        string = 'def test_fun(*,a, c, **d):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testStarArgs5(self):
        string = 'def impl(qualname, *, device_types=("cpu", "cuda"), func=None):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStarArgs5_1(self):
        string = 'def impl(qualname, *, device_types="cpu", func=None):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStarArgs5_2(self):
        string = 'def impl(*, device_types=("cpu", "cuda")):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testStarArgs5_3(self):
        string = 'def impl(device_types=("cpu", "cuda")):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStarArgs5_3_1(self):
        string = 'def impl(device_types=("cpu", "cuda"), ):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStarArgs5_4(self):
        string = 'def impl(d=("D", "c") ,x=7):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStarArgs5_5(self):
        string = 'def impl(kw1=("D",a), kw2=7):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStarArgs5_6(self):
        string = 'def impl(kw1=[(1, kw_name),r]):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStarArgs5_7(self):
        string = 'def impl(kw1=[(1, kw_name),r],):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStarArgs6(self):
        string = 'def impl(qualname, *,  func=None):\n  pass\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testListAndTuple(self):
        string = """isinstance(
    operands, (list, tuple)
    )
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNLWithWhiteSpaces(self):
        string = """def test_fun():\n  table = Table(show_header=True, header_style="bold magenta")\n \n  pass
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNLWithWhiteSpaces01(self):
        string = """def test_fun():\n  Table(a)\n \n  pass
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNLWithWhiteSpaces1(self):
        string = """def test_fun():\n  table = Table(show_header=True, header_style="bold magenta")\n\n \n  pass
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNLWithWhiteSpaces1_1(self):
        string = """def test_fun():\n  table = Table(show_header=True, header_style="bold magenta")\n\n \n     \n  pass
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNLWithWhiteSpaces1_2(self):
        string = """def test_fun():\n  table = Table(show_header=True, header_style="bold magenta")\n \n \n     \n  pass
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNLWithWhiteSpaces2(self):
        string = """def test_fun():\n  table = Table(show_header=True, header_style="bold magenta")\n \n  pass"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testNLWithAnyVarArg(self):
        string = """def test_fun(a, *_):\n  pass"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
