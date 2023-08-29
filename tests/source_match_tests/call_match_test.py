import unittest

import pytest

from fun_with_ast.manipulate_node import create_node as create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from tests.source_match_tests.base_test_utils import BaseTestUtils


class CallMatcherTest(BaseTestUtils):

    def testBasicMatch(self):
        node = create_node.Call('a')
        string = 'a()'
        self._verify_match(node, string)
    def testBasicMatchWarp(self):
        node = create_node.Call('a')
        string = '(a())'
        self._verify_match(node, string)
    def testBasicMatchWS(self):
        node = create_node.Call('a')
        string = ' a()'
        self._verify_match(node, string)

    def testBasicMatchWS2(self):
        node = create_node.Call('a.b')
        string = ' a.b()'
        self._verify_match(node, string)
    def testMatchStarargs(self):
        node = create_node.Call('a', args=[create_node.Starred('args')])
        string = 'a(*args)'
        self._verify_match(node, string)
    def testMatchStarargs2(self):
        node = create_node.Call('a', args=[create_node.Name('b'), create_node.Starred('args')])
        string = 'a(b, *args)'
        self._verify_match(node, string)

    def testNoMatchStarargs(self):
        node = create_node.Call('a', args=[create_node.Name('b'), create_node.Starred('arrrggs')])
        string = 'a(b, *args)'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)
    def testNoMatchStarargs2(self):
        node = create_node.Call('a', args=[create_node.Name('c'), create_node.Starred('args')])
        string = 'a(b, *args)'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testMatchWithStarargsBeforeKeyword(self):
        node = create_node.Call('a', args=[create_node.Name('d')], keywords=[create_node.keyword('b', 'c')])
        string = 'a(d \t , \t b= c)'
        self._verify_match(node, string)
    def testMatchWithStarargsBeforeKeyword2(self):
        node = create_node.Call('a', args=[create_node.Stared('fun-with-ast')],
                                keywords=[create_node.keyword('b', 'c'), create_node.keyword('e', 'f')])
        string = 'a(*fun-with-ast, b=c, e = f)'
        self._verify_match(node, string)

    def testMatchWithStarargsBeforeKeyword3(self):
        node = create_node.Call('a', args=[create_node.Name('d'), create_node.Stared('starred')],
                                keywords=[create_node.keyword('b', 'c'), create_node.keyword('e', 'f')])
        string = 'a(d,   *starred, b=c, e = f )'
        self._verify_match(node, string)


    def testMatchKeywordOnly(self):
        node = create_node.Call('a', keywords=[create_node.keyword('b', 'c')])
        string = 'a(b=c)'
        self._verify_match(node, string)

    def testCallWithAttribute(self):
        node = create_node.Call('a.b')
        string = 'a.b()'
        self._verify_match(node, string)

    def testCallWithAttributeAndParam(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "'")])
        string = 'a.b(\'fun-with-ast\')'
        self._verify_match(node, string)

    def testCallWithAttributeAndNone(self):
        node = create_node.Call('a.b', args=[create_node.CreateNone('None')])
        string = 'a.b(None)'
        self._verify_match(node, string)

    def testCallWithAttributeAndNoneNoMatch(self):
        node = create_node.Call('a.b', args=[create_node.CreateNone('None')])
        string = 'a.b(none)'
        with pytest.raises(BadlySpecifiedTemplateError):
            self._verify_match(node, string)

    def testCallWithAttributeAndParamAndQuate(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "\"")])
        string = "a.b(\"fun-with-ast\")"
        self._verify_match(node, string)

    def testNoMatchCallWithAttributeAndParamAndQuate(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "'")])
        string = "a.b(\"fun-with-ast\")"
        self._verify_match(node, string)
    def testNoMatchCallWithAttributeAndParamAndQuate2(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "\"")]) #TODO: do not need second param
        string = "a.b('fun-with-ast')"
        self._verify_match(node, string)

    def testCallWithAttributeAndParam2(self):
        node = create_node.Call('a.b', args=[create_node.Num('1')])
        string = 'a.b(1)'
        self._verify_match(node, string)
    def testCallWithAttributeAndParam4(self):
        node = create_node.Call('a.b', args=[create_node.Num('1'), create_node.Num('2')])
        string = 'a.b(1,2)'
        self._verify_match(node, string)

    def testCallWithAttributeAndParam5(self):
        node = create_node.Call('a', args=[create_node.Num('1'), create_node.Num('2')])
        string = 'a( 1,2)'
        self._verify_match(node, string)

    def testCallWithAttributeAndParam6(self):
        node = create_node.Call('a', args=[create_node.Num('1'), create_node.Num('2')])
        string = 'a(1,2)'
        self._verify_match(node, string)

    def testCallWithAttributeAndParam7(self):
        node = create_node.Call('a', args=[create_node.Num('1')])
        string = 'a(1)'
        self._verify_match(node, string)
    def testCallWithAttributeAndParam3(self):
        node = create_node.Call('a.b', args=[create_node.Num('1')])
        string = '(a.b(1))'
        self._verify_match(node, string)

    def testCallWithMultiLines(self):
        string = "fileparse.parse_csv(lines,\n \
                                     select=['name', 'shares', 'price'],\n \
                                     types=[str, int, float],\n \
                                    **opts)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallWithMultiLinesSimple(self):
        string = "fileparse.parse_csv(lines,\n \
                                      a)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithMultiLinesSimple2(self):
        string = "fileparse.parse_csv(lines)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithMultiLinesSimple3(self):
        string = "a.b(c,\n \
                       d=[e])\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithMultiLinesSimple4(self):
        string = "a.b(d=[])\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithMultiLinesSimple5(self):
        string = "a(d=c)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallMatchWithKwargs(self):
        string = "a(**args)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallMatchWithKwargs2(self):
        string = "a(**args, **args2)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallMatchWithKwargs3(self):
        string = "a(*stared, **args2)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    @pytest.mark.skip('not implemented yet')
    def testCallMatchWithKwargs4(self):
        string = "a(b=c, *d)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallMatchWithKwargs4_5(self):
        string = "a(b=c, *d, e)\n"
        with pytest.raises(SyntaxError):
            node = GetNodeFromInput(string)

    @pytest.mark.skip('not implemented yet')
    def testCallMatchWithKwargs4_5(self):
        string = "a(b=c, *d, e=f)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallMatchWithKwargs5(self):
        string = "a(*d, b=c)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallMatchWithKwargs6(self):
        string = "a(*d, *c)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallMatchWithKwargs7(self):
        string = "a(c=d, e)\n"
        with pytest.raises(SyntaxError):
            node = GetNodeFromInput(string)

    def testCallDoubeAttribute(self):
        string = "torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-06, affine=True)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithJoinedString(self):
        string = "logger.info(f'vqgan is loaded from: {model_path} [params_ema]')\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallWithJoinedString2(self):
        string = "logger.info(f\"vqgan is loaded from: {model_path} [params_ema]\")\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallWithJoinedString3(self):
        string = "c(f'a')\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallWithFloat(self):
        string = "c(1.0)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallWithFloat2(self):
        string = "c(1.0,)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testNoMatchCallWithIntAndComma(self):
        string = "c(1),\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testCallDoubeAttribute2(self):
        string = "torch.nn.GroupNorm(eps=1e-06)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallDoubeAttribute3(self):
        string = "torch.nn.GroupNorm(n=32, a=1, eps=1e-06)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallDoubeAttribute4(self):
        string = "torch.nn.GroupNorm(eps=1e-06, a=True)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithAttributeAndParamWS(self):
        node = create_node.Call('a.b', args=[create_node.Constant('fun-with-ast', "'")])
        string = 'a.b(\'fun-with-ast\')'
        self._verify_match(node, string)
    def testCallDoubeAttributeWithParams(self):
        string = "a.b(x)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallDoubeAttributeWithParams2(self):
        string = "a().b(x)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallDoubeAttributeWithParams3(self):
        string = "a(z, \ty).b(x=3)    \n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallDoubeAttributeWithParams31(self):
        string = "b(x=3)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallDoubeAttributeWithParams4(self):
        string = "super().__init__()\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallDoubeAttributeWithParams5(self):
        string = """ACT2FN.update({"smelu": smelu()})\n"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallMultipleQuotes(self):
        string = "A(\"a\", 'b')\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallCommaAtTheEnd(self):
        string = """return self._normalize(
                self,
                x,
                rms_sq,
                reduction_axes,
                feature_axes,
                self.dtype,
                self.param_dtype,
                self.epsilon,
                self.use_scale,
                self.scale_init
            )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallCommaAtTheEnd2(self):
        string = """return self._normalize(
                self,
                x,
                rms_sq,
                reduction_axes,
                feature_axes,
                self.dtype,
                self.param_dtype,
                self.epsilon,
                self.use_scale,
                self.scale_init,
            )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallStringParam(self):
        string = """return self._normalize(
                self,
                "x",
                rms_sq,
                reduction_axe
            )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallCallParam(self):
        string = """return self._normalize(
                self,
                "x",
                C(1.0),
                reduction_axe
            )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallCallParam2(self):
        string = """return self._normalize(
                self,
                "x",
                C(1.0),
            ),"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallCommaAtTheEnd3(self):
        string = """self._normalize(self,)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithNewLineAndComment(self):
        string = """T(a='cpu')\n# X\n"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testCallWithNewLineAndComment3(self):
        string = """T(a)\n # X\n   # Y\n"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    @pytest.mark.skip('not implemented yes - last comment in modul must end with \n ')
    def testCallWithNewLineAndComment2(self):
        string = """T(a,b=x)\n     # X ,  \n   # Y"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallMultipleQuotes2(self):
        string = "A('a', \"b\")\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallTwoComments(self):
        string = """b(a)\n  #if p\n # m\n"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallTwoComments2(self):
        string = """
l(a)        
        #      load(x,y)\n
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallThreeComments(self):
        string =  """
l(a)        
        #      load(x,y)\n
        #      load(z,w)\n
        #      load(i,o)\n
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    @pytest.mark.skip('not implemented yet - unbalanced parentheses in comments')
    def testCallThreeComments2(self):
        string = """
l(a)        
        #      load(x,y)\n
        #      load(z,w))\n
        #      load(i,o)\n
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallTwoComments3(self):
        string =  """
def __init__():
     chkpt = torch.load(model_path, map_location='cpu')
         #if 'params_d' in chkpt:
         #    self.load_state_dict(torch.load(model_path, map_location='cpu')['params_d'])\n
"""

        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    #@pytest.mark.skip('issue #95')
    def testCallWithLists(self):
        string =  """portdicts = fileparse.parse_csv(lines,
                                        select=['name', 'shares', 'price'], # <-- See this line of AST unparse results
                                        types=[str, int, float],
                                        **opts)
                                        # <-- See this line is missing in of AST unparse results
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallWithLists1(self):
        string =  """fileparse.parse_csv(lines,
                                        select=['name', 'shares', 'c'],# <-- See this line of AST unparse results
                                        types=[str, int, float])
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallWithLists2(self):
        string =  """fileparse.parse_csv(lines,
                                        select=['name'],
                                        types=[str])
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallWithLists21(self):
        string = """fileparse.parse_csv(lines,
                                        select=['name'], # <-- See this line of AST unparse results
                                        types=[str])
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallWithLists3(self):
        string =  """fileparse.parse_csv(lines,
                                        select=['name'], # <-- See this line of AST unparse results
                                        )
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
