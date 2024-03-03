import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput, FailedToCreateNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class CallMatcherTest(BaseTestUtils):

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
    def testCallWithMultiLinesSimple5_1(self):
        string = "a(d)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithMultiLinesSimple5_2(self):
        string = "a(d,)\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithMultiLinesSimple5_3(self):
        string = "a(d,7)\n"
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
        with pytest.raises(FailedToCreateNodeFromInput):
            node = GetNodeFromInput(string)

    @pytest.mark.skip('not implemented yet')
    def testCallMatchWithKwargs4_51(self):
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
        with pytest.raises(FailedToCreateNodeFromInput):
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

    def testCallDoubeAttributeWithParams32(self):
        string = "b(1.0,) # comment\n"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallDoubeAttributeWithParams33(self):
        string = "b(x=3) # comment"
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

    def testCallOneComments(self):
        string = """b()\n  #if"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallSimplest(self): # SHOULD NOT MATCH
        string = """b()\n"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testCallSimplest2(self): # SHOULD NOT MATCH
        string = """b()"""
        node = GetNodeFromInput(string)
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

    def testCallWithLists22(self):
        string = """a.b(lines, # comment
                        select)"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallWithLists3(self):
        string =  """fileparse.parse_csv(lines,
                                        select=['name'], # <-- See this line of AST unparse results
                                        )
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallWithTuple(self):
        string = """lax(
            (0,1),
            (2, 3),
        )
        """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallWithTuple1_1(self):
        string = """lax(
            (0,1),
            (2, 3),
        )
        """
        node = GetNodeFromInput(string, get_module=False) # note the False
        self._verify_match(node, string)

    def testCallWithTuple1(self):
        string = """lax.dynamic_slice(
            0, 0,
            1, 1,
        )"""
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string)

    def testCallWithTuple1_2(self):
        string = """lax.dynamic_slice(
            0, 0,
            1, 1,
        )
        """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testCallWithTuple2(self):
        string = """a.b(0, 0, 1, 1,
        )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithTuple3(self):
        string = """a.b(0, 0, 1, 1,
        )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCall1_1(self):
        string = """a() #comment 
         """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testCall1_2(self):
        string = """a() #comment """
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string)

    def testCall2(self):
        string = """a.b()"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testCallWitJoinedStr61(self):
        string = """layer(name=f"A_{i}",)(a,c,)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWitJoinedStr62(self):
        string = """layer(f"A_{i}",)"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testCallWithNLString(self):
        string = """replace("\\n", "")"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCallWithNLString2(self):
        string = "replace(\"\\n\", \"\")"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testStringWithParenthesisAndChainedCall(self):
        string = "line.split('(')[1].strip(' \\t\\n\\r')"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithParenthesisAndChainedCall2(self):
        string = "strip(' \\n')"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithParenthesisAndChainedCall3(self):
        string = "strip(' \\n\\t')"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithParenthesisAndChainedCall3(self):
        string = "strip(' \\n\\r')"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithParenthesisAndChainedCall1(self):
        string = "line.split('(').strip('a')"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithParenthesis2(self):
        string = "line.split('(')"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithParenthesis3(self):
        string = "a('(')"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithParenthesis3_1(self):
        string = "a('(')# comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithParenthesis3_2(self):
        string = "a('(')# comm(ent"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithParenthesis3_3(self):
        string = "a('()(((')"
        node = GetNodeFromInput(string)
        #with pytest.raises(NotImplementedError):
        self._verify_match(node, string)

    def testStringWithParenthesis3_4(self): # TODO: this test pass but it is incorrect, see test 3_5
        string = "a(')')"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    @pytest.mark.skip(reason="not implemented yet see 3_4 above")
    def testStringWithParenthesis3_5(self):
        string = "a(')',)"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithParenthesis4(self):
        string = "a('()')"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testStringWithParenthesis5(self):
        string = "a('())')"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithParenthesis5_1(self):
        string = "a('())') # comment\n#comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testStringWithParenthesis5_2(self):
        string = "a('()') # comment)\n#comment"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testModule18Partial(self):
        string = "sysconfig.get_path('include')"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testModule18Partial2(self):
        string = "sysconfig.get_path(\"include\")"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testCaLLNLWithWhiteSpaces01(self):
        string = """Table(a)\n \n 
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

