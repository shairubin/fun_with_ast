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
    def testSimpleRaiseWithString31_1(self):
        string = """raise ValueError(
                f"X"
                f"Y"
                f"Z"
            )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSimpleRaiseWithString31_2(self):
        string = """raise ValueError(
                f"X"
                f"Y"
                f"Z")"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSimpleRaiseWithString31_3(self):
        string = """raise ValueError(
                f"X"
                f"Y")"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testSimpleRaiseWithString32(self):
        string = """raise ValueError(
                f"X"
            )"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testRaiseWithMixedJstr(self):
        string = """raise ValueError('could not identify license file '\nf'for {root}') from None"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testRaiseWithMixedJstr2(self):
        string = """raise ValueError('could not identify license file '\n   f'for {root}') from None"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testRaiseWithNewLine(self):
        string = """raise x\n     """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testRaiseWithNewLine2(self):
        string = """raise x\n     """
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string, trim_suffix_spaces=True)

    def testRaiseWithQuotes(self):
        string = """raise ValueError(f"mode '{mode}' is not supported")"""
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string, trim_suffix_spaces=True)


    def testRaiseWithNewLine2_1(self):
        string = """raise x\n     """
        node = GetNodeFromInput(string, get_module=False)
        with pytest.raises(AssertionError): # 'Expr node does not support training white spaces'
            self._verify_match(node, string, trim_suffix_spaces=False)

    def testRaiseWithSplitJstr(self):
        string = """raise TypeError(
            f"tree_unflatten(leaves, treespec): Expected `treespec` to be "
            f"instance of TreeSpec but got item of type {type(treespec)}.",
        )
"""
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string)
    def testRaiseWithSplitJstr2(self):
        string = """raise TypeError(
            f"tree_unflatten(leaves, treespec): Expected `treespec` to be "
            f"instance of TreeSpec but got item of type {type(treespec)}."
        )
"""
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string)

    def testRaiseWithSplitJstr3(self):
        string = """raise TypeError(f"tree_unflatten(leaves, treespec): Expected `treespec` to be "
            f"instance of TreeSpec but got item of type {type(treespec)}.")"""
        node = GetNodeFromInput(string, get_module=False)
        self._verify_match(node, string)
