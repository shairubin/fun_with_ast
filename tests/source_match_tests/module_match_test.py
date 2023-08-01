import unittest

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ModuleMatcherTest(BaseTestUtils):
    def testModuleBasicFailed(self):
        node = create_node.Module(create_node.FunctionDef(name='myfunc', body=[
            create_node.AugAssign('a', create_node.Add(), create_node.Name('c'))]))
        string = 'def myfunc():\n \t a += c\n'
        self._validate_match(node, string)

    def testModuleBasic(self):
        node = create_node.Module(create_node.FunctionDef(name='myfunc', body=[
            create_node.AugAssign('a', create_node.Add(), create_node.Name('c'))]))
        string = 'def myfunc():\n\ta += c\n'
        self._validate_match(node, string)

    def testBasicMatch(self):
        node = create_node.Module(create_node.Expr(create_node.Name('a')))
        string = 'a\n'
        self._validate_match(node, string)

    def testBasicMatchEndsWithComent(self):
        node = create_node.Module(create_node.Expr(create_node.Name('a')))
        string = '   a  \t  \n'
        self._validate_match(node, string)

    def testBasicMatchWithEmptyLines(self):
        node = create_node.Module(
            create_node.Expr(create_node.Name('a')),
            create_node.Expr(create_node.Name('b')))
        string = 'a\n\nb\n'
        self._validate_match(node, string)

    def testBasicMatchWithCommentLines(self):
        node = create_node.Module(
            create_node.Expr(create_node.Name('a')),
            create_node.Expr(create_node.Name('b')))
        string = 'a\n#blah\nb\n'
        self._validate_match(node, string)

    def _validate_match(self, node, string):
        matcher = GetDynamicMatcher(node)
        matcher.do_match(string)
        matched_string = matcher.GetSource()
        self.assertEqual(string, matched_string)
    def testFromInput(self):
        node = GetNodeFromInput('a=1')
        string = 'a=1'
        self._verify_match(node, string)
    def testFromInput2(self):
        node = GetNodeFromInput('a=1', get_module=True)
        string = 'a=1'
        self._verify_match(node, string)
    @pytest.mark.skip(reason="issue #41")
    def testTupleOperation(self):
        node = GetNodeFromInput('(1,)+tuple(ch_mult)', get_module=True)
        string = '(1,)+tuple(ch_mult)'
        self._verify_match(node, string)

    def testFromInput3(self):
        node = GetNodeFromInput("chkpt = torch.load(model_path, map_location='cpu')", get_module=True)
        string = "chkpt = torch.load(model_path, map_location='cpu')"
        self._verify_match(node, string)
    def testFromInput4(self):
        string = """
# @ARCH_REGISTRY.register()
# class VQAutoEncoder(nn.Module):
#     def __init__(self, img_size, nf, ch_mult, quantizer="nearest", res_blocks=2, attn_resolutions=None, codebook_size=1024, emb_dim=256,
#                  beta=0.25, gamma=0.99, decay=0.99, hidden_dim=128, num_layers=2, use_checkpoint=False, checkpoint_path=None):
chkpt = torch.load(model_path, map_location="cpu")
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInput5(self):
        string = """
# @ARCH_REGISTRY.register()
# class VQAutoEncoder(nn.Module):
chkpt = torch.load(model_path, map_location='cpu')
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInput6(self):
        string = """
#     def __init__("nearest")

chkpt = torch.load(model_path, map_location="cpu")
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInput7(self):
        string = """#     def a("s"):
a.b('cpu')
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInput8(self):
        string = """
a.b('cpu')
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
