import unittest

import pytest

from fun_with_ast.manipulate_node import create_node as create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError
from tests.source_match_tests.base_test_utils import BaseTestUtils

string1 = """
class VQGANDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):
        super().__init__()
        #
        # layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        # ndf_mult = 1
        # ndf_mult_prev = 1
        # for n in range(1, n_layers):  # gradually increase the number of filters
        #     ndf_mult_prev = ndf_mult
        #     ndf_mult = min(2 ** n, 8)
        #     layers += [
        #         nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
        #         nn.BatchNorm2d(ndf * ndf_mult),
        #         nn.LeakyReLU(0.2, True)
        #     ]
        #
        # ndf_mult_prev = ndf_mult
        # ndf_mult = min(2 ** n_layers, 8)
        #
        # layers += [
        #     nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(ndf * ndf_mult),
        #     nn.LeakyReLU(0.2, True)
        # ]
        #
        # layers += [
        #     nn.Conv2d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        # self.main = nn.Sequential(*layers)
        #
        #if model_path is not None:
        chkpt = torch.load(model_path, map_location='cpu')
            #if 'params_d' in chkpt:
            #    self.load_state_dict(torch.load(model_path, map_location='cpu')['params_d'])
            # elif 'params' in chkpt:
            #     self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
            # else:
            #     raise ValueError('Wrong params!')
"""
string3 = """
def __init__():
    chkpt = torch.load(model_path, map_location='cpu')
        #if 'params_d' in chkpt:
        #    self.load_state_dict(torch.load(model_path, map_location='cpu')['params_d'])
"""
string2 = """
class VQGANDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):
        super().__init__()
        chkpt = torch.load(model_path, map_location='cpu')
"""
class ClassMatcherTest(BaseTestUtils):

    def testClassSimple(self):
        string = "class FunWithAST:\n   pass"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testClassSimple2(self):
        string = "class FunWithAST:\n   def __init__(self):\n       pass"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testClassInheritance3(self):
        string = "class FunWithAST(ast):\n  def __init__(self):\n   pass\n  def forward(self, x):\n   return self.main(x)"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testClassSimple4(self):
        string = string1
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testClassSimple45(self):
        string = string2
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testClassSimple46(self):
        string = string3
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testClassSimple5(self):
        string = "class VQGANDiscriminator(nn.Module):\n  def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):\n     super().__init__()\n"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testClassSimple6(self):
        string = "class VQGANDiscriminator(nn.Module):\n  def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):\n     pass"
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testClassSimple7(self):
        string = "class VQGANDiscriminator(nn.Module):\n  def __init__(self):\n     a.b()\n"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
