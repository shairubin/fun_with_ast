import unittest

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.base_matcher import SourceMatcher
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from fun_with_ast.source_matchers.exceptions import BadlySpecifiedTemplateError, EmptyStackException

from fun_with_ast.manipulate_node import create_node
from tests.source_match_tests.base_test_utils import BaseTestUtils

@pytest.fixture(autouse=True)
def run_around_tests(): # TODO not very smart global variable
    SourceMatcher.parentheses_stack.reset()
    yield

class DictMatcherTest(BaseTestUtils):

    def testComplexDictMatch(self):
        string = """deepnet_gain = {
            "encoder": {
                "alpha": lambda config: 0.81
                * (config.encoder_layers**4 * config.decoder_layers) ** 0.0625,
                "beta": lambda config: 0.87
                * (config.encoder_layers**4 * config.decoder_layers) ** -0.0625,
            },
            "decoder": {
                "alpha": lambda config: (3 * config.decoder_layers) ** 0.25,
                "beta": lambda config: (12 * config.decoder_layers) ** -0.25,
            },
    }"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testComplexDictMatch2(self):
        string = """deepnet_gain = {
            "encoder": {
                "alpha": lambda config: 0.81
                * (config.encoder_layers**4 * config.decoder_layers) ** 0.0625,
                "beta": lambda config: 0.87
                * (config.encoder_layers**4 * config.decoder_layers) ** -0.0625,
            },
    }"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testComplexDictMatch21(self):
        string = """deepnet_gain = {
            "encoder": {
                "alpha": lambda config: 0.81
                * (config.encoder_layers**4 * config.decoder_layers) ** 0.0625
            }
    }"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testComplexDictMatch22(self):
        string = """deepnet_gain = {
                "alpha": lambda config: 0.81
                * (config.encoder_layers**4 * config.decoder_layers) ** 0.0625,
    }"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBasicDictMatch28(self):
        string = """subln_gain = {
    "encoder": lambda config: math.sqrt(
        1.0
        / 3.0
        * math.log(3 * config.decoder_layers)
        * math.log(2 * config.encoder_layers),
        ),
    "decoder": lambda config: math.sqrt(math.log(3 * config.decoder_layers))
}"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBasicDictMatch27(self):
        string = """subln_gain = {
    "encoder": lambda config: math.sqrt(math.log(x,),),
    "decoder": lambda config: math.sqrt(math.log(3 * config.decoder_layers))
}"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBasicDictMatch271(self):
        string = """subln_gain = {
    "encoder": lambda config: math.sqrt(math.log(x,)),
    "decoder": lambda config: math.sqrt(math.log(3 * config.decoder_layers))
}"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBasicDictMatch272(self):
        string = """subln_gain = {
    "encoder": lambda config: math.sqrt(math.log(x,)),
    "decoder": lambda config: math.sqrt(math.log(3 * config.decoder_layers),)
}"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBasicDictMatch273(self):
        string = """subln_gain = {
    "encoder": lambda config: math.sqrt(math.log(x,)    ,     ),
    "decoder": lambda config: math.sqrt(math.log(3 * config.decoder_layers),
    ),
}"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBasicDictMatch2(self):
        string = """subln_gain = {
    "encoder": lambda config: math.sqrt(
        1.0
        / 3.0
        * math.log(3 * config.decoder_layers)
        * math.log(2 * config.encoder_layers)
    ),
    "decoder": lambda config: math.sqrt(math.log(3 * config.decoder_layers))
}"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBasicDictMatch25(self):
        string = """subln_gain = {
    "encoder": lambda config: math.sqrt(1.0),
    "decoder": 7
}"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBasicDictMatch26(self):
        string = """subln_gain = {
    "encoder": sqrt(1.0,),
    "decoder": 7
}"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBasicDictMatch24(self):
        string = """subln_gain = {
    "encoder": lambda config: math.sqrt(
        1.0
        / 3.0
        * math.log(3 * config.decoder_layers)
        * math.log(2 * config.encoder_layers)
    ),
}"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBasicDictMatch15(self):
        string = """{
    "E":  C, # comment 
    "D":  B, #comment
}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicDictMatch16(self):
        string = """{
    "E":  C,
    "D":  B
}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicDictMatch16(self):
        string = """{
    "E":  C, #comment2
    "D":  B
}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicDictMatch3(self):
        string = """{
    "E":  C(1.0), # comment1
    "D":  B,
}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicDictMatch32(self):
        string = """{
    "E":  C(1.0),
    "D":  B,
}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicDictMatch31(self):
        string = """{
    "E":  C(1.0,), # comment1
}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicDictMatch34(self):
        string = """{
    "E":  C(1.0,) # comment1
}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicDictMatch33(self):
        string = """{
    "E":  C(1.0) # comment1
}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicDictMatch4(self):
        string = """{
    "encoder":  sqrt(1.0),
    "decoder":  math
}"""

    def testAssignFromSourceWithDictAsKwargs(self):
        string = """{**mmm}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSourceWithDictAsKwargs2(self):
        string = """{"k": a , **mmm}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testAssignFromSourceWithDictAsKwargs3(self):
        string = """{"k": a , **mmm, **kk}"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
