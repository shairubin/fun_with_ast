import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class ImportMatcherTest(BaseTestUtils):

    def testBasicMatchImport(self):
        string = 'import a\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testBasicMatchImport2(self):
        string = 'import a, b\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testBasicMatchImport3(self):
        string = 'import \t a,      b\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBasicMatchImport4(self):
        string = 'import a,b\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    @pytest.skip('issue #102')
    def testImportListWithParentheses(self):
        string = 'import (a,b)\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    @pytest.skip('issue #102')
    def testImportListWithParenthese2(self):
        string = """from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
)
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

