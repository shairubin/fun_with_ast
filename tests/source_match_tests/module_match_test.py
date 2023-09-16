import unittest

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils



module6 = """ 
@nn.compact
def __call__(
self,
hidden_states,
attention_mask,
deterministic: bool = True,
output_attentions: bool = False,
output_hidden_states: bool = False,
return_dict: bool = True,
):
        layer_outputs = layer(
            self.config,
            dtype=self.dtype,
            add_norm=add_norm,
            use_scale=use_scale,
            name=f"FlaxBartEncoderLayer_{i}",
        )(
            hidden_states,
            attention_mask,
            output_attentions,
            deterministic,
        )
"""
module5 = """
def dot_product_attention_weights():
    attn_weights -= jax.nn.logsumexp(attn_weights, axis=-1, keepdims=True) """
module4 = """def dot_product_attention_weights(
    query: Any,
    key: Any,
    bias: Optional[Any] = None,
    mask: Optional[Any] = None,
    embed_pos: Optional[Any] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Any = jnp.float32,
    precision: PrecisionLike = None,
    sinkhorn_iters: int = 1,
    is_encoder: bool = False,
    tau=None,
):
    \"\"\"
    Computes dot-product attention weights given query and key.
    mask is included into the bias.

    Adapted from flax.linen.attention.dot_product_attention_weights"
    \"\"\"
    # add relative position
    if embed_pos is not None:
        attn_weights = attn_weights + embed_pos

    # normalize the attention weights
    if not is_encoder or sinkhorn_iters == 1:
        # sinkhorn does not work for causal (leaks info of future tokens into past)
        attn_weights = jax.nn.softmax(attn_weights).astype(dtype)
    else:
        # adapted from https://github.com/lucidrains/sinkhorn-transformer
        for i in range(sinkhorn_iters):
            # when causal, some attn_weights have been set to -inf through bias
            if i % 2 == 0:
                attn_weights -= jax.nn.logsumexp(attn_weights, axis=-1, keepdims=True)
            else:
                attn_weights -= jax.nn.logsumexp(attn_weights, axis=-2, keepdims=True)
            if mask is not None:
                attn_weights = jnp.where(mask, attn_weights, -jnp.inf)
        attn_weights = jnp.exp(attn_weights).astype(dtype)

"""

module3 = """class FlaxBartAttention(FlaxBartAttention):


    def __call__():

        if self.has_variable("cache", "cached_key"):
            causal_mask = lax.dynamic_slice(
                self.causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
"""


module2 = """class FlaxBartAttention(FlaxBartAttention):
    \"\"\"
    Edits:
    - causal mask is used only in decoder and considers image_length
    - scale attention heads per NormFormer paper
    \"\"\"


    def __call__():
        \"\"\"Input shape: Batch x Time x Channel\"\"\"

        # handle cache prepare causal attention mask
        if self.causal:
            query_length, key_length = query_states.shape[1], key_states.shape[1]
            if self.has_variable("cache", "cached_key"):
                causal_mask = lax.dynamic_slice(
                    self.causal_mask,
                    (0, 0, mask_shift, 0),
                    (1, 1, query_length, max_decoder_length),
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = jnp.broadcast_to(
                causal_mask, (batch_size,) + causal_mask.shape[1:]
            )
"""

module1 = """ 
def norm(type, *args, **kwargs):
    if True:
        raise ValueError(f"Unknown norm type {type}")

def dot_product_attention_weights(
):
    \"\"\"
    Adapted from flax.linen.attention.dot_product_attention_weights"
    \"\"\"

    pass
"""
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

    def testFromInputParantheses(self):
        string = """(a)"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputParantheses2(self):
        string = """(   (a) )  """
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule1(self):
        string = module1
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule2(self):
        string = module2
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule3(self):
        string = module3
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule4(self):
        string = module4
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    def testFromInputModule5(self):
        string = module5
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule51(self):
        string = """
def dot_product_attention_weights():
    attn_weights -= a(attn_weights)
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    def testFromInputModule52(self):
        string = """
def dot_product_attention_weights():
    attn_weights -= a
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule53(self):
        string = """
def dot_product_attention_weights():
    attn_weights -= a()
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule6(self):
        string = module6
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
