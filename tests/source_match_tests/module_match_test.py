

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils
module_19 = """
backend_test.exclude(r'(test_hardsigmoid'  # Does not support Hardsigmoid.
                     '|test_hardmax'  # Does not support Hardmax.
                     '|test_.*FLOAT16.*'  # Does not support Cast on Float16.
                     '|test_depthtospace.*'  # Does not support DepthToSpace.
                     '|test_reduce_l1.*'  # Does not support ReduceL1.
                     ')')
"""
module_18 = """
## @package get_python_cmake_flags
# Module scripts.get_python_cmake_flags
##############################################################################
# Use this script to find your preferred python installation.
##############################################################################
#
# You can use the following to build with your preferred version of python
# if your installation is not being properly detected by CMake.
#
#   mkdir -p build && cd build
#   cmake $(python ../scripts/get_python_cmake_flags.py) ..
#   make
#


import sys
import sysconfig

flags = [
    f"-DPYTHON_EXECUTABLE:FILEPATH={sys.executable}",
    f"-DPYTHON_INCLUDE_DIR={sysconfig.get_path('include')}",
]

print(" ".join(flags), end="")
"""
module_17 = """

def _translate_api(self, query, from_lang, to_lang):

    # Build request
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {
        "appid": f"{self.appid}",
        "q": f"{query}",
        "from": from_lang,
        "to": to_lang,
        "salt": f"{salt}",
        "sign": f"{sign}",
    }

    # Send request
    time.sleep(1 / self.qps)
"""
module_16 = """
def collect_license(current):
    try:
       ident = identify_license(license_file)
    except ValueError:
        raise ValueError('could not identify license file '
                        f'for {root}') from None
"""

module_15_1 = """
#!/usr/bin/env python3

def main() -> None:
    print(f"Exporting labels for {args.org}/{args.repo}")
    labels_file_name = "pytorch_labels.json"
    obj = boto3.resource("s3").Object("ossci-metrics", labels_file_name)

"""

module_15 = """
#!/usr/bin/env python3

def main() -> None:
    args = parse_args()
    print(f"Exporting labels for {args.org}/{args.repo}")
    labels_file_name = "pytorch_labels.json"
    obj = boto3.resource("s3").Object("ossci-metrics", labels_file_name)
    obj.put(Body=json.dumps(gh_get_labels(args.org, args.repo)).encode())

"""
module_14 = """
def get_source_lines_and_file(
    obj: Any,
    error_msg: Optional[str] = None,
) -> Tuple[List[str], int, Optional[str]]:
    \"\"\"
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lino, filename)
    \"\"\"
    filename = None  # in case getsourcefile throws
    try:
        filename = inspect.getsourcefile(obj)
        sourcelines, file_lineno = inspect.getsourcelines(obj)
    except OSError as e:
        msg = (
            f"Can't get source for {obj}. TorchScript requires source access in "
            "order to carry out compilation, make sure original .py files are "
            "available."
        )
        if error_msg:
            msg += "\\n" + error_msg
        raise OSError(msg) from e

    return sourcelines, file_lineno, filename
"""

module_13_1 = """def argumenttype_type(
    t: Type, *, mutable: bool, binds: ArgName, symint: bool
    ) -> NamedCType:
    pass
"""

module_13 = """def argumenttype_type(
    t: Type, *, mutable: bool, binds: ArgName, symint: bool
    ) -> NamedCType:
    return NamedCType(binds, MutRefCType(tensor_type))
"""

module_12_1= """
def throw_abstract_impl_not_imported_error(opname, module, context):
    if module in sys.modules:
        pass
    else:
        raise NotImplementedError(
            f"{opname}: operator. "
            f"The '{module}' "
            f"Python  {context}"
        )
"""

module_12 = """
def throw_abstract_impl_not_imported_error(opname, module, context):
    if module in sys.modules:
        pass
    else:
        raise NotImplementedError(
            f"{opname}: We could not find the abstract impl for this operator. "
            f"The operator specified that you may need to import the '{module}' "
            f"Python module to load the abstract impl. {context}"
        )
"""

module_11_1 = """

WRAPPER_SRC_NAMES = {
    # add additoonal:
    "A": "B",
    
    "C": "D"
}
"""

module_11 = """

WRAPPER_SRC_NAMES = {

    # add additoonal:
    "ALL_AVX512F_MICROKERNEL_SRCS": "defined(__i386__) || defined(__i686__) || defined(__x86_64__)",
    "PROD_SCALAR_MICROKERNEL_SRCS": "defined(__arm__)",

}

"""
module_10_1 = """
if line.startswith("SET") and line.split('(')[1].strip(' \\t\\n\\r'):
                name = x
"""


module_10 = """

def update_sources(xnnpack_path, cmakefile = "XNNPACK/CMakeLists.txt"):
    sources = collections.defaultdict(list)
    while i < len(lines):
            line = lines[i]


            if line.startswith("SET") and line.split('(')[1].strip(' \\t\\n\\r') in set(WRAPPER_SRC_NAMES.keys()) | set(SRC_NAMES):
                name = line.split('(')[1].strip(' \\t\\n\\r')
            else:
                i += 1
    return sources
"""
module9_1 = """


def _generate_continue(self, sequences, model, tokenizer):
                generated_sequences[i * self.create_n + ii].replace(" ", "").replace("\\n", "")

"""
module9 = """



class SentenceContinue:

    @paddle.no_grad()
    def _generate_continue(self, sequences, model, tokenizer):
        for i, sequence in enumerate(sequences):
            augmented_sequence = []
            for ii in range(self.create_n):
                continue_sequence = (
                    generated_sequences[i * self.create_n + ii].replace(" ", "").replace("\\n", "").replace("\\t", "")
                )
                augmented_sequence.append(sequence + continue_sequence)
            augmented_sequences.append(augmented_sequence)
        return augmented_sequences

"""
module8 = """
def generate(
            **model_kwargs,
    ):

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            if model_kwargs.get("encoder_outputs") is None:
                model_kwargs_input = dict(model_kwargs)
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                    input_ids,
                    params,
                    {"attention_mask": attention_mask, **model_kwargs_input},
                )
            # prepare decoder_input_ids for generation
            input_ids = (
                    jnp.ones((input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id
            )
"""

module7 = \
"""
class DalleBart(PretrainedFromWandbMixin, FlaxBartForConditionalGeneration):

    \"\"\"
    Edits:
    - renamed from FlaxBartForConditionalGeneration
    - uses custom FlaxBartForConditionalGenerationModule
    - no bias in decode method
    - custom prepare_inputs_for_generation using "max_length - 1" to avoid issues
      related to position embedding during model.generate()
    - custom generate method to allow super conditions
    - num_params property
    - unscan function
    \"\"\"

    module_class = FlaxBartForConditionalGenerationModule
    config_class = DalleBartConfig

    def unscan(self, params):
        if self.config.use_scan:
            self.config.use_scan = False
            params = flatten_dict(params)
            scanned_keys = [k for k in params.keys() if "layers" in k]
            for k in scanned_keys:
                v = params[k]
                name_idx = k.index("layers") + 1
                for i in range(len(v)):
                    new_k = (
                        *k[:name_idx],
                        f"{k[name_idx][:-1]}_{i}",
                        *k[name_idx + 1 :],
                    )
                    params[new_k] = v[i]
                del params[k]
            params = unflatten_dict(params)
        return params


"""

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
module_20 = """def exposed_in(module):
    def wrapper(fn):
        fn.__module__ = module
        return fn
    return wrapper

argnums_t = Union[int, Tuple[int, ...]]
"""

module_22="""
def skip_init(module_cls, *args, **kwargs):
    if 'device' not in inspect.signature(module_cls).parameters:
        raise RuntimeError('Module must support a \\'device\\' arg to skip initialization')
"""
module_23="""def __getattr__(name):
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
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
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testFromInputModule6(self):
        string = module6
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    @pytest.mark.skip(reason="issue #195")
    def testFromInputModule7(self):
        string = module7
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    def testFromInputModule8(self):
        string = module8
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule9(self):
        string = module9
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testFromInputModule9_1(self):
        string = module9_1
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule10(self):
        string = module_10
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule10_1(self):
        string = module_10_1
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule11(self):
        string = module_11
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)




    def testFromInputModule11_1(self):
        string = module_11_1
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule12(self):
        string = module_12
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule12_1(self):
        string = module_12_1
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule13(self):
        string = module_13
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testFromInputModule13_1(self):
        string = module_13_1
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule14(self):
        string = module_14
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    def testFromInputModule15(self):
        string = module_15
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule15_1(self):
        string = module_15_1
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule16(self):
        string = module_16
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule17(self):
        string = module_17
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule18(self):
        string = module_18
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule19(self):
        string = module_19
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule20(self):
        string = module_20
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule22(self):
        string = module_22
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule23(self):
        string = module_23
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
