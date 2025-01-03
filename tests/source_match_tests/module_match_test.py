from fun_with_ast.manipulate_node import create_node
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
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

module_24 = """
def _gen_invalid_iterdatapipe_msg(datapipe):
    return ("This iterator has been invalidated because another iterator has been created "
    f"from the same IterDataPipe: {_generate_iterdatapipe_msg(datapipe)}\\n"
    "This may be caused multiple references to the same IterDataPipe. We recommend "
    "using .fork() if that is necessary.")
"""
module_25 = """
def _process_batched_inputs(
    in_dims: in_dims_t, args: Tuple, func: Callable
) -> Tuple[int, List[Any], List[Any], TreeSpec]:
    if flat_in_dims is None:
        raise ValueError(
            f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): '
            f'in_dims is not compatible with the structure of `inputs`. '
            f'in_dims has structure {tree_flatten(in_dims)[1]} but inputs '
            f'has structure {args_spec}.')
"""
module_26 = """
def name(self):
    if isinstance(self.index, type):
        rep = f'__load_module("{self.index.module}").{self.index.qualname}'
        return f"___odict_getitem({self.base.name()}, {rep})"
"""

module_27 = """
def main() -> None:
    job_link = f"[job]({run_url})" if run_url is not None else "job"
    msg = (
        f"The {args.action} {job_link} was canceled. If "     
    )
"""
module_28 = """
def main() -> None:
    job_link = f"test" if run_url is not None else "job"
    msg = f"test2"         
"""

module_29 = """
def main() -> None:
    msg = (
        f"The {args.action} {job_link} was canceled. If "     
    )
"""
module_30 = """
def main():

        print(
            f"Type {t.__name__} had a minimum time of {10**6 * bench_min} us"
            f" and a standard deviation of {(10**6) * bench_std} us."
        )
"""
module_31 = """
def construct_name(fwd_bwd, test_name):
    return f"{suite_name}[{test_name}]:{'bwd' if bwd else 'fwd'}"
"""
module_32 = """
class HierarchicalModelAverager(averagers.ModelAverager):
    def init(self, period_group_size_dict=None, warmup_steps=0, process_group=None):
        if list(period_group_size_dict.values())[-1] != overall_group_size:
            raise ValueError(
                f"The last value in arg period_process_group_dict {list(period_group_size_dict.values())[-1]} "
                f"must be equal to the size of arg process_group {overall_group_size}."
        )
"""

module_33 = """
def forward(
    ctx, # pyre-ignore[2]: Parameter must be annotated.
    self: DT,
    ) -> DT:
    ctx.previous_placement = self.placements
"""
module_34 = """
def trace_cond(proxy_mode, func_overload, pred, true_fn, false_fn, operands):
    assert isinstance(
        operands, (list, tuple)
    ), "Cond operands must be a list or tuple of tensors"
"""

module_35 = """
def get_type_line(source):

    type_pattern = re.compile("# type:\\ ignore(\\[[a-zA-Z-]+\\])?$")

    if len(type_lines) == 0:
        wrong_type_pattern = re.compile("#[\t ]*type[\t ]*(?!: ignore(\\[.*\\])?$):")

"""
module_36 = """
def get_type_line(source):
    type_pattern = re.compile("# type:\\\\ ignore(\[[a-zA-Z-]+\\\\])?$")
"""

module_37 = 'def get_type_line(source):\n   type_pattern = re.compile("# type:\\ ignore(\[[a-zA-Z-]+\\])?$")'

module_38 = 'npt.assert_allclose(workspace.blobs[output], ref(), rtol=1e-3)'

module_39 = """def initialize(self) -> None:
    a.b(lambda: self.terminate())
"""

module_40 = """def init(
    self,
    strict=False,
    should_fallback_fn=lambda *_: False,
    prims_mode_cls=nullcontext,
    ):
        pass
"""

module_41 = """
class Journal:
    @classmethod
    def from_journal(cls, other: "Journal") -> "Journal":
        \"\"\"Creates a new journal by copying configuration and entries from
        another journal object\"\"\"
        new_journal = cls(other.name, **other.config)
"""
module_42 = """def new_entry(self, raw: str, date=None, sort: bool = True) -> Entry:
    raw = raw.replace("\\\\n ", "\\n").replace("\\\\n", "\\n")
"""

module_43 = """def argument():
    should_default = not is_out
    if isinstance(a, Argument):
        if True:
            pass
            return []
    elif False:
        if should_default:
            default = "{}"
    """


module_44 = """
should_default = not is_out
if isinstance(a, Argument):
    if True:
        pass
        a=1
elif False:
    if should_default:
        default = "{}"
    """

module_45 = """
def _get_source_code_to_add(self, node):
    for index, line in enumerate(lines):
        source_to_add += line + self.source_suffix.removesuffix('\\n') + f'_{index}' + '\\n'
    return source_to_add
"""

module_46 = """SHA256_REGEX = re.compile(r"\s*sha256\s*=\s*['\\"](?P<sha256>[a-zA-Z0-9]{64})['\\"]\s*,")"""
module_47 = """class _ExecOrderTracer:
  def init(self) -> None:
    self.exec_info: Optional[_ExecutionInfo] = None
  def patch_tracer():
    pass
"""
module_48 = """def get_sharding_prop_cache_info():
    return (
        DTensor._propagator.propagate_op_sharding.cache_info()  # type:ignore[attr-defined
    )"""
module_49 = """def ref_lambda_rank_loss():
    def log_sigm(x):
        return -np.log(1 + np.exp(-x))
    dy = np.zeros(n)
    loss = 0
"""
module_50 ="""choices=[
                    ("Bulgarian", "bg"),
                    (
                        "Old Church Slavonic, Church Slavic",
                        "cu",
                    ),
                    ("Malagasy", "mg"),
                ],
"""
module_51 ="""class TestDeviceAnalysis(JitTestCase):
    def zerodim_test_core(self, device_pairs):
        input_shapes = [
            ((1, 2, 2), (2, 2)),  # Different dim, non-zerodim
            ((1, 2, 2), ()),  # one zerodim
            ((), ()),  # both zerodim
        ]
"""

module_52 = """NSFusionElType = Union[
    Callable,  # call_function or call_module type, example: F.linear or nn.Conv2d
    str,  # call_method name, example: "dequantize"
    Tuple[str, Any],  # call_method name and first argument, example: ("to", torch.float16)
]
"""
module_53 = """
@staticmethod
def get_endpoint_priorities(secrets: dict,
                            gpt_endpoint_table_name: str,
                            model_name: str):
    with db_conn.cursor() as cursor:
        cursor.execute(f\"\"\"
                        SELECT endpoint
                        FROM {gpt_endpoint_table_name}
                        WHERE runtime = (SELECT MAX(runtime) FROM {gpt_endpoint_table_name} WHERE modelname='{model_name}') and modelname='{model_name}'
                        ORDER BY avgtime ASC
         \"\"\")"""


module_54 = """
def to_dot(self) -> str:
    return f\"\"\"\
digraph G {{
rankdir = LR;
node [shape=box];
{edges}
}}
\"\"\""""

module_55 = """def ts_lowering_body(schema: LazyIrSchema) -> str:
    return f\"\"\"
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve({len(emplace_arguments)});
    kwarguments.reserve({len(emplace_kwarg_values + emplace_kwarg_scalars)});
    size_t i = 0;
    {emplace_arguments_str}
    {emplace_kwarguments}
    torch::lazy::TSOpVector {schema.aten_name}_out = torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
    TORCH_CHECK_EQ({schema.aten_name}_out.size(), {len(schema.returns)});

    return {schema.aten_name}_out;
\"\"\"
"""

module_56 = """def foo():  
    return {'statusCode': 200, 'body': json.dumps(f'Deleted company {company_id} from session {session_id}'),
        'headers': {'Access-Control-Allow-Origin': '*'}}
"""
module_57 = """def get_cognito_token(client_id,
                      client_secret,
                      cognito_domain):
    # Input params are credentials for AWS Cognito's user pool
    token_request = {
        'url': f'https://{cognito_domain}/oauth2/token',
        'method': 'POST',
        'headers': {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        'data': {
            'grant_type': 'client_credentials'
        }
    }
"""

module_58 = """def get_cognito_token(client_id,
                      client_secret,
                      cognito_domain):
    # Input params are credentials for AWS Cognito's user pool
    token_request = {
        'auth': (client_id, client_secret),
        'headers': {
            'Content-Type': 'application/x-www-form-urlencoded'
        },

    }
"""

module_59 = """def _tensorpipe_construct_rpc_backend_options_handler(
    rpc_timeout,
    init_method,
    num_worker_threads=rpc_constants.DEFAULT_NUM_WORKER_THREADS,
    _transports=None,
    _channels=None,
    **kwargs
):
    from . import TensorPipeRpcBackendOptions

    return TensorPipeRpcBackendOptions(
        rpc_timeout=rpc_timeout,
        init_method=init_method,
        num_worker_threads=num_worker_threads,
        _transports=_transports,
        _channels=_channels,
    )
"""
module_60 = """def create_session(self, session_id, session_source, start_time, username, client_id, tenant_id, user_id):
        data = {'id': session_id, 'session_source': session_source, 'start_time': start_time,
                'username': username, 'client_id': client_id, 'tenant_id': tenant_id, 'user_id': user_id}
        logger.info(f'create_session: {session_id=}, {session_source=}, '
                    f'{start_time=}, {client_id=}, {tenant_id=}, {user_id=}')
        url = self._get_url('chatbot_session')
        response = None
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
        except Exception as e:
            if hasattr(e, 'response'):
                response_json = e.response.json()
                if response_json.get('id') == ['chatbot session with this id already exists.']:
                    logger.info(f'Session already exists {session_id=}')
                    raise SessionAlreadyExists()
                else:
                    logger.error(f'{response_json=}')
            logger.exception(f'Failed to create session {e}')
            raise FailedToCreateSession()
        logger.info(f'create_session returns: {session_id=} ')
        return response.json()
"""

module_61 = """def run(self, backtests: list, verbose=False):
    \"\"\"
    run backtests
    \"\"\"
    start_time = time.time()

    end_time = time.time()
    time_ellapsed = end_time - start_time
    if verbose: print(f"finished backtests in {time_ellapsed} sec.")
    return results
"""


module_62= """def has_url(text, strict_match_protocol=False):
    # remove markdown *
    text = text.replace('*','')

    # Anything that isn't a square closing bracket
    name_regex = "[^]]+"

    # http:// or https:// followed by anything but a closing paren
    url_in_markup_regex = "http[s]?://[^)]+"

    # first look for markup urls
    markup_regex = f"\[({name_regex})]\(\s*({url_in_markup_regex})\s*\)"

    urls = re.findall(markup_regex, text, re.IGNORECASE)

    if len(urls) > 0:
        replacechars = "[]()"

        for url in urls:
            text = re.sub(markup_regex, "", text)
            for ch in replacechars:
                text.replace(ch, '')

    # if none found, look for url without markup
    else:
        if strict_match_protocol:
            bare_url_regex = r"(https{0,1}:\/\/[A-Za-z0-9\-\._~:\/\?#\[\]@!\$&'\(\)\*\+\,;%=]+)"
        else:
            bare_url_regex = r"(?:[a-z]{3,9}:\/\/?[\-;:&=\+\$,\w]+?[a-z0-9\.\-]+|[\/a-z0-9]+\.|[\-;:&=\+\$,\w]+@)[a-z0-9\.\-]+(?:(?:\/[\+~%\/\.\w\-_]*)?\??[\-\+=&;%@\.\w_]*#?[\.\!\/\\\w]*)?"

        urls = re.findall(bare_url_regex, text, re.IGNORECASE)

        for i, url in enumerate(urls):
            urls[i] = [url, url]

    # # return what was found (could be just text)
    return urls, text
"""


module_63= """def has_url(text, strict_match_protocol=False):

        if strict_match_protocol:
            bare_url_regex = r"(https{0,1}:\/\/[A-Za-z0-9\-\._~:\/\?#\[\]@!\$&'\(\)\*\+\,;%=]+)"
        else:
            bare_url_regex = r"(?:[a-z]{3,9}:\/\/?[\-;:&=\+\$,\w]+?[a-z0-9\.\-]+|[\/a-z0-9]+\.|[\-;:&=\+\$,\w]+@)[a-z0-9\.\-]+(?:(?:\/[\+~%\/\.\w\-_]*)?\??[\-\+=&;%@\.\w_]*#?[\.\!\/\\\w]*)?"

        urls = re.findall(bare_url_regex, text, re.IGNORECASE)
"""
module_64 = """
bl_info = {
    "name": "AI Render - Stable Diffusion in Blender",
    "description": "Create amazing images using Stable Diffusion AI",
    "author": "Ben Rugg",
    "version": (1, 1, 0),
    "blender": (3, 0, 0), # this is the line that is failing
    "location": "Render Properties > AI Render",
    "warning": "",
    "doc_url": "https://github.com/benrugg/AI-Render#readme",
    "tracker_url": "https://github.com/benrugg/AI-Render/issues",
    "category": "Render",
}"""

module_65 = """
def generate(params, img_file, filename_prefix, props):

    # send the API request
    try:
        response = requests.post(
            timeout=request_timeout(),
        )
    except requests.exceptions.ReadTimeout:
        img_file.close()
        return operators.handle_error(
            f"The server timed out. Try again in a moment, or get help. [Get help with timeouts]({config.HELP_WITH_TIMEOUTS_URL})",
            "timeout",
        )
"""
module_66 = """
def parse_message_for_error(message):
    if '"Authorization" is missing' in message:
        return "Your DreamStudio API key is missing. Please enter it above.", "api_key"
    elif (
        "Incorrect API key" in message
        or "Unauthenticated" in message
        or "Unable to find corresponding account" in message
    ):
        return (
            f"Your DreamStudio API key is incorrect. Please find it on the DreamStudio website, and re-enter it above. [DreamStudio website]({config.DREAM_STUDIO_URL})",
            "api_key",
        )
"""
module_67 = """
def tree_unflatten(leaves: Iterable[Any], treespec: TreeSpec) -> PyTree:
    \"\"\"Given a list of values and a TreeSpec, builds a pytree.
    This is the inverse operation of `tree_flatten`.
    \"\"\"
    if not isinstance(treespec, TreeSpec):
        raise TypeError(
            f"tree_unflatten(leaves, treespec): Expected `treespec` to be "
            f"instance of TreeSpec but got item of type {type(treespec)}.",
        )
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

    def testFromInput9(self):
        string = """def create_bundled(d, outstream, include_files=False):
    \"\"\"Write the information to an open outstream\"\"\"
    outstream.write(f"Name: {c['Name']}\\n")
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

    def testFromInputModule5_1(self):
        string = """
def dot_product_attention_weights():
    attn_weights -= a(attn_weights)
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule5_2(self):
        string = """
def dot_product_attention_weights():
    attn_weights -= a
"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule5_3(self):
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

    def testFromInputModule24(self):
        string = module_24
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testFromInputModule25(self):
        string = module_25
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testFromInputModule26(self):
        string = module_26
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule27(self):
        string = module_27
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule28(self):
        string = module_28
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule29(self):
        string = module_29
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule30(self):
        string = module_30
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    def testFromInputModule31(self):
        string = module_31
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule32(self):
        string = module_32
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule33(self):
        string = module_33
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testFromInputModule34(self):
        string = module_34
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule35(self):
        string = module_35
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    def testFromInputModule36(self):
        string = module_36
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule37(self):
        string = module_37
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule38(self):
        string = module_38
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule39(self):
        string = module_39
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule40(self):
        string = module_40
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule41(self):
        string = module_41
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule42(self):
        string = module_42
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    # issue 284
    def testFromInputModule43(self):
        string = module_43
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testFromInputModule44(self):
        string = module_44
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)


    def testFromInputModule45(self):
        string = module_45
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testFromInputModule46(self):
        string = module_46
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testFromInputModule47(self):
        string = module_47
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule48(self):
        string = module_48
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule49(self):
        string = module_49
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule50(self):
        string = module_50
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule51(self):
        string = module_51
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule52(self):
        string = module_52
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    #@pytest.mark.skip("issue 329")
    def testFromInputModule53(self):
        string = module_53
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule54(self):
        string = module_54
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testFromInputModule55(self):
        string = module_55
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testFromInputModule56(self):
        string = module_56
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule57(self):
        string = module_57
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    import pytest
    def testFromInputModule58(self):
        string = module_58
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule59(self):
        string = module_59
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule60(self):
        string = module_60
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    @pytest.mark.skip("issue 359")
    def testFromInputModule61(self):
        string = module_61
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule62(self):
        string = module_62
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule63(self):
        string = module_63
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule64(self):
        string = module_64
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule65(self):
        string = module_65
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testFromInputModule66(self):
        string = module_66
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testFromInputModule67(self):
        string = module_67
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
