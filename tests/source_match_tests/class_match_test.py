from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
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

string2 = """
class VQGANDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):
        super().__init__()
        chkpt = torch.load(model_path, map_location='cpu')
"""
string_c = """
class RMSNorm(nn.Module):
    \"\"\"
    From "Root Mean Square Layer Normalization" by https://arxiv.org/abs/1910.07467

    Adapted from flax.linen.LayerNorm
    \"\"\"

    epsilon: float = 1e-06
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    use_scale: bool = True
    scale_init: Any = jax.nn.initializers.ones

    @nn.compact
    def __call__(self, x):
        reduction_axes = (-1,)
        feature_axes = (-1,)

        rms_sq = self._compute_rms_sq(x, reduction_axes)

        return self._normalize(
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
        )

    def _compute_rms_sq(self, x, axes):
        x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
        rms_sq = jnp.mean(jax.lax.square(x), axes)
        return rms_sq

    def _normalize(
        self,
        mdl,
        x,
        rms_sq,
        reduction_axes,
        feature_axes,
        dtype,
        param_dtype,
        epsilon,
        use_scale,
        scale_init,
    ):
        reduction_axes = nn.normalization._canonicalize_axes(x.ndim, reduction_axes)
        feature_axes = nn.normalization._canonicalize_axes(x.ndim, feature_axes)
        stats_shape = list(x.shape)
        for axis in reduction_axes:
            stats_shape[axis] = 1
        rms_sq = rms_sq.reshape(stats_shape)
        feature_shape = [1] * x.ndim
        reduced_feature_shape = []
        for ax in feature_axes:
            feature_shape[ax] = x.shape[ax]
            reduced_feature_shape.append(x.shape[ax])
        mul = lax.rsqrt(rms_sq + epsilon)
        if use_scale:
            scale = mdl.param(
                "scale", scale_init, reduced_feature_shape, param_dtype
            ).reshape(feature_shape)
            mul *= scale
        y = mul * x
        return jnp.asarray(y, dtype)
"""
string_c1 = """
class RMSNorm(nn.Module):
    \"\"\"
    From "Root Mean Square Layer Normalization" by https://arxiv.org/abs/1910.07467

    Adapted from flax.linen.LayerNorm
    \"\"\"

    epsilon: float = 1e-06
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    use_scale: bool = True
    scale_init: Any = jax.nn.initializers.ones

    @nn.compact
    def __call__(self, x):
        reduction_axes = (-1,)
        feature_axes = (-1,)

        rms_sq = self._compute_rms_sq(x, reduction_axes)

        return self._normalize(
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
        )

    def _compute_rms_sq(self, x, axes):
        x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
        rms_sq = jnp.mean(jax.lax.square(x), axes)
        return rms_sq
"""
string_c2 = """
class RMSNorm(nn.Module):


    def _normalize(
        self,
        mdl,
        x,
        rms_sq,
        reduction_axes,
        feature_axes,
        dtype,
        param_dtype,
        epsilon,
        use_scale,
        scale_init,
    ):
        pass
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

    def testClassWithDocString(self):
        string = string_c
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testClassWithDocString1(self):
        string = string_c1
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testClassWithDocString12(self):
        string = string_c2
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testClassWithDecorators(self):
        string = "@dec1\n@dec2\nclass VQGANDiscriminator(nn.Module):\n  def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):\n     pass"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testClassWithDecorators2(self):
        string = "@dec1\n@dec2()\nclass VQGANDiscriminator(nn.Module):\n  def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):\n     pass"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
    def testClassWithDecorators3(self):
        string = "@dec1\n@dec2() #comment \nclass VQGANDiscriminator(nn.Module):\n  def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):\n     pass"
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)
