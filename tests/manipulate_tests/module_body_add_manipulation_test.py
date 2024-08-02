import ast
import sys

import pytest

from fun_with_ast.get_source import GetSource
from fun_with_ast.manipulate_node.body_manipulator import BodyManipulator
from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.manipulate_tests.base_test_utils_manipulate import bcolors

input_legend = ('inject-source', 'location', 'original-if', 'expected', 'match-expected', 'injected_second_source')


@pytest.fixture(params=[
    ('a.b()\n', 0, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a.b()\n   a=1', True, 'b.a()'),  # 0
    ('a.c()\n', 0, 'if (c.d()):\n   a=1\n', 'if (c.d()):\n   a.c()\n   a=1\n', True, 'print(test)'),  # 1
    ('a=44\n', 0, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=44\n   a=1', True, 'pass # comment'),  # 2
    ("s='fun_with_ast'\n", 0, 'if (c.d()):\n   a=1', "if (c.d()):\n   s='fun_with_ast'\n   a=1", True, 'raise(test)'),
    # 3
    ("", 0, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=1', True, 'a.x()'),  # 4
    ('a.b()\n', 1, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=1\n   a.b()\n', True, 'False'),  # 5
    ('a.c()\n', 1, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=1\n   a.c()\n', True, '   # only comment'),  # 6
    ("", 1, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=1', True, 'pass'),  # 7
    ('a.bb()\n', 0, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a.b()\n   a=1\n', False, 'a.b.c'),  # 8
    ('a.c()\n', 0, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a.b()\n   a=1\n', False, 'a=99'),  # 9
    ('a=45\n', 0, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=44\n\n   a=1\n', False, ''),
    ("s='fun_with_ast2'\n", 0, 'if (c.d()):\n   a=1', "if (c.d()):\n   s='fun_with_ast2'\n   a=1", True, 'raise'),
    ("", 0, 'if (c.d()):\n   a=1', 'if (c.d()):\n    a=1', False, 'a<b'),  # 12
    ('a.b()\n', 1, 'if (c.d()):\n   a=1', 'if (c.x()):\n   a=1\n   a.b()\n', False, 'pass'),  # 13
    ('a.c()\n', 1, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=1\n   a.b()\n', False, 'pass'),  # 14
    ("", 1, 'if (c.d()):\n   a=1', 'if (c.d()):\n   a=2', False, ''),  # 15
    ('a.b()\n', 0, 'if (c.d()):\n #comment-line\n   a=1',  # 16
     'if (c.d()):\n   a.b()\n #comment-line\n   a=1', True, '   # another comment'),
    ('a.b()\n', 1, 'if (c.d()):\n #comment-line\n   a=1',  # 17
     'if (c.d()):\n #comment-line\n   a.b()\n   a=1', True, 'pass'),
    ('a.b()\n', 1, 'if (c.d()):\n #comment-line\n   a=1',  # 18
     'if (c.d()):\n #comment----line\n   a.b()\n   a=1\n', False, 'pass'),
    ('a.b()\n', 0, 'if (c.d()):\n\n   a=1',  # 19
     'if (c.d()):\n   a.b()\n\n   a=1', True, 'pass'),  # TODO: this is currently a weird behavior in which
    # empty line is counted as a line
    ('a.b()\n', 1, 'if (c.d()):\n\n   a=1',  # 20
     'if (c.d()):\n\n   a.b()\n   a=1', True, 'pass'),  # TODO: this is currently a weird behavior in
    # which empty line is counted as a line #24
    ('a.b()\n', 0, 'if (c.d()):\n   a=1\n # comment',  # 21
     'if (c.d()):\n   a.b()\n   a=1\n # comment', True, 'pass'),
    ('a.b()\n', 0, 'if (c.d()):\n   a=1\n   b=1',  # 22
     'if (c.d()):\n   a.b()\n   a=1\n   b=1', True, 'pass'),
    ('a.b()\n', 0, 'if (c.d()):\n   if True:\n      a=111\n   b=11\n   c=12\n',  # 23
    'if (c.d()):\n   a.b()\n   if True:\n      a=111\n   b=11\n   c=12\n', True, 'pass'),
    ('a.b()\n', 0, """if _utils.is_sparse(A):
        if len(A.shape) != 2:
            raise ValueError("pca_lowrank input is expected to be 2-dimensional tensor")
        c = torch.sparse.sum(A, dim=(-2,)) / m
""",  # 24
     """if _utils.is_sparse(A):
        a.b()
        if len(A.shape) != 2:
            raise ValueError("pca_lowrank input is expected to be 2-dimensional tensor")
        c = torch.sparse.sum(A, dim=(-2,)) / m
"""
     , True, 'b.a()'),

    ('a.b()\n', 0, """if True:
            if False:
                raise ValueError("test")
            c = a
""",  # 25
                """if True:
            a.b()
            if False:
                raise ValueError("test")
            c = a
""", True, ''),

    ('a.b()\n', 0, """if True:
        if False:
            a=1
        c = a
""",  # 26
     """if True:
        a.b()
        if False:
            a=1
        c = a
""" , True, ''),

('a.b()\n', 0, """if first_card == 100:
        self.direction = -1
        self.can_add_card = self.can_add_card_down
""",  # 27
     """if first_card == 100:
        a.b()
        self.direction = -1
        self.can_add_card = self.can_add_card_down
""" , True, ''),
('a.b()\n', 0, """if first_card == 100:
        self.direction = -1
        self.can_add_card = self.can_add_card_down
""",  # 28
     """if first_card == 100:
        a.b()
        self.direction = -1
        self.can_add_card = self.can_add_card_down)
""" , False, ''),


('injected.code()\n', 0, """""" , True, ''),
])
def module_source(request):
    yield request.param

module_1 = """class foo():
  def __init__(self):
    a=1
    def bar(self):
        b=2
    c=4"""
module_2 = """class foo():
    def bar():
        with a as p:
            a=1
            b=2
            c=3"""

module_3 = """def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output"""
module_4= """def _tensorpipe_construct_rpc_backend_options_handler(
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
@pytest.fixture(params=[
    ({"source": module_1, "injected_source": "d=10\n",
      "inject_into_body":"module_node.body" , "inject_to_indexes": [(0,0,0),(1,6,0)]}),
    ({"source": module_1, "injected_source": "d=11\n",
      "inject_into_body": "module_node.body[0].body", "inject_to_indexes": [(0, 1,2), (1,6,2)]}),
    ({"source": module_1, "injected_source": "d=12\n",
      "inject_into_body": "module_node.body[0].body[0].body",
      "inject_to_indexes": [(0, 2, 4), (1, 3, 4), (2,5,4), (3,6,4)]}),
    ({"source": module_1, "injected_source": "d=13\n",
      "inject_into_body": "module_node.body[0].body[0].body[1].body",
      "inject_to_indexes": [(0, 4, 8), (1,5,8)]}),
    ({"source": module_2, "injected_source": "d=14\n",
      "inject_into_body": "module_node.body[0].body[0].body",
      "inject_to_indexes": [(0, 2, 8), (1, 6, 8)]}),
    ({"source": module_2, "injected_source": "d=14\n",
      "inject_into_body": "module_node.body[0].body[0].body",
      "inject_to_indexes": [(0, 2, 8), (1, 6, 8)]}),
    ({"source": module_2, "injected_source": "d=15\n",
      "inject_into_body": "module_node.body[0].body[0].body[0].body",
      "inject_to_indexes": [(0, 3, 12), (1, 4, 12), (2, 5, 12) , (3, 6, 12)]}),
    ({"source": module_3, "injected_source": "d=16\n",
      "inject_into_body": "module_node.body[0].body",
      "inject_to_indexes": [(0, 1, 8), (1, 2, 8)],
      "double_injection": [((1, 5), (2, 8), (6, 8))] }),
    ({"source": module_3, "injected_source": "d=16\n",
      "inject_into_body": "module_node.body[0].body",
      "inject_to_indexes": [(0, 1, 8), (1, 2, 8)],
      "double_injection": [((1, 1), (2, 8), (3, 8))]}),
    ({"source": module_3, "injected_source": "d=16\n",
      "inject_into_body": "module_node.body[0].body",
      "inject_to_indexes": [(0, 1, 8), (1, 2, 8)],
      "double_injection": [((3, 9), (4, 8), (9, 8))]}),
    # ({"source": module_4, "injected_source": "d=16\n",
    #   "inject_into_body": "module_node.body[0].body",
    #   "inject_to_indexes": [(0, 1, 8)]}),

])
def module_source_2(request):
    yield request.param

def _get_tuple_as_dict(in_tuple):
    return dict(zip(input_legend, in_tuple))

class TestIfManupulation:

    def test_Module_Body_Manipulation(self, module_source, capsys):
        original_module_source = 'a=1\na=2\nif c.d():\n   b=1\n   b=2\nelse:\n   c=1\n   c=2'
        for index in [0,1,2,3]:
            module_node, injected_node = self._create_nodes(capsys, module_source[0], original_module_source,
                                                            is_module=True)
            manipulator = BodyManipulator(module_node.body)
            manipulator.inject_node([injected_node], index)
            print("\n insert in index:" + str(index))
            composed_source = self._source_after_composition(module_node, capsys)
            composed_source_lines = composed_source.split('\n')
            if index in [0,1,2] and module_source[0]:
                assert composed_source_lines[index] + '\n' == module_source[0]
            if index ==  3 and module_source[0]:
                assert composed_source_lines[-2] + '\n' == module_source[0]
            ast.parse(composed_source)

    def test_dynamic_Module_Body_Manipulation(self, module_source_2, capsys):
            source = module_source_2['source']
            ast.parse(source)
            injected_source = module_source_2['injected_source']
            for index in module_source_2['inject_to_indexes']:
                module_node, injected_node = self._create_nodes(capsys, injected_source, source, is_module=True)
                inject_to_body = eval(module_source_2['inject_into_body'])
                manipulator = BodyManipulator(inject_to_body)
                manipulator.inject_node(injected_node.body, index[0])
                print("\n insert in index:" + str(index))
                composed_source = self._source_after_composition(module_node, capsys)
                composed_source_lines = composed_source.split('\n')
                assert composed_source_lines[index[1]] + "\n" == ' '*index[2] +injected_source
                ast.parse(composed_source)

    def test_dynamic_Module_Body_double_Manipulation(self, module_source_2, capsys):
            source = module_source_2['source']
            ast.parse(source)
            injected_source = module_source_2['injected_source']
            if not module_source_2.get("double_injection", None):
                pytest.skip("double injection not defined")
            for index in module_source_2['double_injection']:
                module_node, injected_node = self._create_nodes(capsys, injected_source, source, is_module=True)
                inject_to_body = eval(module_source_2['inject_into_body'])
                manipulator = BodyManipulator(inject_to_body)
                manipulator.inject_node(injected_node.body, index[0][0])
                print("\n insert in index:" + str(index[0][0]))
                composed_source = self._source_after_composition(module_node, capsys)
                manipulator.inject_node(injected_node.body, index[0][1])
                print("\n insert in index:" + str(index[0][1]))
                composed_source = self._source_after_composition(module_node, capsys)
                composed_source_lines = composed_source.split('\n')
                assert composed_source_lines[index[1][0]] + "\n" == ' '*index[1][1] +injected_source
                assert composed_source_lines[index[2][0]] + "\n" == ' '*index[2][1] +injected_source
                ast.parse(composed_source)

    def _create_nodes(self, capsys, injected_source, original_source, injected_second_source='', is_module=False):
        self._capture_source(capsys, original_source, 'original source:', bcolors.OKBLUE)
        if_node = self._create_if_node(original_source, is_module)
        injected_node, injected_node_source = self._create_injected_node(injected_source, injected_second_source)
        return if_node, injected_node

    def _create_injected_node(self, injected_source, injected_second_source):
        injected_node_source = injected_source
        #module = False
        #if injected_second_source:
        injected_node_source +=  injected_second_source
        #    module=True
        injected_node = GetNodeFromInput(injected_node_source, get_module=True)
        injected_node_matcher = GetDynamicMatcher(injected_node)
        injected_node_matcher.do_match(injected_node_source)
        source_from_matcher = injected_node_matcher.GetSource()
        assert source_from_matcher == injected_node_source
        source_from_get_source = GetSource(injected_node, assume_no_indent=True)
        assert source_from_get_source == injected_node_source
        return injected_node, injected_node_source

    def _create_if_node(self, original_source, is_module):
        node = GetNodeFromInput(original_source, get_module=is_module)
        node_matcher = GetDynamicMatcher(node)
        node_matcher.do_match(original_source)
        if_node_source = node_matcher.GetSource()
        assert if_node_source == original_source
        # if_node.matcher = if_node_matcher
        return node

    def _capture_source(self, capsys, source, title, color, ignore_ident=False):
        if not ignore_ident:
            try:
                compile(source, '<string>', mode='exec')
            except Exception as e:
                assert False, f'Error in source compilation'
        print(color + '\n' + title + '\n' + source + bcolors.ENDC)
        out, _ = capsys.readouterr()
        sys.stdout.write(out)

    def _source_after_composition(self, if_node, capsys):
        composed_source = GetSource(if_node, assume_no_indent=True)
        self._capture_source(capsys, composed_source, 'Modified source', bcolors.OKCYAN)
        return composed_source
