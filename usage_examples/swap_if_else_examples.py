import ast
import json
import os

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from fun_with_ast.manipulate_node.if_manipulator import ManipulateIfNode, IfManipulatorConfig
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'




#def if_body_rewrite(example_number, original_if_source, source_to_add, body_index, location_in_body_index):
def if_body_rewrite(example_title, example_input):
    body_index, location_in_body_index, original_if_source = _extract_example_data(example_input)
    _print_example_parameters(example_title, original_if_source)
    if_node, if_node_matcher = _get_if_source(original_if_source)
    config = IfManipulatorConfig(body_index, location_in_body_index)
    manipulator = ManipulateIfNode(if_node, config)
    original_body_source = manipulator.get_body_orelse_source()
    print(bcolors.OKGREEN + "\nOriginal Body source:\n" + original_body_source + bcolors.ENDC)
    config.body_index = 1
    orig_else_source = manipulator.get_body_orelse_source()
    print(bcolors.OKGREEN + "\nOriginal Else source:\n" + orig_else_source + bcolors.ENDC)
    config.body_index = 0
    manipulator.rerplace_body(orig_else_source)
    new_body_source = manipulator.get_body_orelse_source()
    print(bcolors.OKGREEN + "\nNew Body source (old else body):\n" + new_body_source + bcolors.ENDC)
    config.body_index = 1
    manipulator.rerplace_body(original_body_source)
    new_else_source = manipulator.get_body_orelse_source()
    print(bcolors.OKGREEN + "\nNew Else source (old body):\n" + new_else_source + bcolors.ENDC)
    new_if = if_node_matcher.GetSource()
    print(bcolors.OKCYAN + "\nFun Wtih AST code after swap :\n" + new_if + bcolors.ENDC)
    unparsed_code = ast.unparse(if_node)
    print(bcolors.WARNING + "\nAST Unparse Code after swap:\n" + unparsed_code + bcolors.ENDC)


def _get_if_source(original_if_source):
    if_node = GetNodeFromInput(original_if_source)
    if_node_matcher = GetDynamicMatcher(if_node)
    if_node_matcher.do_match(original_if_source)
    if_node.matcher = if_node_matcher
    return if_node, if_node_matcher


def _print_example_parameters(example_title, original_if_source):
    print(bcolors.UNDERLINE + bcolors.FAIL + '\n'  + str(example_title) + bcolors.ENDC)
    print(bcolors.OKBLUE + "Developer Code:\n" + original_if_source + bcolors.ENDC)


def _extract_example_data(example_input):
    original_if_source = example_input['example values']['original source code']
    body_index = example_input['example values']["manipulator config"]['body index']
    location_in_body_index = example_input['example values']["manipulator config"]['location in body']
    return body_index, location_in_body_index, original_if_source


if __name__ == "__main__":

    cwd = os.getcwd()
    if not cwd.endswith('examples'):
        raise ValueError("Please run this script from examples directory")

    with open('if_else_swap.json') as json_file:
        examples = json.load(json_file)
    for index, input in enumerate(examples):
        if_body_rewrite(f"Example number {index+1}: {input['example name']}", input)