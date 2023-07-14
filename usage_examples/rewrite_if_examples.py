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
    body_index, location_in_body_index, original_if_source, source_to_add = _extract_example_data(example_input)
    _print_example_parameters(example_title, original_if_source)
    if_node = GetNodeFromInput(original_if_source)
    if_node_matcher = GetDynamicMatcher(if_node)
    if_node_matcher.do_match(original_if_source)
    node_to_add = GetNodeFromInput(source_to_add)
    if_manipulator = ManipulateIfNode(if_node, IfManipulatorConfig(body_index=body_index, location_in_body_index=location_in_body_index))
    if_manipulator.add_nodes([node_to_add])
    new_code = if_node_matcher.GetSource()
    print(bcolors.OKCYAN + "\nFun With AST Code:\n" + new_code + bcolors.ENDC)


def _print_example_parameters(example_title, original_if_source):
    print(bcolors.UNDERLINE + bcolors.FAIL + '\n'  + str(example_title) + bcolors.ENDC)
    print(bcolors.OKBLUE + "Developer Code:\n" + original_if_source + bcolors.ENDC)
    unparsed_code = ast.unparse(ast.parse(original_if_source))
    print(bcolors.OKGREEN + "\nAST Unparse Code:\n" + unparsed_code + bcolors.ENDC)


def _extract_example_data(example_input):
    original_if_source = example_input['example values']['original source code']
    source_to_add = example_input['example values']['source to add']
    body_index = example_input['example values']["manipulator config"]['body index']
    location_in_body_index = example_input['example values']["manipulator config"]['location in body']
    return body_index, location_in_body_index, original_if_source, source_to_add


if __name__ == "__main__":

    cwd = os.getcwd()
    if not cwd.endswith('examples'):
        raise ValueError("Please run this script from examples directory")

    with open('if_rewrite.json') as json_file:
        examples = json.load(json_file)
    for index, input in enumerate(examples):
        if_body_rewrite(f"Example number {index+1}: {input['example name']}", input)