import unittest

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput

from fun_with_ast.manipulate_node import create_node
from fun_with_ast.source_matchers.matcher_resolver import GetDynamicMatcher
from tests.source_match_tests.base_test_utils import BaseTestUtils


class AsyncFunctionDefMatcherTest(BaseTestUtils):

    def testBodyFromString(self):
        string = 'async def test_fun():\n  a\n'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
    def testBodyFromString2(self):
        string = 'async def test_fun():\n  return 0'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBodyFromString3(self):
        string = 'async \t def test_fun():\n  return a or b'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testBodyFromString4(self):
        string = 'async     def test_fun():\n  return a or b\ndef test_fun2():\n  return d or c'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBodyFromString7(self):
        string = 'def test_fun():\n  return isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)\ndef test_fun2():\n  return d or c'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBodyFromString8(self):
        string = 'def test_fun():\n  return isPerfectSquare(5 * n * n + 4) or isPerfectSquare(5 * n * n - 4)\n#\nasync def test_fun2():\n  return d or c'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testBodyFromString9(self):
        string = 'async def test_fun():\n  (b)\n#\ndef test_fun2():\n  pass'
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

    def testInnerDef(self):
        string = """def convert_exception_to_response(get_response):

            if iscoroutinefunction(get_response):

                @wraps(get_response)
                async def inner(request):
                    try:
                        response = await get_response(request)
                    except Exception as exc:
                        response = await sync_to_async(
                            response_for_exception, thread_sensitive=False
                        )(request, exc)
                    return response

                return inner"""
        node = GetNodeFromInput(string, get_module=True)
        self._verify_match(node, string)

