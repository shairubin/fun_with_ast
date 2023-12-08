import unittest

import pytest

from fun_with_ast.manipulate_node.get_node_from_input import GetNodeFromInput
from tests.source_match_tests.base_test_utils import BaseTestUtils


class AnnotationMatcherTest(BaseTestUtils):

    def testAnnotationsFromSource(self):
        string = 'a: int'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAnnotationsOnDef(self):
        string = 'def vec2(x: T, y: T) -> List[T]:\n   return [x, y]'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAnnotationsOnDef2(self):
        string = 'def vec2(x: T1, y: T2) -> List[T]:\n   return [x, y]'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAnnotationsOnDef3(self):
        string = 'def vec2(x: T1, y: T2) -> List[T1,T2]:\n   return [x, y]'
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    @pytest.mark.skip(reason="issue 179")
    def testAnnotationsOnClass(self):
        string ="""class Artifact(object):
    roles: Optional[
        List[
            Literal[
                "userSpecifiedConfiguration",
                "debugOutputFile",
            ]
        ]
    ] = dataclasses.field(default=None, metadata={"schema_property_name": "roles"})
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    @pytest.mark.skip(reason="issue 179")
    def testAnnotationsOnClass2(self):
        string ="""class Artifact(object):
    roles: Optional[
        List[
            Literal[
                "userSpecifiedConfiguration",
                "debugOutputFile",
            ]
        ]
    ]
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAnnotationsOnClass3(self):
        string ="""class Artifact(object):
    roles: Optional[
        List[
            Literal[
                "userSpecifiedConfiguration",
            ]
        ]
    ]
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    @pytest.mark.skip(reason="issue 179")
    def testAnnotationsOnClass4(self):
        string ="""class Artifact(object):
    roles:  Literal[
                "userSpecifiedConfiguration",
                "debugOutputFile",
            ]
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)


    def testAnnotationsOnClass5(self):
        string ="""class Artifact(object):
    roles:  Literal["userSpecifiedConfiguration","debugOutputFile"]
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAnnotationsOnClass6(self):
        string ="""class Artifact(object):
    roles:  Literal["A","B"]
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    @pytest.mark.skip(reason="issue 179")
    def testAnnotationsOnClass7(self):
        string ="""class Artifact(object):
    roles:  Literal["A",
                    "B"]
"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAnnotationsOnClass8(self):
        string ="""roles:  Literal["A","B"]"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)

    def testAnnotationsOnClass9(self):
        string ="""roles:  Literal["A",
                                   "B"]"""
        node = GetNodeFromInput(string)
        self._verify_match(node, string)
