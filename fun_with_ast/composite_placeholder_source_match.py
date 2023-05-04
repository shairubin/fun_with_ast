import _ast

from fun_with_ast.source_matchers.exceptions_source_match import BadlySpecifiedTemplateError

from fun_with_ast.placeholder_source_match import Placeholder
from fun_with_ast.string_parser import StringParser
from fun_with_ast.node_placeholder_source_match import NodePlaceholder


class CompositePlaceholder(Placeholder):
    """Node which wraps one or more other nodes."""

    def Match(self, node, string):
        """Makes sure node.(self.field_name) is in string."""
        self.Validate(node)
        elements = self.GetElements(node)
        for element in elements:
            element.parent = node
        parser = StringParser(
            string, elements, starting_parens=self.starting_parens)
        return parser.GetMatchedText()

    def GetSource(self, node):
        return ''.join(
            element.GetSource(node) for element in self.GetElements(node))

    def Validate(self, unused_node):
        return True


class FieldPlaceholder(CompositePlaceholder):
    """Placeholder for a field."""

    def __init__(
            self, field_name, before_placeholder=None):
        super(FieldPlaceholder, self).__init__()
        self.field_name = field_name
        self.before_placeholder = before_placeholder

    def GetElements(self, node):
        if isinstance(node, _ast.Call) and self.field_name == 'kwargs':
            field_value = getattr(node, self.field_name, None)
        else:
            field_value = getattr(node, self.field_name)

        if not field_value:
            return []

        elements = []
        if self.before_placeholder:
            elements.append(self.before_placeholder)
        elements.append(NodePlaceholder(field_value))
        return elements

    def Match(self, node, string):
        return super(FieldPlaceholder, self).Match(node, string)

    def Validate(self, node):
        if isinstance(node, _ast.Call) and self.field_name == 'kwargs':
            field_value = getattr(node, self.field_name, None)
        else:
            field_value = getattr(node, self.field_name)
        if isinstance(field_value, (list, tuple)):
            raise BadlySpecifiedTemplateError(
                'Field {} of node {} is a list. please use a ListFieldPlaceholder'
                'instead of a FieldPlaceholder'.format(self.field_name, node))

    def __repr__(self):
        return 'FieldPlaceholder for field "{}"'.format(
            self.field_name)

