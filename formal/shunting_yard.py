__author__ = 'guydavidson'

# @ as a biconditional operator in one character
CONDITIONAL_OPERATORS = ('->', '=>', '-->', '==>')
CONDITIONAL = '>'

BICONDITIONAL_OPERATOS = ('<>', '<->', '<=>', '<-->', '<==>')
BICONDITIONAL = '@'

OPERATORS = ('&', 'v', '|', '>', '~', '@')
LEFT_PARENTHESIS = '('
RIGHT_PARENTHESIS = ')'
PARENTHESIS = (LEFT_PARENTHESIS, RIGHT_PARENTHESIS)

def pre_process_expression(expression):
    '''
    Helper function for shunting_yard - pre-processes operators into one character representations
    :param expression: The expression to be pre-processed
    :return:
    '''
    for op in CONDITIONAL_OPERATORS:
        expression.replace(op, CONDITIONAL)

    for op in BICONDITIONAL_OPERATOS:
        expression.replace(op, BICONDITIONAL)

    return expression

def peek(stack):
    '''
    Python's list doesn't feature a peek function. I
    :param stack: The stack to peek
    :return: The first element in the stack (not popped)
    '''
    if stack:
        return stack[len(stack) - 1]

    return None

def shunting_yard(expression):
    '''
    An adaptation of djikstra's shunting yard algorithm for SL WFFs:
    https://en.wikipedia.org/wiki/Shunting-yard_algorithm
    :param expression: The expression as a string
    :return: The expression in reverse polish notation (postfix)
    '''
    expression = pre_process_expression(expression)
    output_queue = []
    operator_stack = []

    for c in expression:
        # print 'Output:', output_queue, 'Operators:', operator_stack
        c = c.strip()

        if not c:
            continue

        # if a letter, append to output queue
        if c.isalpha():
            output_queue.append(c)

        # if c is an operator, manipulate the operator stack
        elif c in OPERATORS:
            while operator_stack and not peek(operator_stack) in PARENTHESIS:
                output_queue.append(operator_stack.pop())

            operator_stack.append(c)

        elif c == LEFT_PARENTHESIS:
            operator_stack.append(c)

        elif c == RIGHT_PARENTHESIS:
            while operator_stack and not peek(operator_stack) == LEFT_PARENTHESIS:
                output_queue.append(operator_stack.pop())

            if not operator_stack:
                print 'Mismatached parentheses!'
                return None

            operator_stack.pop()

    while operator_stack:
        if peek(operator_stack) in PARENTHESIS:
            print 'Mismatached parentheses!'
            return None

        output_queue.append(operator_stack.pop())

    return ''.join(output_queue)


def main():
    print shunting_yard('((~A & B) > (A v B))')
    print shunting_yard('((~(A) & B) > (A v B))')

if __name__ == '__main__':
    main()