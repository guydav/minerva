import random


CHILD_ON_LEFT = True
CHILD_ON_RIGHT = False


class Node:
    def __init__(self, val):
        self.l_child = None
        self.r_child = None
        self.data = val

    def __repr__(self):
        return '<{data}>'.format(data=self.data)


def insert(root, node):
    if root is None:
        return node

    else:
        if root.data > node.data:
            if root.l_child is None:
                root.l_child = node
            else:
                insert(root.l_child, node)
        else:
            if root.r_child is None:
                root.r_child = node
            else:
                insert(root.r_child, node)

    return root


def search(root, value):
    if root is None:
        return None

    if root.data == value:
        return root

    if root.data > value:
        if root.l_child is not None:
            return search(root.l_child, value)

        else:
            return None

    # root.data < value
    if root.r_child is not None:
        return search(root.r_child, value)

    else:
        return None


def find_parent(root, node):
    if root is None or root == node:
        return None, None

    if node == root.l_child:
        return root, CHILD_ON_LEFT

    if node == root.r_child:
        return root, CHILD_ON_RIGHT


    if root.data > node.data:
        if root.l_child is not None:
            return find_parent(root.l_child, node)

        else:
            return None, None

    # root.data < value
    if root.r_child is not None:
        return find_parent(root.r_child, node)

    else:
        return None, None


def set_parent_new_child(parent, side, new_child):
    if CHILD_ON_RIGHT == side:
        parent.r_child = new_child
    else:
        parent.l_child = new_child


def find_successor(node):
    current = node.r_child
    if not current:
        return None, None

    parent = node
    while current.l_child is not None:
        parent = current
        current = current.l_child

    return current, parent


def delete(root, node):
    result = search(root, node.data)

    if result != node:  # node not in the tree
        raise ValueError("Node not found in the tree, assuming unique values.")

    parent, side = find_parent(root, node)
    no_l_child = node.l_child is None
    no_r_child = node.r_child is None

    if not parent:  # parent not found
        if node != root:
            raise ValueError("Failed to find parent for a non-root node in the tree. Aborting...")

        if no_l_child:
            return node.r_child

        elif not no_l_child and no_r_child:
            return node.l_child

        successor, successor_parent = find_successor(node)
        successor.l_child = node.l_child

        if successor != node.r_child:
            successor_parent.l_child = successor.r_child
            successor.r_child = node.r_child

        return successor

    else:  # parent found
        if no_l_child:
            set_parent_new_child(parent, side, node.r_child)

        elif not no_l_child and no_r_child:
            set_parent_new_child(parent, side, node.l_child)

        else:
            successor, successor_parent = find_successor(node)
            successor.l_child = node.l_child
            set_parent_new_child(parent, side, successor)

            if successor != node.r_child:
                successor_parent.l_child = successor.r_child
                successor.r_child = node.r_child

    return root


def in_order_print(root):
    if root is None:
        return

    if root.l_child:
        in_order_print(root.l_child)

    print root.data, '-',

    if root.r_child:
        in_order_print(root.r_child)


def print_tree(root, depth=0, is_right=None):
    if root is None:
        return

    if depth > 10:
        raise ValueError("Unable to comply, building in progress")

    if root.r_child:
        print_tree(root.r_child, depth + 1, CHILD_ON_RIGHT)

    if is_right is None:
        symbol = '--'
    elif is_right == CHILD_ON_RIGHT:
        symbol = '/'
    else:
        symbol = '\\'

    print '   ' * depth + ' {symbol} {data}'.format(symbol=symbol, data=root.data)

    if root.l_child:
        print_tree(root.l_child, depth + 1, CHILD_ON_LEFT)


def print_tree_plus(root):
    print_tree(root)
    # in_order_print(root)
    print
    print '=' * 20


def find_duplicates(root):
    queue = [root]

    while queue:
        current = queue.pop()
        successor = find_successor(current)[0]
        if successor is not None and current.data == successor.data:
            return True

        if current.l_child:
            queue.append(current.l_child)

        if current.r_child:
            queue.append(current.r_child)

    return False


current = None


def integrity(root):
    global current

    if root is None:
        return

    if root.l_child:
        integrity(root.l_child)

    if root.data < current:
        print root.data
        return False

    current = root.data

    if root.r_child:
        integrity(root.r_child)

    return True


def main():
    numbers = range(10)
    random.shuffle(numbers)
    root = None

    for n in numbers:
        print 'Inserting ', n
        root = insert(root, Node(n))
        print_tree_plus(root)

    for n in numbers:
        if search(root, n) is None:
            print n

    for n in range(11, 20):
        if search(root, n) is not None:
            print n

    random.shuffle(numbers)
    for n in numbers:
        print 'Deleting ', n
        root = delete(root, search(root, n))
        print_tree_plus(root)

if __name__ == '__main__':
    # for i in xrange(100):
    #     main()

    nums = range(20)
    root = None
    random.shuffle(nums)

    for n in nums:
        root = insert(root, Node(n))

    search(root, 0).l_child = Node(13)

    print integrity(root)

    # print find_duplicates(root)
    #
    # insert(root, Node(13))
    # print find_duplicates(root)












