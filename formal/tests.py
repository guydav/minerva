from san_francisco_search import nodes

### class Node is defined for you in san_francisco_search.py
### insert breadthFirst and depthFirst functions here

# def breadthFirst(startingNode, soughtValue)...
# def depthFirst(startingNode, soughtValue)...


### Nodes are identified by an x and y coordinate that locates
### the intersection in San Francisco.
###
### Unlike in Jeremy Kun's code, the values of the nodes in this
### graph are pairs: (x_coordinate, y_coordinate)
###
### Both x_coordinate and y_coordinate are between 0 and 10000.
###
### To search, pull a node from the "nodes" dictionary like this:
###   nodes[(x_coordinate, y_coordinate)]
### from that node, search for another node's (x, y) value.

# These two lines print the result of searching from (1501, 4118)
# to (6173, 7065):

print "Breadth-first search:", breadthFirst(nodes[(1501,4118)], (6173,7065))
print "Depth-first search:", depthFirst(nodes[(1501,4118)], (6173,7065))
