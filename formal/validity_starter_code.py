##############################################
# Validity checker for two variable arguments
# Starter Code
##############################################

# Auxiliary functions (not necessarily complete)
        
def foo_or(a,b):
    # returns a or b
    
def foo_and(a,b):
     # returns a and b   

def implication(a,b):
    # returns a->b
        
def id1(a,b):
    # returns a
    
def id2(a,b):
    # returns b
    
#############################################
# Function to run
    
def valid(f,g,h):
    # inputs: two-variable functions f,g,and h
    #   where f and g represent premises, and
    #   h represents a conclusion
    # returns True if the argument is valid
    # returns False otherwise

#############################################
# Examples of use

# tests premises: A or B, B, conclusion: A
print valid(foo_or,id2,id1)

# tests premises: A, A->B, conclusion: B
print valid(id1,implication,id2)

# tests premises: A or (A and B), B, conclusion: A
print valid(foo_or(id1,foo_and),id2,id1)

