__author__ = 'guydavidson'

GREETING = 'Hello, Minerva!'

class Greeter(object):
    def __init__(self):
        pass

    def greet(self):
        print GREETING

def main():
    Greeter().greet()

if __name__ == '__main__':
    main()