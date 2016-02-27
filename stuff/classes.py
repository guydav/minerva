people = [{'first_name': 'Colette',
           'middle_name': 'Marie',
           'last_name': 'Brown',
           'age': 18,
           'position': 'intern',},]

def print_name(person_dict):
    print person_dict['first_name'], \
        person_dict['last_name']

def print_salary(person_dict):
    if person_dict['position'] == 'intern':
        print 15
    else:
        print person_dict['salary']



class Person(object):
    def __init__(self, first_name, last_name, age, position,
                 salary):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.position = position
        self.salary = salary

    def __str__(self):
        return '%s %s is %d years old, works as a %s and makes $%f an hour' % (self.first_name,
                                                                        self.last_name, self.age, self.position, self.salary)

    def give_raise(self, raise_amount):
        if raise_amount <= 0:
            print "We're only giving raises here, buddy"

        self.salary += raise_amount


class Intern(Person):
    def __init__(self, first_name, last_name, age, position):
        Person.__init__(self, first_name, last_name, age,
                        'intern for ' + position, 15)
        self.hours_worked_this_week = 0
        self.money = 0

    def work(self, hours):
        self.hours_worked_this_week += hours

    def pay(self):
        self.money += self.hours_worked_this_week * self.salary
        self.hours_worked_this_week = 0

# colette = Person('Colette', 'Brown', 18, 'intern', 15)
# print colette
# colette.give_raise(5)
# print colette
#
# alex = Intern('Alexander', 'Whitney', 19, 'AA')
# print alex

class Point(object):
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return '({x}, {y}, {z})'.format(x=self.x, y=self.y,
                                        z=self.z)

    def distance(self, other_point):
        return ((self.x - other_point.x) ** 2 + \
               (self.y - other_point.y) ** 2 + \
                (self.z - other_point.z) ** 2) ** 0.5

    def is_on_line(self, m, b):
        # y = mx + b
        return (self.y == m * self.x + b)

    def is_on_plane(self, a, b, c, d):
        # Ax + By + Cz + d = 0
        return (a * self.x + b * self.y + c * self.z + d == 0)


p1 = Point(5, 6, 7)
p2 = Point(5, 6)
print p1, p2