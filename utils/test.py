__author__ = 'guydavidson'



def main_function():
    # load first situation
    result = run_situation(description, options, true_option)

    if result:
        # pass to next situation

    else:
        # return to beginning

def run_situation(description, options, true_option):
    print description

    for option in options:
        print option

    user_choice = raw_input('Please choose... ')

    return user_choice == true_option

