import random


def slots():
    # defining the symbols list is not efficient, but can be nice for printing purposes
    symbols = ["BAR", "BELL", "LEMON", "CHERRY"]
    slot1 = symbols[random.randint(0, 3)]
    slot2 = symbols[random.randint(0, 3)]
    slot3 = symbols[random.randint(0, 3)]
    if slot1 == slot2 == slot3 == "BAR":
        return 20
    if slot1 == slot2 == slot3 == "BELL":
        return 15
    if slot1 == slot2 == slot3 == "LEMON":
        return 5
    return [slot1, slot2, slot3].count("CHERRY")


def run_slots():
    loss = 0
    for i in range(1000):
        balance = 10
        stop = 0
        while balance > 0 and stop < 100:
            balance -= 1
            winnings = slots()
            balance += winnings
            stop += 1
        if balance <= 0:
            loss += 1
            print(balance, stop)
    print("Total losses: ", loss)


def dup_birthday(N):
    people = []
    for i in range(N):
        # For range of N, add a random birthday to the birthday list
        i_birthday = random.randint(0, 364)
        people.append(i_birthday)
    # Check if all elements are unique, if not, we have two same birthdays
    if len(set(people)) == len(people):
        return False
    else:
        return True


def birthday_part_1():
    iterations = 100
    success = 0
    bottom = 10
    upper = 50
    for N in range(bottom, upper + 1):
        match = 0
        for i in range(iterations):
            if dup_birthday(N):
                match += 1
        print("N =", N, "gives duplicates:", match / iterations)
        if match / iterations >= 0.5:
            success += 1
    return success


def run_birthday_part_1():
    success = birthday_part_1()
    print("We get duplicates a fraction of:", success / 40)


def check_birthdays(people):
    if len(set(people)) == 365:
        return True
    else:
        return False

def birthday_part_2():
    people = []
    finished = False
    while not finished:
        birthday = random.randint(0, 364)
        people.append(birthday)
        finished = check_birthdays(people)
    return len(people)


def run_birthday_part_2():
    total = 0
    iterations = 1000
    for i in range(iterations):
        result = birthday_part_2()
        total += result
    average = total / iterations
    print("Average after", iterations, "iterations: ", average)

run_birthday_part_2()