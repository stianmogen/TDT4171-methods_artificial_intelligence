from collections import Counter
import math
import numpy as np
from pathlib import Path
from typing import Tuple



def learn_decision_tree(examples, attributes, target_attribute):
    # Count the number of positive and negative examples in the dataset
    count = Counter([example[target_attribute] for example in examples])

    # If all examples are positive, return the leaf node "True"
    if count['Yes'] == len(examples):
        return True

    # If all examples are negative, return the leaf node "False"
    if count['No'] == len(examples):
        return False

    # If there are no more attributes to split on, return the majority class
    if not attributes:
        return count.most_common(1)[0][0]

    # Choose the best attribute to split on
    best_attribute = choose_best_attribute(examples, attributes, target_attribute)

    # Create a new decision tree with the best attribute as the root
    tree = {best_attribute: {}}

    # Remove the best attribute from the attribute list
    attributes.remove(best_attribute)

    # For each possible value of the best attribute, create a new branch
    for value in get_values(examples, best_attribute):
        # Split the dataset based on the value of the best attribute
        new_examples = [example for example in examples if example[best_attribute] == value]

        # Recursively build the subtree on the new dataset
        subtree = learn_decision_tree(new_examples, attributes.copy(), target_attribute)

        # Add the new subtree to the root node
        tree[best_attribute][value] = subtree

    # Return the completed tree
    return tree


def entropy(examples):
    # Count the number of positive and negative examples in the dataset
    count = Counter([example['Class'] for example in examples])

    # Calculate the entropy of the dataset
    entropy = 0.0
    for label in count:
        probability = count[label] / len(examples)
        entropy -= probability * math.log2(probability)

    return entropy


def information_gain(examples, attribute, target_attribute):
    # Calculate the entropy of the original dataset
    original_entropy = entropy(examples)

    # Calculate the entropy of the dataset after splitting on the given attribute
    new_entropy = 0.0
    for value in get_values(examples, attribute):
        subset = [example for example in examples if example[attribute] == value]
        subset_entropy = entropy(subset)
        new_entropy += (len(subset) / len(examples)) * subset_entropy

    # Calculate the information gain of the attribute
    information_gain = original_entropy - new_entropy

    return information_gain


def choose_best_attribute(examples, attributes, target_attribute):
    # Initialize the best attribute and maximum information gain
    best_attribute = None
    max_gain = -1

    # Calculate the information gain of each attribute and choose the best one
    for attribute in attributes:
        gain = information_gain(examples, attribute, target_attribute)
        if gain > max_gain:
            best_attribute = attribute
            max_gain = gain

    return best_attribute


def get_values(examples, attribute):
    # Collect all unique values of the given attribute from the examples
    values = set()
    for example in examples:
        values.add(example[attribute])
    return values

def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        pred = tree.classify(example[:-1])
        correct += pred == example[-1]
    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test




if __name__ == '__main__':

    train, test = load_data()

    # information_gain or random
    measure = "information_gain"

    tree = learn_decision_tree(examples=train,
                    attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                    parent_examples=None,
                    parent=None,
                    branch_value=None,
                    measure=measure)

    print(f"Training Accuracy {accuracy(tree, train)}")
    print(f"Test Accuracy {accuracy(tree, test)}")