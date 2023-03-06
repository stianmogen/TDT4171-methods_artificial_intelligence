import numpy as np
from pathlib import Path
from typing import Tuple
import random



class Node:
    """ Node class used to build the decision tree"""
    def __init__(self):
        self.children = {}
        self.parent = None
        self.attribute = None
        self.value = None

    def classify(self, example):
        if self.value is not None:
            return self.value
        return self.children[example[self.attribute]].classify(example)



def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value

def entropy(q):
    # book p.680 define boolean entropy for probability q
    # B(q)=−(qlog2q + (1−q)log2(1−q)).
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """
    # if measure is random, return attribute from a random index
    if measure == "random":
        return attributes[random.randrange(0, len(attributes))]
    else:
        p, n = np.count_nonzero(examples[:, -1] == 1), np.count_nonzero(examples[:, -1] == 2)
        # B(q) = B(pk / (pk + nk))
        boolean_entropy = entropy(p / (p + n))
        importance = []
        for a in attributes:
            # defining the arrays of the two classifications.
            # note that this method does not work out the box if scaled up with more possible classes
            e_1 = np.array([e for e in examples if e[a] == 1])
            e_2 = np.array([e for e in examples if e[a] == 2])
            # count the occurences of 1
            p_1 = np.count_nonzero(e_1[:, -1] == 1)
            p_2 = np.count_nonzero(e_2[:, -1] == 1)

            # The information gain from the attribute test on A is the expected reduction in entropy: p.680
            gain = boolean_entropy - p_1 / (p + n) * entropy(p_1 / len(e_1)) - p_2 / (
                        p + n) * entropy(p_2 / len(e_2))
            importance.append(gain)

        # the resulting attribute is the one with index equal to that of greatest importance
        return attributes[np.argmax(importance)]


def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node’s parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """

    # Creates a node and links the node to its parent if the parent exists
    node = Node()
    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent

    # Empty check then return plurality value of parents

    if not examples.any():
        return plurality_value(parent_examples)

    # Check if all classifications are equal to that of the first element in classifications.
    # If true return node with value of given classification
    classifications = examples[:, -1]
    if np.all(classifications == classifications[0]):
        node.value = classifications[0]
        return node

    elif attributes.size == 0:
        return plurality_value(examples)

    else:
        A = importance(attributes, examples, measure)
        node.attribute = A
        for v in range(1, 3):
            exs = np.array([e for e in examples if e[A] == v])
            learn_decision_tree(examples=exs,
                                attributes=np.array([attr for attr in attributes if attr != A]),
                                parent_examples=examples,
                                parent=node,
                                branch_value=v,
                                measure=measure)
    return node



def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        print(example[:-1])
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
    epochs = 50

    # information_gain or random
    measures = ["random", "information_gain"]


    random_train_acc = []
    random_test_acc = []
    gain_train_acc = []
    gain_test_acc = []

    for e in range(epochs):
        for m in measures:
            tree = learn_decision_tree(examples=train,
                            attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                            parent_examples=None,
                            parent=None,
                            branch_value=None,
                            measure=m)
            if m == measures[0]:
                random_train_acc.append(accuracy(tree, train))
                random_test_acc.append(accuracy(tree, test))
            else:
                gain_train_acc.append(accuracy(tree, train))
                gain_test_acc.append(accuracy(tree, test))

    print("Running training for: ", epochs, " epochs:\n-------------------------")
    #print(f"Random Training Accuracy {accuracy(tree, train)}")
    #print(f"Random Test Accuracy {accuracy(tree, test)}")
    print("\n-------------------------")
    #print(f"Gain Training Accuracy {accuracy(tree, train)}")
    #print(f"Gain Test Accuracy {accuracy(tree, test)}")