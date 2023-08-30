import numpy as np
import pandas as pd

class ActionGenerator:
    def __init__(self):
        self.next_node = np.array([1, 2, 3, 4, 5])
        self.charge = np.array([0, 0.3, 0.5, 0.8])
        self.rest = np.array([0, 0.3, 0.6, 0.9, 1])

    def generate_actions(self):
        actions = []

        for node in self.next_node:
            if node in [1, 2, 3]:
                for ch in self.charge:
                    actions.append([node, ch, 0])
            elif node in [4, 5]:
                for r in self.rest:
                    actions.append([node, 0, r])

        return actions

    def save_to_csv(self, filename):
        actions = self.generate_actions()
        df = pd.DataFrame(actions, columns=['next_node', 'charge', 'rest'])
        print(df)
        df.to_csv(filename, index=False)


generator = ActionGenerator()
generator.save_to_csv("actions.csv")



