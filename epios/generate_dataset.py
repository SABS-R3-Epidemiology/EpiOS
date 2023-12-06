import random
import pandas as pd

# number of cells
num_cells = 10

# max num of microcells in a cell
max_num_microcells = 20

# max num of households in a microcell
max_num_households = 15

# max num of people per household
max_num_people = 8

# max age
max_age = 100

# min age
min_age = 1

table = []

for c in range(num_cells):

    num_microcells = random.randint(1, max_num_microcells)

    for m in range(num_microcells):

        num_households = random.randint(1, max_num_households)

        for h in range(num_households):

            num_people = random.randint(1, max_num_people)

            for p in range(num_people):

                id = str(c)+"."+str(m)+"."+str(h)+"."+str(p)

                age = random.randint(min_age, max_age)

                table.append([id, age, c, m, h])

headings = ["ID","age","cell","microcell","household"]

df = pd.DataFrame(table, columns=headings)

df.to_csv("./input/test_data.csv", encoding='utf-8', index=False)

print(df)
