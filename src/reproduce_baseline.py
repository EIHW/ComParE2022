import os

os.system('python src/main_modelling.py -m hr -t baseline -tID EIHW -sID 1')
os.system('python src/main_modelling.py -m steps -t baseline -tID EIHW -sID 2')
os.system('python src/main_modelling.py -m xyz -t baseline -tID EIHW -sID 3')

os.system('python src/main_modelling.py -m hr steps -t baseline -tID EIHW -sID 4')
os.system('python src/main_modelling.py -m hr xyz -t baseline -tID EIHW -sID 5')
os.system('python src/main_modelling.py -m steps xyz -t baseline -tID EIHW -sID 6')

os.system('python src/main_modelling.py -m hr steps xyz -t baseline -tID EIHW -sID 7')
