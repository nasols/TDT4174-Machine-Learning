import os
import pickle

custom_storage_location = 'funkesje'

def save_variable(name, value):
    with open(os.path.join(custom_storage_location, name + '.pkl'), 'wb') as file:
        pickle.dump(value, file)

def load_variable(name):
    with open(os.path.join(custom_storage_location, name + '.pkl'), 'rb') as file:
        return pickle.load(file)