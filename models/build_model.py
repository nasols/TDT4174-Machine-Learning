import sklearn.ensemble

class SK_Model() : 

    def __init__(self, name:str, model:sklearn.ensemble) -> None:
        self.name = name 
        self.model = model

        pass 



m = SK_Model("ada", sklearn.ensemble.AdaBoostRegressor)

