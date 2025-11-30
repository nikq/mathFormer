class ModelParam:

    def __init__(self, modelType: str, NTokens: int):
        self.NTokens = NTokens
        self.setup(modelType)
    
    def __repr__(self):
        return (f"ModelParam(NTokens={self.NTokens}, NInp={self.NInp}, NHead={self.NHead}, "
                f"NHid={self.NHid}, NLayers={self.NLayers}, Dropout={self.Dropout})")
    
    def setParam(self, modelType, NTokens, NInp, NHead, NHid, NLayers, Dropout):
        self.modelType = modelType
        self.NTokens = NTokens
        self.NInp = NInp
        self.NHead = NHead
        self.NHid = NHid
        self.NLayers = NLayers
        self.Dropout = Dropout

    def setup(self, modelType: str):
        if modelType == "tiny":
            self.setParam(
                modelType=modelType,
                NTokens=self.NTokens,
                NInp=32,
                NHead=2,
                NHid=64,
                NLayers=2,
                Dropout=0.05
            )
        elif modelType == "small":
            self.setParam(
                modelType=modelType,
                NTokens=self.NTokens,
                NInp=64,
                NHead=4,
                NHid=128,
                NLayers=4,
                Dropout=0.05
            )
        elif modelType == "medium":
            self.setParam(
                modelType=modelType,
                NTokens=self.NTokens,
                NInp=1024,
                NHead=8,
                NHid=2048,
                NLayers=8,
                Dropout=0.05
            )
        else:
            raise ValueError(f"Unknown model type: {modelType}")

    def hash(self):
        return hash((self.NTokens, self.NInp, self.NHead, self.NHid, self.NLayers, self.Dropout))
    
    def type(self):
        return self.modelType