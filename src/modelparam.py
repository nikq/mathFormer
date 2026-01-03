class ModelParam:

    def __init__(self, modelType: str, NTokens: int):
        self.NTokens = NTokens
        # Initialize new attributes with default values
        self.NumExperts = 0
        self.ActiveExperts = 0
        self.setup(modelType)
    
    def __str__(self):
        return (f"ModelParam(NTokens={self.NTokens}, NInp={self.NInp}, NHead={self.NHead}, NHid={self.NHid}, "
                f"NLayers={self.NLayers}, Dropout={self.Dropout}, NumExperts={self.NumExperts}, ActiveExperts={self.ActiveExperts})")
    
    def setParam(self, modelType, NTokens, NInp, NHead, NHid, NLayers, Dropout, NumExperts=0, ActiveExperts=0):
        self.modelType = modelType
        self.NTokens = NTokens
        self.NInp = NInp
        self.NHead = NHead
        self.NHid = NHid
        self.NLayers = NLayers
        self.Dropout = Dropout
        self.NumExperts = NumExperts
        self.ActiveExperts = ActiveExperts

    def setup(self, modelType: str):
        if modelType == "tiny":
            self.setParam(
                modelType=modelType,
                NTokens=self.NTokens,
                NInp=32,
                NHead=2,
                NHid=128,
                NLayers=2,
                Dropout=0.05
            )
        elif modelType == "small":
            self.setParam(
                modelType=modelType,
                NTokens=self.NTokens,
                NInp=64,
                NHead=4,
                NHid=256,
                NLayers=4,
                Dropout=0.05
            )
        elif modelType == "medium":
            self.setParam(
                modelType=modelType,
                NTokens=self.NTokens,
                NInp=512,
                NHead=8,
                NHid=2048,
                NLayers=8,
                Dropout=0.05
            )
        elif modelType == "large":
            self.setParam(
                modelType=modelType,
                NTokens=self.NTokens,
                NInp=768,
                NHead=8,
                NHid=3072,
                NLayers=8,
                Dropout=0.05
            )
        else:
            raise ValueError(f"Unknown model type: {modelType}")

    def hash(self):
        return hash((self.NTokens, self.NInp, self.NHead, self.NHid, self.NLayers, self.Dropout))
    
    def type(self):
        return self.modelType