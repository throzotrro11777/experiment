class Config:
    def __init__(self):
        self.sigma = 25
        self.patch_size = 8
        self.search_window = 39
        self.max_match = 16
        self.Threshold_Hard3D = 2.7 * self.sigma
        self.Beta_Kaiser = 2.0