import numpy as np

class KalmanBoxTracker:
 
    count = 0

    def __init__(self, bbox):

        self.kf = self._init_kalman_filter()
        self.kf['x'][:4] = bbox
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def _init_kalman_filter(self):
        kf = {}
        kf['x'] = np.zeros((7, 1))
        kf['P'] = np.eye(7)
        kf['F'] = np.eye(7)
        kf['Q'] = np.eye(7) * 0.01
        kf['H'] = np.eye(4, 7)
        kf['R'] = np.eye(4)
        return kf

    def update(self, bbox):
        
        z = np.expand_dims(bbox, axis=-1)
        H = self.kf['H']
        x = self.kf['x']
        P = self.kf['P']
        R = self.kf['R']
        y = z - np.dot(H, x)
        S = np.dot(np.dot(H, P), H.T) + R
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        self.kf['x'] += np.dot(K, y)
        self.kf['P'] = np.dot(np.eye(7) - np.dot(K, H), P)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def predict(self):
      
        F = self.kf['F']
        self.kf['x'] = np.dot(F, self.kf['x'])
        self.kf['P'] = np.dot(np.dot(F, self.kf['P']), F.T) + self.kf['Q']
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf['x'][:4].reshape(-1)

    def get_state(self):
        return self.kf['x'][:4].reshape(-1)
