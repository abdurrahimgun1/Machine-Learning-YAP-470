class LinearRegression:
    def __init__ (self, learning_rate=0.000005, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.m1 = 1
        self.m2 = 2
        self.b = 0
        self.train_losses = []
        self.train_r2_scores = []
        
    def loss_function(self, x_train, y_train, z_train):
        total_error = 0
        
        for i in range(len(z_train)):
             x1 = x_train[i]
             x2 = y_train[i]
             y = z_train[i]
             total_error += (y - (self.m1 * x1 + self.m2 * x2 + self.b)) ** 2

        loss = total_error / float(len(z_train))
        return loss
    
    def r2_score(self, y_true, y_pred):
        mean_y_true = sum(y_true) / len(y_true)
        total_sum_of_squares = sum((y - mean_y_true) ** 2 for y in y_true)
        residual_sum_of_squares = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))

        r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
        return r2
    
    def gradient_descent(self, m1_now, m2_now, b_now, x_train, y_train, z_train):
        m1_gradient = 0
        m2_gradient = 0
        b_gradient = 0
        
        n = len(z_train)
        
        for i in range(n):
            x1 = x_train[i]
            x2 = y_train[i]
            y = z_train[i]
            
            m1_gradient += -(2/n) * x1 * (y - (m1_now * x1 + m2_now * x2 + b_now))
            m2_gradient += -(2/n) * x2 * (y - (m1_now * x1 + m2_now * x2 + b_now))
            b_gradient += -(2/n) * (y - (m1_now * x1 + m2_now * x2 + b_now))
            
        m1 = m1_now - m1_gradient * self.learning_rate
        m2 = m2_now - m2_gradient * self.learning_rate
        b = b_now - b_gradient * self.learning_rate
        return m1, m2, b
    
    def fit(self, x_train, y_train, z_train):
        for i in range(self.epoch):
            if i % 50 == 0:
                print(f"Epoch: {i}")
                print(f"coeff 1: {self.m1}, coeff 2: {self.m2}, constant: {self.b}")
            self.train_losses.append(self.loss_function(x_train, y_train, z_train))
            self.train_r2_scores.append(self.r2_score(z_train, self.predict(x_train, y_train)))
            self.m1, self.m2, self.b = self.gradient_descent(self.m1, self.m2, self.b, x_train, y_train, z_train)
        
    def predict(self, x_test, y_test):
        predictions = []
        for i in range(len(x_test)):
            element = self.m1 * x_test[i] + self.m2 * y_test[i] + self.b
            predictions.append(element)
            
        return predictions