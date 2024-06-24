class DecisionTreeClassifier:
    tree = None
    def __init__ (self, max_depth=5):
        self.max_depth = max_depth
        
    def isPure(self, data):
        if len(list(set([sublist[-1] for sublist in data]))) == 1:
            return True
        else:
            return False
        
    def classify(self, data):
        elementCounts = {}
        y_train = [sublist[-1] for sublist in data]
        
        for item in y_train:
            count = y_train.count(item)
            elementCounts[item] = count

        temp = sorted(elementCounts.items(), key=lambda x: x[0])
        counts = [sublist[-1] for sublist in temp]
        value = max(counts)
        classification = None
        for key, val in elementCounts.items():
            if val == value:
                classification = key
                break
        return classification
    
    def potential_splits(self, data):
        potential_splits = {}
        number_of_columns = None
        y_train = [sublist[-1] for sublist in data]
        x_train = [row[:-1] for row in data]
        
        if isinstance(x_train[0], list):
            number_of_columns = len(x_train[0])
            for column_index in range(number_of_columns):
                potential_splits[column_index] = []
                values = [sublist[column_index] for sublist in x_train]
                unique_values = list(set(values))
                unique_values.sort()
                for index in range(len(unique_values)):
                    if index != 0:
                        current = unique_values[index]
                        previous = unique_values[index - 1]
                        potential = (current + previous) / 2
                        
                        potential_splits[column_index].append(potential)
        else:
            number_of_columns = 1
            
            for column_index in range(number_of_columns):
                potential_splits[column_index] = []
                values = x_train
                unique_values = list(set(values))
                unique_values.sort()
                for index in range(len(unique_values)):
                    if index != 0:
                        current = unique_values[index]
                        previous = unique_values[index - 1]
                        potential = (current + previous) / 2
                        
                        potential_splits[column_index].append(potential)
                    
        return potential_splits
    
    def split(self, data, split_column, split_value):
        dataBelow = [k for k in data if k[split_column] <= split_value]
        dataAbove = [l for l in data if l[split_column] > split_value]
    
        return dataBelow, dataAbove
    
    def calculate_gini(self, data):
        species_column = [sublist[-1] for sublist in data]
        elementCounts = {}

        for item in species_column:
            count = species_column.count(item)
            elementCounts[item] = count

        temp = sorted(elementCounts.items(), key=lambda x: x[0])
        counts = [sublist[-1] for sublist in temp]
        gini = 1 - sum([x**2 for x in [y / sum(counts) for y in counts]])
        return gini
    
    def calculate_overall_gini(self, dataBelow, dataAbove):
        totalDataPoints = len(dataBelow) + len(dataAbove)
        pDataBelow = len(dataBelow) / totalDataPoints
        pDataAbove = len(dataAbove) / totalDataPoints

        overallGini = (pDataBelow * self.calculate_gini(dataBelow) + pDataAbove * self.calculate_gini(dataAbove))
        
        return overallGini
    
    def best_split(self, data):
        potentialSplits = self.potential_splits(data)
        
        overallGini = 1000
        for columnIndex in potentialSplits:
            for value in potentialSplits[columnIndex]:
                dataBelow, dataAbove = self.split(data, columnIndex, value)
                currentGini = self.calculate_overall_gini(dataBelow, dataAbove)

                if currentGini < overallGini:
                    overallGini = currentGini
                    bestSplitColumn = columnIndex
                    bestSplitValue = value
        return bestSplitColumn, bestSplitValue
    
    def fit(self, x_train, y_train, counter=0, min_samples=2):
        data = []
        for item in x_train:
            data.append(list(item))
            
        for x, y in zip(data, y_train):
            x.append(y)
            
        self.tree = self.algorithm(data)
    
    def algorithm(self, data, counter=0, min_samples=2):
        if counter == 0:
            global COLUMNS
            COLUMNS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm",
                       "PetalWidthCm"]
                
        if self.isPure(data) or (len(data) < min_samples) or (counter == self.max_depth):
            classification = self.classify(data)
            return classification
        
        else:
            counter += 1
            splitColumn, splitValue = self.best_split(data)
            dataBelow, dataAbove = self.split(data, splitColumn, splitValue)
            feature = COLUMNS[splitColumn]
            question = "{} <= {}".format(feature, splitValue)
            tree = {question: []}
            
            answer1 = self.algorithm(data=dataBelow, counter=counter, min_samples=min_samples)
            answer2 = self.algorithm(data=dataAbove, counter=counter, min_samples=min_samples)
            
            if answer1 == answer2:
                tree = answer1
            else:
                tree[question].append(answer1)
                tree[question].append(answer2)
            
            return tree
        
    def predict(self, x_test):
        results = []
        for i in x_test:
            results.append(self.classify_test(i, self.tree))
        return results
        
    def classify_test(self, instance, tree):
        question = list(tree.keys())[0]
        feature, comparison, value = question.split()
        columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm",
                       "PetalWidthCm"]
        
        if instance[columns.index(feature)] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
            
        if not isinstance(answer, dict):
            return answer
        else:
            residualTree = answer
            return self.classify_test(instance, residualTree)