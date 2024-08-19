import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class SWaTegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        #nrows = 100000

        #data = pd.read_csv(data_path + '/train.csv',engine='python',on_bad_lines='skip',nrows=nrows)
        data = pd.read_csv(data_path + '/train.csv',engine='python',on_bad_lines='skip')
        data = data.drop(["Timestamp", "Normal/Attack"], axis=1)
        for i in list(data):
            data[i] = data[i].apply(lambda x: str(x).replace(",", "."))
        data = data.astype(float)

        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = pd.read_csv(data_path + '/test.csv',sep=';',engine='python',on_bad_lines='skip')
        labels = [float(label != 'Normal') for label in test_data["Normal/Attack"].values]
        test_data = test_data.drop(["Timestamp", "Normal/Attack"], axis=1)
        for i in list(test_data):
            test_data[i] = test_data[i].apply(lambda x: str(x).replace(",", "."))
        test_data = test_data.astype(float)

        test_data = self.scaler.transform(test_data)

        self.test = test_data
        self.train = data
        self.val = self.test
        self.test_labels = labels

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

    def save_processed_data(self, train_path, test_path ,labels_path):
        # Save the processed training data
        train_df = pd.DataFrame(self.train, columns=[f'Feature_{i}' for i in range(self.train.shape[1])])
        train_df.to_csv(train_path, index=False, header=False)
        print(f'Training data saved to {train_path}')

        # Save the processed testing data
        test_df = pd.DataFrame(self.test, columns=[f'Feature_{i}' for i in range(self.test.shape[1])])
        
        test_df.to_csv(test_path, index=False, header=False)
        print(f'Testing data saved to {test_path}')
        
        # Save the test labels
        labels_df = pd.DataFrame(self.test_labels, columns=['Label'])
        labels_df.to_csv(labels_path, index=False, header=False)
        print(f'Test labels saved to {labels_path}')

# Example usage:
data_path = 'testSWaT'
win_size = 100
step = 100
loader = SWaTegLoader(data_path, win_size, step, mode="train")
loader.save_processed_data('processed_data/processed_train.csv', 'processed_data/processed_test.csv', 'processed_data/test_labels.csv')
#loader.save_processed_data('processed_train.csv', 'processed_test.csv', 'test_labels.csv')
