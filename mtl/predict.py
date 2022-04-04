from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

def test_model(file_name, mtl, data_y, data_x, num_of_sentiments = True, ):

    if mtl == False:
        if num_of_sentiments == True:
            class_names = ['1.0', '2.0', '3.0', '4.0', '5.0']
        else:
            class_names = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0']

        predictor = load_model(file_name)
        predictions = predictor.predict(data_x)
        predictions = np.argmax(predictions, axis=1)
        predictions = [class_names[pred] for pred in predictions]

        data_test = [str(x) for x in data_y.sentiment.tolist()]

        print("Accuracy: {:.2f}%".format(accuracy_score(data_test, predictions) * 100))
        print("\nF1 Score: {:.2f}".format(f1_score(data_test, predictions, average='micro') * 100))

    else:
        if num_of_sentiments == True:
            class_names = ['1.0', '2.0', '3.0', '4.0', '5.0']
        else:
            class_names = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0']

        predictor = load_model(file_name)
        predictions = predictor.predict(data_x)[0]
        print(predictions[0].shape)
        predictions = np.argmax(predictions, axis=1)
        predictions = [class_names[pred] for pred in predictions]

        data_test = [str(x) for x in data_y.sentiment.tolist()]

        print("Accuracy: {:.2f}%".format(accuracy_score(data_test, predictions) * 100))
        print("\nF1 Score: {:.2f}".format(f1_score(data_test, predictions, average='micro') * 100))