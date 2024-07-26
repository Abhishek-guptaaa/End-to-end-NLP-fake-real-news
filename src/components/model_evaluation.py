from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, train_seq, y_train, test_seq, y_test):
    y_pred_train = (model.predict(train_seq) > 0.5).astype("int32")
    y_pred_test = (model.predict(test_seq) > 0.5).astype("int32")
    
    print(f'Train Accuracy: {accuracy_score(y_train, y_pred_train) * 100:.2f} %')
    print(f'Test Accuracy: {accuracy_score(y_test, y_pred_test) * 100:.2f} %')
    
    print(f'Classification Report (Train): \n\n{classification_report(y_train, y_pred_train)}')
    print('-----------------------------------------------------')
    print(f'Classification Report (Test): \n\n{classification_report(y_test, y_pred_test)}')
