def train(model, x_train, y_train):
    model.fit(x_train, y_train)


def evaluate(model, x_test, y_test):
    return model.evaluate(x_test, y_test)


def save(model, path):
    model = model.export_model()
    model.save(path)
