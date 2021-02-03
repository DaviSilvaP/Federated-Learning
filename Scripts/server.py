from fastapi import Request, FastAPI
from pydantic import BaseModel

from MLP import MLP
from dataProcessing import load_data
from federatedLearning import try_to_update


class Item(BaseModel):
    layer: int
    weights: list
    data_qtd: int


global_model = MLP()
global_model.build()

app = FastAPI()
clients_weights = {}


@app.get("/get_weights/")
async def get_weights(layer: int):
    return {layer: global_model.get_weights()[layer].tolist()}


@app.post("/send_weights/")
async def send_weights(item: Item, request: Request):
    if request.client.host in clients_weights:
        clients_weights[request.client.host][item.layer] = item.weights
    else:
        clients_weights[request.client.host] = {}
        clients_weights[request.client.host][item.layer] = item.weights
        clients_weights[request.client.host]["data_qtd"] = item.data_qtd
    weights = try_to_update(clients_weights, global_model.get_weights())
    if not isinstance(weights, bool):
        global_model.compile()
        global_model.set_weights(weights)
        print('=============== Updated ===============')
        _, _, X_test, y_test = load_data()
        test_loss, test_accuracy = global_model.test_model(X_test, y_test)
        print("Loss: {:.3f}, Accuracy: {:.3f}".format(test_loss,
                                                      test_accuracy))
    return {'Result': True}
