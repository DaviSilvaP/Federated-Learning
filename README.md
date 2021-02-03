# Federate Learning Examples

The examples can be executed by using docker-compose.

***

## Steps
1. Install Docker
2. Install Docker-Compose
3. Run ```sudo docker-compose up``` or ```sudo docker-compose up -d```
4. For a better visualization, execute ```sudo docker logs <container-name> -f``` to each client and server.

***
# First Example

The first example is about a simple implementation using Federated Averaging Algorithm to improve the global model and then, the client's model as well.

You can increase the number of clients, change the MLP's structure and another configurations, testing to what extent the model can improve.
