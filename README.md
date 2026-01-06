# valence-with-tuned-llama
Some simple experients with a model fine-tuned from Llama 3.1 8B on Dutch sentences. 

Sorry, model is not available for public review

To run the server:

`furiosa-llm serve tuned-model --devices "npu:0"`

To run the client 

`python script.py --lang nl --runtime local`
