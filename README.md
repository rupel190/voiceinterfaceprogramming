# Voice Interface Programming

- Change directory to where the Dockerfile resides
- Build and run the image: `docker build -t myimage . && docker run -p 5000:5000 myimage`
(Note that the built image remains)
- Interact with the REST API through a client such as cURL

## Interaction

On interaction the context is set which resets if an answer is not understood.
Therefore it is best to adhere to the communication scheme like in the following example.
```
curl localhost:5000/detectIntent -X POST -d "What can you do"
curl localhost:5000/detectIntent -X POST -d "Move an orange"
curl localhost:5000/detectIntent -X POST -d "CRATE"
curl localhost:5000/detectIntent -X POST -d "inside of it"
```

---
pip3 install spacy tqdm 
python3 -m spacy download en

